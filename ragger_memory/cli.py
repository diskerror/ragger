"""
Command-line interface for Ragger Memory
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Optional

from .memory import RaggerMemory
from .mcp_server import run_mcp_server
from .server import run_server, DEFAULT_PORT
from .config import DEFAULT_SEARCH_LIMIT, DEFAULT_MIN_SCORE, DEFAULT_CHUNK_SIZE

logger = logging.getLogger(__name__)


def import_file(
    memory: RaggerMemory,
    filepath: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    metadata: Optional[dict] = None
):
    """
    Import a file into memory with paragraph-aware chunking
    
    Args:
        memory: RaggerMemory instance
        filepath: Path to file
        chunk_size: Max characters per chunk (default: 500)
        metadata: Additional metadata to attach
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    text = path.read_text()

    # Strip embedded base64 image data (noise for text embeddings)
    text = re.sub(r'!\[[^\]]*\]\(data:[^)]+\)', '', text)  # ![alt](data:...)
    text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '', text)  # bare data URIs

    # Keep relative image references as-is (e.g. O_artifacts/image_000007_00fb...png)
    # Path rewriting happens at query time, not import time — see TOOLS.md

    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse extra blank lines left behind

    source_meta = {"source": str(path), "filename": path.name, "title": path.stem}
    if metadata:
        source_meta.update(metadata)
    
    # Split on paragraph boundaries with heading-aware chunking.
    #
    # Headings are tracked as a breadcrumb trail (e.g. "Intro - Setup - Step 1").
    # Each chunk gets:
    #   - The current heading(s) prepended to the text (improves embeddings)
    #   - A "section" metadata field with the breadcrumb trail
    #
    # Consecutive headings accumulate and attach to the next body paragraph.
    # Every body paragraph under a heading inherits that heading context.

    raw_paragraphs = text.split('\n\n')

    def _heading_level(line: str) -> int:
        """Return heading level (1-6) or 0 if not a heading."""
        m = re.match(r'^(#{1,6})\s', line)
        return len(m.group(1)) if m else 0

    def _heading_text(line: str) -> str:
        """Strip '#' prefix from heading line."""
        return re.sub(r'^#{1,6}\s+', '', line).strip()

    # State: breadcrumb stack tracks the current section hierarchy
    # Each entry is (level, heading_text)
    heading_stack: list[tuple[int, str]] = []
    pending_headings: list[str] = []  # raw heading lines waiting for body

    def _current_section() -> str:
        """Build breadcrumb from heading stack."""
        return ' - '.join(h[1] for h in heading_stack)

    def _current_heading_block() -> str:
        """Build the heading lines to prepend to chunk text."""
        if pending_headings:
            return '\n\n'.join(pending_headings)
        # Use the deepest heading in the stack
        if heading_stack:
            level, txt = heading_stack[-1]
            return '#' * level + ' ' + txt
        return ''

    # Two-pass: first build (text, section) tuples, then chunk them
    # Each tuple is a body paragraph with its heading context
    annotated: list[tuple[str, str]] = []  # (text_with_heading, section_breadcrumb)

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        level = _heading_level(para)
        if level > 0:
            # Update the heading stack: pop anything at this level or deeper
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, _heading_text(para)))
            pending_headings.append(para)
        else:
            # Body paragraph — attach pending headings and section breadcrumb
            heading_block = '\n\n'.join(pending_headings) if pending_headings else _current_heading_block()
            section = _current_section()
            if heading_block:
                full_text = heading_block + '\n\n' + para
            else:
                full_text = para
            annotated.append((full_text, section))
            pending_headings = []

    # Trailing headings with no body
    if pending_headings:
        section = _current_section()
        annotated.append(('\n\n'.join(pending_headings), section))
        pending_headings = []

    # Now chunk the annotated paragraphs, preserving section metadata.
    # When paragraphs are merged into one chunk, use the section from
    # the first paragraph in the chunk.
    chunks: list[tuple[str, str]] = []  # (chunk_text, section)
    current = ""
    current_section = ""

    for full_text, section in annotated:
        # If this paragraph alone exceeds chunk_size, split on sentences
        if len(full_text) > chunk_size:
            if current:
                chunks.append((current.strip(), current_section))
                current = ""
                current_section = ""
            # Split on sentence boundaries (. ! ? followed by space or end)
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            for sentence in sentences:
                candidate = current + (' ' if current else '') + sentence
                if len(candidate) > chunk_size and current:
                    chunks.append((current.strip(), section))
                    current = sentence
                else:
                    current = candidate
            current_section = section
        else:
            candidate = current + ('\n\n' if current else '') + full_text
            if len(candidate) > chunk_size and current:
                chunks.append((current.strip(), current_section))
                current = full_text
                current_section = section
            else:
                if not current:
                    current_section = section
                current = candidate
    if current.strip():
        chunks.append((current.strip(), current_section))

    # Filter out empty chunks
    chunks = [(t, s) for t, s in chunks if t]
    
    print(f"Importing {len(chunks)} chunks from {path.name}...")
    for i, (chunk_text, section) in enumerate(chunks, 1):
        chunk_meta = source_meta.copy()
        chunk_meta.update({"chunk": i, "total_chunks": len(chunks)})
        if section:
            chunk_meta["section"] = section
        memory_id = memory.store(chunk_text, chunk_meta)
        print(f"  Chunk {i}/{len(chunks)}: {memory_id}")
    print(f"✓ Imported {len(chunks)} chunks")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Ragger Memory - MongoDB RAG Memory Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Store a memory
  ragger.py --store "Reid prefers MacPorts over Homebrew"
  
  # Search memories
  ragger.py --search "package manager preferences"
  
  # Import a file
  ragger.py --import-file MEMORY.md
  
  # Import with chunking
  ragger.py --import-file large_doc.txt --chunk-size 500
  
  # Run as MCP server for OpenClaw
  ragger.py --mcp
        """
    )
    
    parser.add_argument('--store', type=str, help="Store a memory")
    parser.add_argument('--search', type=str, help="Search memories")
    parser.add_argument('--import-file', type=str, help="Import a file")
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help=f"Max chars per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument('--limit', type=int, default=DEFAULT_SEARCH_LIMIT, help=f"Max search results (default: {DEFAULT_SEARCH_LIMIT})")
    parser.add_argument('--min-score', type=float, default=DEFAULT_MIN_SCORE, help=f"Min similarity score (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument('--count', action='store_true', help="Show number of stored memories")
    parser.add_argument('--server', action='store_true', help="Run HTTP server")
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f"HTTP server port (default: {DEFAULT_PORT})")
    parser.add_argument('--update-model', action='store_true', help="Download or update the embedding model")
    parser.add_argument('--mcp', action='store_true', help="Run as MCP server (JSON-RPC over stdin/stdout)")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(Path(__file__).parent.parent / 'ragger_memory.log'),
            logging.StreamHandler()
        ]
    )
    
    if args.update_model:
        RaggerMemory.download_model()
        print("✓ Model is up to date")
        return
    
    if args.server:
        run_server(args.port)
        return
    
    if args.mcp:
        run_mcp_server()
    else:
        memory = RaggerMemory()
        
        try:
            if args.count:
                print(f"Memories stored: {memory.count()}")
            
            elif args.store:
                memory_id = memory.store(args.store)
                print(f"✓ Stored: {memory_id}")
            
            elif args.import_file:
                import_file(memory, args.import_file, args.chunk_size)
            
            elif args.search:
                results = memory.search(args.search, args.limit, args.min_score)
                print(f"\nFound {len(results)} results:\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. [score: {r['score']:.3f}] {r['text'][:100]}...")
                    if r.get('metadata'):
                        print(f"   metadata: {r['metadata']}")
                    print()
            
            else:
                parser.print_help()
        
        finally:
            memory.close()


if __name__ == '__main__':
    main()
