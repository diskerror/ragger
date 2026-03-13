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

VALID_ENGINES = ("mongodb", "sqlite")


def convert_backend(source_engine: str, dest_engine: str):
    """
    Copy all memories from one backend to another.
    Embeddings are copied as-is (no re-encoding).

    Args:
        source_engine: Source backend name ("mongodb" or "sqlite")
        dest_engine: Destination backend name ("mongodb" or "sqlite")
    """
    from .embedding import Embedder
    from datetime import datetime

    if source_engine == dest_engine:
        print(f"Source and destination are the same ({source_engine}). Nothing to do.")
        return

    for eng in (source_engine, dest_engine):
        if eng not in VALID_ENGINES:
            print(f"Unknown engine: {eng}. Must be one of {VALID_ENGINES}")
            return

    # Shared embedder — needed by both backends but won't re-encode anything
    embedder = Embedder()

    # Open source
    if source_engine == "mongodb":
        from .backend.mongo import MongoBackend
        source = MongoBackend(embedder)
    else:
        from .backend.sqlite import SqliteBackend
        source = SqliteBackend(embedder)

    # Open destination
    if dest_engine == "mongodb":
        from .backend.mongo import MongoBackend
        dest = MongoBackend(embedder)
    else:
        from .backend.sqlite import SqliteBackend
        dest = SqliteBackend(embedder)

    src_count = source.count()
    if src_count == 0:
        print(f"Source ({source_engine}) is empty. Nothing to convert.")
        source.close()
        dest.close()
        return

    print(f"Converting {src_count} memories: {source_engine} → {dest_engine}")

    ids, texts, embeddings, metadata_list, timestamps = source.load_all_embeddings()

    copied = 0
    for i in range(len(ids)):
        # Normalize timestamp to datetime if it's a string
        ts = timestamps[i]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        elif ts is None:
            from datetime import timezone
            ts = datetime.now(timezone.utc)

        dest.store_raw(
            text=texts[i],
            embedding=embeddings[i].tolist(),
            metadata=metadata_list[i],
            timestamp=ts
        )
        copied += 1
        if copied % 500 == 0:
            print(f"  {copied}/{src_count}...")

    print(f"✓ Converted {copied} memories from {source_engine} to {dest_engine}")

    source.close()
    dest.close()


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

    # Collapse OCR multi-space artifacts to single space (per line, preserving structure)
    lines = text.split('\n')
    lines = [re.sub(r'  +', ' ', line) for line in lines]
    text = '\n'.join(lines)

    text = re.sub(r'\n{3,}', '\n\n', text)  # collapse extra blank lines left behind

    source_meta = {"source": str(path), "filename": path.name}
    if metadata:
        source_meta.update(metadata)
    
    # Split on paragraph boundaries with heading-aware chunking.
    #
    # Headings are tracked as a breadcrumb trail (e.g. "Intro - Setup - Step 1").
    # Each chunk gets:
    #   - The current heading(s) prepended to the text (improves embeddings)
    #   - A "section" metadata field with the breadcrumb trail
    #
    # Paragraphs are kept whole — never split mid-paragraph. Small consecutive
    # paragraphs under the same heading are merged up to chunk_size.

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
        return ' » '.join(h[1] for h in heading_stack)

    def _current_heading_block() -> str:
        """Build the full heading chain to prepend to chunk text.
        
        Always uses the heading stack (which reflects the full hierarchy).
        """
        if heading_stack:
            return '\n\n'.join('#' * level + ' ' + txt for level, txt in heading_stack)
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
            # Body paragraph — attach full heading chain and section breadcrumb
            heading_block = _current_heading_block()
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

    # Chunk annotated paragraphs: merge small ones up to chunk_size,
    # but NEVER split a paragraph. Oversized paragraphs become their own chunk.
    chunks: list[tuple[str, str]] = []  # (chunk_text, section)
    current = ""
    current_section = ""

    for full_text, section in annotated:
        candidate = current + ('\n\n' if current else '') + full_text
        if len(candidate) > chunk_size and current:
            # Current buffer is full — flush it
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
  
  # Import multiple files
  ragger.py --import-file doc1.md doc2.md doc3.md
  
  # Import with chunking
  ragger.py --import-file large_doc.txt --chunk-size 500
  
  # Run as MCP server for OpenClaw
  ragger.py --mcp
  
  # Convert all memories from MongoDB to SQLite
  ragger.py --convert mongodb sqlite
  
  # Export documents back to files
  ragger.py --export-docs music ./exported/music/
  
  # Export conversation memories
  ragger.py --export-memories ./exported/memories/
  
  # Export everything
  ragger.py --export-all ./exported/
        """
    )
    
    parser.add_argument('--store', type=str, help="Store a memory")
    parser.add_argument('--search', type=str, help="Search memories")
    parser.add_argument('--import-file', type=str, nargs='+', help="Import one or more files")
    parser.add_argument('--chunk-size', type=int, default=DEFAULT_CHUNK_SIZE, help=f"Max chars per chunk (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument('--collection', type=str, default=None, help="Collection name for import or search (e.g. music, sibelius, forscore, memory)")
    parser.add_argument('--collections', type=str, nargs='+', default=None, help="Collections to search (default: memory only; use '*' for all)")
    parser.add_argument('--limit', type=int, default=DEFAULT_SEARCH_LIMIT, help=f"Max search results (default: {DEFAULT_SEARCH_LIMIT})")
    parser.add_argument('--min-score', type=float, default=DEFAULT_MIN_SCORE, help=f"Min similarity score (default: {DEFAULT_MIN_SCORE})")
    parser.add_argument('--count', action='store_true', help="Show number of stored memories")
    parser.add_argument('--server', action='store_true', help="Run HTTP server")
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f"HTTP server port (default: {DEFAULT_PORT})")
    parser.add_argument('--update-model', action='store_true', help="Download or update the embedding model")
    parser.add_argument('--convert', nargs=2, metavar=('FROM', 'TO'),
                        help="Convert memories between backends (e.g. --convert mongodb sqlite)")
    parser.add_argument('--export-docs', nargs=2, metavar=('COLLECTION', 'DEST'),
                        help="Export a document collection back to files")
    parser.add_argument('--export-memories', type=str, metavar='DEST',
                        help="Export conversation memories as markdown")
    parser.add_argument('--export-all', type=str, metavar='DEST',
                        help="Export everything (docs by collection, memories grouped)")
    parser.add_argument('--group-by', type=str, default='date',
                        choices=['date', 'category', 'collection'],
                        help="Grouping for memory export (default: date)")
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
    
    if args.convert:
        convert_backend(args.convert[0], args.convert[1])
        return

    if args.export_docs:
        from .export import export_docs
        export_docs(args.export_docs[0], args.export_docs[1])
        return

    if args.export_memories:
        from .export import export_memories
        export_memories(args.export_memories, args.group_by)
        return

    if args.export_all:
        from .export import export_all
        export_all(args.export_all, args.group_by)
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
                import_meta = {}
                if args.collection:
                    import_meta['collection'] = args.collection
                for filepath in args.import_file:
                    import_file(memory, filepath, args.chunk_size, import_meta if import_meta else None)
            
            elif args.search:
                collections = args.collections or (
                    [args.collection] if args.collection else None
                )
                search_result = memory.search(args.search, args.limit, args.min_score, collections)
                results = search_result["results"]
                timing = search_result.get("timing", {})
                print(f"\nFound {len(results)} results:\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. [score: {r['score']:.3f}] {r['text'][:100]}...")
                    if r.get('metadata'):
                        print(f"   metadata: {r['metadata']}")
                    print()
                if timing:
                    print(f"Timing: embed {timing.get('embedding_ms', '?')}ms, "
                          f"search {timing.get('search_ms', '?')}ms, "
                          f"total {timing.get('total_ms', '?')}ms "
                          f"({timing.get('corpus_size', '?')} chunks)")
            
            else:
                parser.print_help()
        
        finally:
            memory.close()


if __name__ == '__main__':
    main()
