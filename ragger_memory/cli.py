"""
Command-line interface for Ragger Memory

Verb-style CLI: ragger <verb> [options] [args]
No verb or 'help' prints usage.
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from .memory import RaggerMemory
from .client import RaggerClient, is_daemon_running
from .auth import load_token, ensure_token
from .mcp_server import run_mcp_server
from .server import run_server
from .config import get_config
from .inference import InferenceClient

logger = logging.getLogger(__name__)


def import_file(
    memory: RaggerMemory,
    filepath: str,
    min_chunk_size: int = 300,
    metadata: Optional[dict] = None
):
    """
    Import a file into memory with paragraph-aware chunking

    Args:
        memory: RaggerMemory instance
        filepath: Path to file
        min_chunk_size: Merge short paragraphs until at least this size
        metadata: Additional metadata to attach
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    text = path.read_text()

    # Normalize line endings (Windows → Unix) and strip null bytes
    text = text.replace('\r\n', '\n').replace('\r', '\n').replace('\x00', '')

    # Strip embedded base64 image data (noise for text embeddings)
    text = re.sub(r'!\[[^\]]*\]\(data:[^)]+\)', '', text)
    text = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '', text)

    # Collapse OCR multi-space artifacts to single space
    lines = text.split('\n')
    lines = [re.sub(r'  +', ' ', line) for line in lines]
    text = '\n'.join(lines)

    text = re.sub(r'\n{3,}', '\n\n', text)

    source_meta = {"source": str(path)}
    if metadata:
        source_meta.update(metadata)

    # Heading-aware paragraph chunking
    raw_paragraphs = text.split('\n\n')

    def _heading_level(line: str) -> int:
        m = re.match(r'^(#{1,6})\s', line)
        return len(m.group(1)) if m else 0

    def _heading_text(line: str) -> str:
        return re.sub(r'^#{1,6}\s+', '', line).strip()

    heading_stack: list[tuple[int, str]] = []
    pending_headings: list[str] = []

    def _current_section() -> str:
        return ' » '.join(h[1] for h in heading_stack)

    def _current_heading_block() -> str:
        if heading_stack:
            return '\n\n'.join('#' * level + ' ' + txt for level, txt in heading_stack)
        return ''

    annotated: list[tuple[str, str, str]] = []

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        level = _heading_level(para)
        if level > 0:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, _heading_text(para)))
            pending_headings.append(para)
        else:
            heading_block = _current_heading_block()
            section = _current_section()
            annotated.append((para, heading_block, section))
            pending_headings = []

    if pending_headings:
        section = _current_section()
        annotated.append(('\n\n'.join(pending_headings), '', section))

    chunks: list[tuple[str, str]] = []
    current = ""
    current_section = ""

    for body, heading_block, section in annotated:
        if not current:
            current = (heading_block + '\n\n' + body) if heading_block else body
            current_section = section
        elif len(current) >= min_chunk_size:
            chunks.append((current.strip(), current_section))
            current = (heading_block + '\n\n' + body) if heading_block else body
            current_section = section
        else:
            if section != current_section and heading_block:
                current = current + '\n\n' + heading_block + '\n\n' + body
                current_section = section
            else:
                current = current + '\n\n' + body
    if current.strip():
        chunks.append((current.strip(), current_section))

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


def _load_workspace_files() -> str:
    """
    Load workspace MD files for the system prompt.

    System files (shared personality): /var/ragger/ first, ~/.ragger/ fallback
      - SOUL.md, AGENTS.md, TOOLS.md

    User files (personal, private): ~/.ragger/ only
      - USER.md, MEMORY.md

    Returns combined text for injection into system prompt.
    """
    from .config import expand_path, system_data_dir

    user_dir = expand_path("~/.ragger")
    common_dir = system_data_dir()  # /var/ragger/

    sections = []

    # System files: /var/ragger/ first, ~/.ragger/ fallback
    for filename in ("SOUL.md", "AGENTS.md", "TOOLS.md"):
        path = os.path.join(common_dir, filename)
        if not os.path.isfile(path):
            path = os.path.join(user_dir, filename)
        if os.path.isfile(path):
            with open(path, "r") as f:
                sections.append(f.read().strip())

    # User files: ~/.ragger/ only
    for filename in ("USER.md", "MEMORY.md"):
        path = os.path.join(user_dir, filename)
        if os.path.isfile(path):
            with open(path, "r") as f:
                sections.append(f.read().strip())
    
    return "\n\n---\n\n".join(sections) if sections else ""


def _summarize_conversation(inference, model: str, turns: list) -> str:
    """Ask the LLM to summarize conversation turns into a memory entry."""
    if not turns:
        return ""

    conversation_text = ""
    for turn in turns:
        conversation_text += f"**{turn['role'].title()}:** {turn['content']}\n\n"

    summary_messages = [
        {"role": "system", "content": (
            "Summarize this conversation into a concise memory entry. "
            "Extract: key facts, decisions, questions asked, topics discussed. "
            "Write in third person past tense. Be brief — this will be stored "
            "as a memory chunk for future retrieval."
        )},
        {"role": "user", "content": conversation_text}
    ]

    try:
        response = inference.chat(summary_messages, stream=False, model=model)
        return inference.extract_content(response, model=model)
    except Exception as e:
        logger.warning(f"Summary generation failed: {e}")
        return ""


def run_chat():
    """
    Simple REPL: chat with memory context injection and persistence.

    Persistence (all configurable per-user in [chat]):
    - store_turns: store each exchange as it happens
    - summarize_on_pause: summarize after pause_minutes of idle
    - summarize_on_quit: summarize full conversation on exit
    """
    import time

    cfg = get_config()

    # Initialize format search dirs from config
    from . import api_formats
    api_formats.init_formats_dir(cfg.get("formats_dir", "/var/ragger/formats"))

    # Build inference client from config
    inference = InferenceClient.from_config(cfg)
    model = cfg.get("inference_model", "claude-sonnet-4-5")

    if not inference._endpoints:
        print("Error: no inference endpoints configured.")
        print("Add to ragger.ini:")
        print()
        print("  [inference]")
        print("  api_url = http://localhost:1234/v1")
        print("  api_key = lmstudio-local")
        print()
        print("Or for multiple endpoints:")
        print()
        print("  [inference.local]")
        print("  api_url = http://localhost:1234/v1")
        print("  api_key = lmstudio-local")
        print("  models = qwen/*, llama/*")
        return

    # Chat persistence settings
    store_turns = cfg.get("chat_store_turns", True)
    summarize_on_pause = cfg.get("chat_summarize_on_pause", True)
    pause_minutes = cfg.get("chat_pause_minutes", 10)
    summarize_on_quit = cfg.get("chat_summarize_on_quit", True)

    # Memory client
    use_client = is_daemon_running(cfg["host"], cfg["port"])
    if use_client:
        token = load_token()
        memory = RaggerClient(cfg["host"], cfg["port"], token)
    else:
        memory = RaggerMemory()

    # Load workspace files for persona
    workspace = _load_workspace_files()

    # Conversation state
    messages = []
    if workspace:
        messages.append({"role": "system", "content": workspace})

    unsummarized_turns = []  # turns since last summary
    last_activity = time.time()

    print(f"Ragger Chat (model: {model})")
    print("Type '/quit' or Ctrl+D to exit\n")

    def _store_turn(user_text: str, assistant_text: str):
        """Store a single exchange in memory."""
        if not store_turns:
            return
        try:
            turn_text = f"Chat turn:\nUser: {user_text}\nAssistant: {assistant_text}"
            memory.store(turn_text, {
                "collection": "memory",
                "category": "conversation",
                "source": "ragger-chat",
            })
        except Exception as e:
            logger.warning(f"Failed to store turn: {e}")

    def _check_pause_summary():
        """If idle long enough, summarize and store unsummarized turns."""
        nonlocal unsummarized_turns, last_activity
        if not summarize_on_pause or not unsummarized_turns:
            return
        idle_seconds = time.time() - last_activity
        if idle_seconds < pause_minutes * 60:
            return

        summary = _summarize_conversation(inference, model, unsummarized_turns)
        if summary:
            try:
                memory.store(summary, {
                    "collection": "memory",
                    "category": "conversation-summary",
                    "source": "ragger-chat",
                    "turns": len(unsummarized_turns),
                })
                logger.info(f"Stored pause summary ({len(unsummarized_turns)} turns)")
            except Exception as e:
                logger.warning(f"Failed to store pause summary: {e}")
        unsummarized_turns = []

    def _quit_summary():
        """Summarize full conversation on exit."""
        if not summarize_on_quit or not unsummarized_turns:
            return

        print("Summarizing conversation...", end="", flush=True)
        summary = _summarize_conversation(inference, model, unsummarized_turns)
        if summary:
            try:
                memory.store(summary, {
                    "collection": "memory",
                    "category": "conversation-summary",
                    "source": "ragger-chat",
                    "turns": len(unsummarized_turns),
                })
                print(" stored.")
            except Exception as e:
                print(f" failed: {e}")
        else:
            print(" skipped (empty).")

    try:
        while True:
            # Check for pause summary before waiting for input
            _check_pause_summary()

            try:
                user_input = input("You: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input in ("/quit", "/exit"):
                print("Goodbye!")
                break

            last_activity = time.time()

            # Search memory for context
            context_chunks = []
            try:
                result = memory.search(user_input, limit=3, min_score=0.3)
                results = result.get("results", [])
                if results:
                    context_chunks = [r["text"] for r in results]
            except Exception as e:
                logger.warning(f"Memory search failed: {e}")

            # Build message with context
            current_messages = [m.copy() for m in messages]

            if context_chunks:
                context_text = "\n\n---\n\n".join(context_chunks)
                memory_block = f"\n\n## Relevant memories:\n\n{context_text}"
                if current_messages and current_messages[0]["role"] == "system":
                    current_messages[0]["content"] += memory_block
                else:
                    current_messages.insert(0, {
                        "role": "system",
                        "content": memory_block.strip()
                    })

            current_messages.append({
                "role": "user",
                "content": user_input
            })

            # Send to inference API (streaming)
            print("Assistant: ", end="", flush=True)
            response_text = ""

            try:
                stream = inference.chat(current_messages, stream=True)
                for chunk in stream:
                    delta = inference.extract_delta(chunk)
                    if delta:
                        print(delta, end="", flush=True)
                        response_text += delta
                print()  # newline after response
            except Exception as e:
                print(f"\nError: {e}")
                continue

            # Update conversation history
            messages.append({"role": "user", "content": user_input})
            messages.append({"role": "assistant", "content": response_text})

            # Persistence: store turn and track for summary
            _store_turn(user_input, response_text)
            unsummarized_turns.append({"role": "user", "content": user_input})
            unsummarized_turns.append({"role": "assistant", "content": response_text})
            last_activity = time.time()

            print()  # blank line between exchanges

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        _quit_summary()
        memory.close()


def main():
    """CLI entry point — verb-style commands"""
    cfg = get_config()

    parser = argparse.ArgumentParser(
        prog="ragger",
        description="Ragger Memory — semantic memory store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verbs:
  serve             Start HTTP server
  chat              Chat with memory context (REPL)
  search <query>    Search memories
  store <text>      Store a memory
  count             Show memory count
  import <files>    Import files into memory
  export            Export memories to files
  mcp               Run as MCP server (stdin/stdout)
  rebuild-bm25      Rebuild BM25 index
  update-model      Download/update embedding model
  help              Show this help

Examples:
  ragger serve
  ragger serve --host 0.0.0.0 --port 9000
  ragger chat
  ragger search "deployment requirements"
  ragger search "transposition" --collections sibelius
  ragger store "The deploy script requires Node 18+"
  ragger count
  ragger import notes.md --collection docs
  ragger import doc1.md doc2.md --collection reference
        """
    )

    # Subcommands
    sub = parser.add_subparsers(dest="verb")

    # --- serve ---
    p_serve = sub.add_parser("serve", help="Start HTTP server")
    p_serve.add_argument("--host", type=str, default=cfg["host"],
                         help=f"Bind address (default: {cfg['host']})")
    p_serve.add_argument("--port", type=int, default=cfg["port"],
                         help=f"Port (default: {cfg['port']})")

    # --- chat ---
    sub.add_parser("chat", help="Chat with memory context (simple REPL)")

    # --- search ---
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", type=str, help="Search query")
    p_search.add_argument("--limit", type=int, default=cfg["default_search_limit"],
                          help=f"Max results (default: {cfg['default_search_limit']})")
    p_search.add_argument("--min-score", type=float, default=cfg["default_min_score"],
                          help=f"Min similarity (default: {cfg['default_min_score']})")
    p_search.add_argument("--collections", type=str, nargs="+", default=None,
                          help="Collections to search (default: all; use '*' for all)")
    p_search.add_argument("--collection", type=str, default=None,
                          help="Single collection to search")

    # --- store ---
    p_store = sub.add_parser("store", help="Store a memory")
    p_store.add_argument("text", type=str, help="Text to store")
    p_store.add_argument("--collection", type=str, default=None,
                         help="Collection name")
    p_store.add_argument("--public", action="store_true",
                         help="Store to shared system memory (default: private)")

    # --- count ---
    sub.add_parser("count", help="Show memory count")

    # --- import ---
    p_import = sub.add_parser("import", help="Import files into memory")
    p_import.add_argument("files", type=str, nargs="+", help="Files to import")
    p_import.add_argument("--collection", type=str, default=None,
                          help="Collection name for imported chunks")
    p_import.add_argument("--public", action="store_true",
                          help="Import to shared system memory (default: private)")
    p_import.add_argument("--min-chunk-size", type=int,
                          default=cfg["minimum_chunk_size"],
                          help=f"Min chunk size (default: {cfg['minimum_chunk_size']})")

    # --- export ---
    p_export = sub.add_parser("export", help="Export memories to files")
    p_export.add_argument("mode", choices=["docs", "memories", "all"],
                          help="What to export")
    p_export.add_argument("dest", type=str, help="Destination directory")
    p_export.add_argument("--collection", type=str, default=None,
                          help="Collection for docs export")
    p_export.add_argument("--group-by", type=str, default="date",
                          choices=["date", "category", "collection"],
                          help="Grouping for memory export")

    # --- mcp ---
    sub.add_parser("mcp", help="Run as MCP server (JSON-RPC over stdin/stdout)")

    # --- rebuild-bm25 ---
    sub.add_parser("rebuild-bm25", help="Rebuild BM25 index from all documents")

    # --- update-model ---
    sub.add_parser("update-model", help="Download/update embedding model")

    # --- help ---
    sub.add_parser("help", help="Show help")
    sub.add_parser("version", help="Show version")

    # --- Global options ---
    parser.add_argument("--config", type=str, default="",
                        help="Path to config file (overrides /etc/ragger.ini and ~/.ragger/ragger.ini)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose logging")
    parser.add_argument("-V", "--version", action="store_true",
                        help="Show version")

    args = parser.parse_args()

    if args.version or (hasattr(args, 'verb') and args.verb == "version"):
        from . import build_version
        print(f"ragger {build_version()}")
        return

    # No verb or 'help' → print help with version header
    if not args.verb or args.verb == "help":
        from . import build_version
        print(f"ragger {build_version()}\n")
        parser.print_help()
        return

    # Configure logging
    from .logs import setup_logging
    setup_logging(
        verbose=args.verbose,
        server_mode=(args.verb == "serve")
    )

    # --- Dispatch ---

    if args.verb == "serve":
        run_server(args.host, args.port)

    elif args.verb == "chat":
        run_chat()

    elif args.verb == "mcp":
        run_mcp_server()

    elif args.verb == "update-model":
        RaggerMemory.download_model()
        print("✓ Model is up to date")

    elif args.verb == "rebuild-bm25":
        from .embedding import Embedder
        from .sqlite_backend import SqliteBackend
        embedder = Embedder()
        backend = SqliteBackend(embedder)
        count = backend.rebuild_bm25_index()
        print(f"✓ BM25 index rebuilt: {count} documents")
        backend.close()

    elif args.verb == "export":
        if args.mode == "docs":
            if not args.collection:
                print("Error: --collection required for docs export")
                return
            from .export import export_docs
            export_docs(args.collection, args.dest)
        elif args.mode == "memories":
            from .export import export_memories
            export_memories(args.dest, args.group_by)
        elif args.mode == "all":
            from .export import export_all
            export_all(args.dest, args.group_by)

    else:
        # Commands that need memory access.
        # If daemon is running, use thin HTTP client (no model loading).
        # Otherwise, load the model directly.
        use_client = args.verb in ("search", "store", "count") and \
                     is_daemon_running(cfg["host"], cfg["port"])

        if use_client:
            token = load_token()
            memory = RaggerClient(cfg["host"], cfg["port"], token)
        else:
            memory = RaggerMemory()

        try:
            if args.verb == "count":
                print(memory.count())

            elif args.verb == "store":
                meta = {}
                if args.collection:
                    meta["collection"] = args.collection
                memory_id = memory.store(args.text, meta if meta else None)
                print(f"Stored with id: {memory_id}")

            elif args.verb == "import":
                # Default is private (user's DB). --public routes to system DB.
                if getattr(args, 'public', False):
                    # TODO: system DB path from system config when multi-user is active
                    import_memory = memory  # for now, same DB
                else:
                    import_memory = memory

                import_meta = {}
                if args.collection:
                    import_meta["collection"] = args.collection
                for filepath in args.files:
                    import_file(import_memory, filepath, args.min_chunk_size,
                                import_meta if import_meta else None)
                if import_memory is not memory:
                    import_memory.close()

            elif args.verb == "search":
                collections = args.collections or (
                    [args.collection] if args.collection else None
                )
                result = memory.search(args.query, args.limit,
                                       args.min_score, collections)
                results = result["results"]
                timing = result.get("timing", {})
                print(f"\nFound {len(results)} results:\n")
                for i, r in enumerate(results, 1):
                    print(f"{i}. [score: {r['score']:.3f}] {r['text'][:100]}...")
                    if r.get("metadata"):
                        print(f"   metadata: {r['metadata']}")
                    print()
                if timing:
                    print(f"Timing: embed {timing.get('embedding_ms', '?')}ms, "
                          f"search {timing.get('search_ms', '?')}ms, "
                          f"total {timing.get('total_ms', '?')}ms "
                          f"({timing.get('corpus_size', '?')} chunks)")
        finally:
            memory.close()


if __name__ == "__main__":
    main()
