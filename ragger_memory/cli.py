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


def _load_workspace_files(max_chars: int = 0) -> str:
    """
    Load workspace MD files for the system prompt.

    System files (shared personality): /var/ragger/ first, ~/.ragger/ fallback
      - SOUL.md, AGENTS.md, TOOLS.md

    User files (personal, private): ~/.ragger/ only
      - USER.md, MEMORY.md

    Args:
        max_chars: Maximum total characters to return. 0 = unlimited.
                   When limited, files are included in priority order
                   (SOUL > USER > AGENTS > TOOLS > MEMORY) and
                   the last included file is truncated to fit.

    Returns combined text for injection into system prompt.
    """
    from .config import expand_path, system_data_dir

    user_dir = expand_path("~/.ragger")
    common_dir = system_data_dir()  # /var/ragger/

    # Priority order: identity first, reference last
    file_list = [
        # (filename, search_common_first)
        ("SOUL.md", True),
        ("USER.md", False),    # user-only
        ("AGENTS.md", True),
        ("TOOLS.md", True),
        ("MEMORY.md", False),  # user-only
    ]

    sections = []
    total = 0

    for filename, search_common in file_list:
        path = None
        if search_common:
            candidate = os.path.join(common_dir, filename)
            if os.path.isfile(candidate):
                path = candidate
        if path is None:
            candidate = os.path.join(user_dir, filename)
            if os.path.isfile(candidate):
                path = candidate
        if path is None:
            continue

        with open(path, "r") as f:
            content = f.read().strip()

        if not content:
            continue

        if max_chars > 0:
            remaining = max_chars - total - (len("\n\n---\n\n") * len(sections))
            if remaining <= 0:
                break
            if len(content) > remaining:
                # Truncate to fit, try to break at a paragraph
                content = content[:remaining]
                last_break = content.rfind('\n\n')
                if last_break > remaining // 2:
                    content = content[:last_break]
                content += f"\n\n[... {filename} truncated ...]"
                sections.append(content)
                break

        sections.append(content)
        total += len(content)

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
    - store_turns: "true" (per-turn), "session" (one growing entry), or "false" (summaries only)
    - summarize_on_pause: summarize after pause_minutes of idle
    - summarize_on_quit: summarize full conversation on exit
    """
    import time
    from datetime import datetime, timezone, timedelta

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
    store_turns = cfg.get("chat_store_turns", "true")  # "true", "session", or "false"
    # Normalize to lowercase for comparison
    if isinstance(store_turns, bool):
        store_turns = "true" if store_turns else "false"
    else:
        store_turns = str(store_turns).lower()
    
    summarize_on_pause = cfg.get("chat_summarize_on_pause", True)
    pause_minutes = cfg.get("chat_pause_minutes", 10)
    summarize_on_quit = cfg.get("chat_summarize_on_quit", True)
    
    # System hard limits
    max_turn_retention_minutes = cfg.get("chat_max_turn_retention_minutes", 60)
    max_turns_stored = cfg.get("chat_max_turns_stored", 100)

    # Context sizing — only constrain persona for small context windows
    PERSONA_SIZING_THRESHOLD = 32768  # tokens — below this, apply persona_pct
    user_max_persona = cfg.get("chat_max_persona_chars", 0)
    max_memory_results = cfg.get("chat_max_memory_results", 3)
    persona_pct = cfg.get("chat_persona_pct", 25)
    chars_per_token = cfg.get("chat_chars_per_token", 4.0)

    # Check endpoint's context window
    ep = inference._resolve_endpoint(model)
    if 0 < ep.max_context < PERSONA_SIZING_THRESHOLD:
        # Small context — apply percentage-based sizing
        persona_budget = int(ep.max_context * chars_per_token * (persona_pct / 100.0))
        persona_budget = max(persona_budget, 500)
        if user_max_persona > 0:
            max_persona_chars = min(user_max_persona, persona_budget)
        else:
            max_persona_chars = persona_budget
    else:
        # Large or unknown context — load everything, user cap still applies
        max_persona_chars = user_max_persona  # 0 = unlimited

    # Memory client
    use_client = is_daemon_running(cfg["host"], cfg["port"])
    if use_client:
        token = load_token()
        memory = RaggerClient(cfg["host"], cfg["port"], token)
    else:
        memory = RaggerMemory()

    # Load workspace files for persona
    workspace = _load_workspace_files(max_chars=max_persona_chars)

    # Conversation state
    messages = []
    if workspace:
        messages.append({"role": "system", "content": workspace})

    unsummarized_turns = []  # turns since last summary
    last_activity = time.time()
    session_memory_id = None  # for "session" mode

    print(f"Ragger Chat (model: {model})")
    print(f"Turn storage: {store_turns}")
    if 0 < ep.max_context < PERSONA_SIZING_THRESHOLD:
        print(f"Context: {ep.max_context} tokens ({ep.name}) → {persona_pct}% = {max_persona_chars} chars persona")
    else:
        persona_info = f"{max_persona_chars} chars" if max_persona_chars > 0 else "unlimited"
        ctx_info = f"{ep.max_context} tokens" if ep.max_context > 0 else "unknown"
        print(f"Context: {ctx_info} ({ep.name}) | Persona: {persona_info}")
    print("Type '/quit' or Ctrl+D to exit\n")

    def _store_turn(user_text: str, assistant_text: str):
        """Store a single exchange in memory (mode-dependent)."""
        nonlocal session_memory_id
        
        if store_turns == "false":
            return  # No raw turn storage, summaries only
        
        try:
            turn_text = f"User: {user_text}\n\nAssistant: {assistant_text}"
            
            if store_turns == "true":
                # Per-turn mode: each exchange is a separate memory
                memory.store(turn_text, {
                    "collection": "memory",
                    "category": "conversation",
                    "source": "ragger-chat",
                })
            
            elif store_turns == "session":
                # Session mode: one growing memory entry, update by deleting and re-storing
                if session_memory_id:
                    # Load existing session, append new turn, delete old, store new
                    # For now, we'll just append by deleting and re-storing with accumulated text
                    # This is a simplified approach - ideally we'd have an update method
                    try:
                        memory.delete(session_memory_id)
                    except Exception as e:
                        logger.warning(f"Failed to delete old session memory: {e}")
                
                # Build full session text from unsummarized_turns
                session_text = "\n\n---\n\n".join(
                    f"{t['role'].title()}: {t['content']}" 
                    for t in unsummarized_turns
                ) + f"\n\n---\n\nUser: {user_text}\n\nAssistant: {assistant_text}"
                
                session_memory_id = memory.store(session_text, {
                    "collection": "memory",
                    "category": "conversation",
                    "source": "ragger-chat",
                    "mode": "session",
                })
        
        except Exception as e:
            logger.warning(f"Failed to store turn: {e}")
    
    def _expire_old_turns():
        """Check for and expire old turns based on system hard limits."""
        if store_turns == "false" or store_turns == "session":
            return  # Only applies to per-turn mode
        
        try:
            # Find all conversation turns
            turns = memory.search_by_metadata({
                "category": "conversation",
                "source": "ragger-chat"
            })
            
            # Filter out session-mode entries
            turns = [t for t in turns if t["metadata"].get("mode") != "session"]
            
            if not turns:
                return
            
            now = datetime.now(timezone.utc)
            cutoff_time = now - timedelta(minutes=max_turn_retention_minutes)
            
            # Find expired turns (older than retention window)
            expired = []
            for turn in turns:
                # Parse timestamp
                ts_str = turn["timestamp"]
                if ts_str:
                    ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                    if ts < cutoff_time:
                        expired.append(turn)
            
            # Also check count limit
            if len(turns) > max_turns_stored:
                # Sort by timestamp, oldest first
                sorted_turns = sorted(turns, key=lambda t: t["timestamp"])
                excess_count = len(turns) - max_turns_stored
                # Add oldest excess turns to expired list (avoid duplicates)
                expired_ids = {t["id"] for t in expired}
                for turn in sorted_turns:
                    if turn["id"] not in expired_ids:
                        expired.append(turn)
                        if len(expired) >= excess_count + len(expired_ids):
                            break
            
            if not expired:
                return
            
            logger.info(f"Expiring {len(expired)} old conversation turns")
            
            # Batch summarize expired turns
            summary_text = "\n\n---\n\n".join(t["text"] for t in expired)
            summary = _summarize_conversation(
                inference, 
                model, 
                [{"role": "assistant", "content": summary_text}]
            )
            
            if summary:
                # Store summary
                memory.store(summary, {
                    "collection": "memory",
                    "category": "conversation-summary",
                    "source": "ragger-chat",
                    "expired_turns": len(expired),
                })
            
            # Delete expired turns
            expired_ids = [t["id"] for t in expired]
            deleted = memory.delete_batch(expired_ids)
            logger.info(f"Deleted {deleted} expired turns, stored summary")
        
        except Exception as e:
            logger.warning(f"Failed to expire old turns: {e}")

    def _bg_summarize(turns_to_summarize):
        """Fork a background process to summarize turns and expire old ones."""
        if not turns_to_summarize:
            return

        turns_copy = list(turns_to_summarize)
        st = store_turns
        max_ret = max_turn_retention_minutes
        max_st = max_turns_stored

        pid = os.fork()
        if pid != 0:
            # Parent: return immediately
            return

        # Child process: fresh connections (SQLite not fork-safe), summarize, exit
        try:
            if use_client:
                child_memory = RaggerClient(cfg["host"], cfg["port"], load_token())
            else:
                child_memory = RaggerMemory()

            child_inference = InferenceClient.from_config(cfg)

            summary = _summarize_conversation(child_inference, model, turns_copy)
            if summary:
                child_memory.store(summary, {
                    "collection": "memory",
                    "category": "conversation-summary",
                    "source": "ragger-chat",
                    "turns": len(turns_copy),
                })

            # Expire old turns
            if st not in ("false", "session"):
                try:
                    turns = child_memory.search_by_metadata({
                        "category": "conversation",
                        "source": "ragger-chat"
                    })
                    turns = [t for t in turns if t.get("metadata", {}).get("mode") != "session"]
                    if turns:
                        now = datetime.now(timezone.utc)
                        cutoff = now - timedelta(minutes=max_ret)
                        expired_ids = []
                        for t in turns:
                            ts_str = t.get("timestamp", "")
                            if ts_str:
                                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                                if ts < cutoff:
                                    expired_ids.append(t["id"])
                        if len(turns) > max_st:
                            sorted_t = sorted(turns, key=lambda x: x.get("timestamp", ""))
                            excess = len(turns) - max_st
                            existing = set(expired_ids)
                            for t in sorted_t[:excess]:
                                if t["id"] not in existing:
                                    expired_ids.append(t["id"])
                        if expired_ids:
                            child_memory.delete_batch(expired_ids)
                except Exception:
                    pass

            child_memory.close()
        except Exception:
            pass
        finally:
            os._exit(0)

    def _check_pause_summary():
        """If idle long enough, summarize unsummarized turns in background."""
        nonlocal unsummarized_turns, last_activity
        if not summarize_on_pause or not unsummarized_turns:
            return
        idle_seconds = time.time() - last_activity
        if idle_seconds < pause_minutes * 60:
            return

        _bg_summarize(unsummarized_turns)
        unsummarized_turns = []

    def _quit_summary():
        """Summarize full conversation on exit in background."""
        if not summarize_on_quit or not unsummarized_turns:
            return
        print("Summarizing in background...")
        _bg_summarize(unsummarized_turns)

    # Launch-time orphan check: summarize and clean up any raw turns from crashed session
    def _check_orphaned_turns():
        """Check for orphaned raw turns from a crashed session and summarize them."""
        try:
            # Find all raw conversation turns
            turns = memory.search_by_metadata({
                "category": "conversation",
                "source": "ragger-chat"
            })
            
            # Filter out session-mode entries and summaries
            turns = [t for t in turns if t["metadata"].get("mode") != "session"]
            
            if not turns:
                return
            
            # Check if these are orphans (no recent summary exists)
            # If we have raw turns but we're starting fresh, they're likely from a crashed session
            # For now, we'll check if there are any turns older than the last summary
            summaries = memory.search_by_metadata({
                "category": "conversation-summary",
                "source": "ragger-chat"
            })
            
            if not summaries:
                # No summaries at all, but we have turns — they're orphans
                if turns:
                    logger.info(f"Found {len(turns)} orphaned turns from previous session, summarizing...")
                    print(f"Recovering {len(turns)} orphaned conversation turns...", flush=True)
                    
                    # Build conversation text for summarization
                    orphan_turns = []
                    for turn in turns:
                        # Parse the turn text to extract user/assistant exchanges
                        text = turn["text"]
                        if "User:" in text and "Assistant:" in text:
                            parts = text.split("\n\n")
                            for part in parts:
                                if part.startswith("User:"):
                                    orphan_turns.append({"role": "user", "content": part[5:].strip()})
                                elif part.startswith("Assistant:"):
                                    orphan_turns.append({"role": "assistant", "content": part[10:].strip()})
                    
                    if orphan_turns:
                        summary = _summarize_conversation(inference, model, orphan_turns)
                        if summary:
                            memory.store(summary, {
                                "collection": "memory",
                                "category": "conversation-summary",
                                "source": "ragger-chat",
                                "turns": len(turns),
                                "recovered": True,
                            })
                            logger.info(f"Stored recovery summary for {len(turns)} orphaned turns")
                    
                    # Now expire the old turns
                    _expire_old_turns()
                    print("✓ Recovery complete")
            else:
                # We have summaries - check if turns are newer than most recent summary
                # Sort summaries by timestamp
                summaries.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
                last_summary_time = summaries[0].get("timestamp", "") if summaries else ""
                
                # Check for turns newer than last summary (shouldn't happen normally)
                # or turns from before last summary that weren't cleaned up
                orphans = []
                if last_summary_time:
                    for turn in turns:
                        turn_time = turn.get("timestamp", "")
                        # If turn is significantly older than last summary, it's an orphan
                        if turn_time < last_summary_time:
                            orphans.append(turn)
                
                if orphans and len(orphans) > len(turns) * 0.5:
                    # If more than half are orphans, likely a cleanup issue
                    logger.info(f"Found {len(orphans)} old turns, cleaning up...")
                    _expire_old_turns()
        
        except Exception as e:
            logger.warning(f"Orphan check failed: {e}")
    
    _check_orphaned_turns()
    
    # Clean up old turns at startup
    _expire_old_turns()

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
                result = memory.search(user_input, limit=max_memory_results, min_score=0.3)
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
        _expire_old_turns()  # Final cleanup
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

    # --- user provisioning ---
    sub.add_parser("add-self", help="Provision yourself: create ~/.ragger/ and token")
    p_add_user = sub.add_parser("add-user", help="Provision a user (requires sudo)")
    p_add_user.add_argument("username", type=str, help="Username to provision")
    p_add_user.add_argument("--admin", action="store_true", help="Grant admin privileges")
    sub.add_parser("add-all", help="Provision all users with home directories (requires sudo)")

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

    elif args.verb == "add-self":
        import getpass
        from .auth import provision_user
        username = getpass.getuser()
        token, created = provision_user(username)
        if created:
            print(f"✓ Created ~/.ragger/token for {username}")
        else:
            print(f"Token already exists for {username}")
        # Register via daemon HTTP API (daemon owns the common DB)
        try:
            from .client import RaggerClient
            client = RaggerClient(cfg["host"], cfg["port"], token)
            if client.is_available():
                result = client.register_user(username)
                print(f"✓ Registered in database (user_id: {result.get('user_id')})")
            else:
                print("Server not running. Will auto-register on first authenticated request.")
        except Exception:
            print("Server will auto-register on first authenticated request.")

    elif args.verb == "add-user":
        import os
        from .auth import provision_user
        username = args.username
        is_admin = getattr(args, 'admin', False)
        try:
            token, created = provision_user(username)
        except KeyError:
            print(f"Error: user '{username}' not found in system")
            return
        except PermissionError:
            print(f"Error: permission denied. Run with sudo to provision other users.")
            return
        if created:
            print(f"✓ Created token for {username}")
        else:
            print(f"Token already exists for {username}")
        # Register via daemon
        try:
            from .client import RaggerClient
            client = RaggerClient(cfg["host"], cfg["port"], token)
            if client.is_available():
                result = client.register_user(username, is_admin=is_admin)
                print(f"✓ Registered in database (user_id: {result.get('user_id')})")
            else:
                print("Server not running. Will auto-register on first authenticated request.")
        except Exception:
            print("Server will auto-register on first authenticated request.")

    elif args.verb == "add-all":
        import os
        import pwd
        from .auth import provision_user
        if os.getuid() != 0:
            print("Error: add-all requires sudo")
            return
        from .client import RaggerClient
        # Scan home directories
        count = 0
        for pw in pwd.getpwall():
            # Skip system users (uid < 500 on macOS, < 1000 on Linux)
            if pw.pw_uid < 500:
                continue
            # Skip users without real home directories
            if not os.path.isdir(pw.pw_dir):
                continue
            # Skip nobody and other special accounts
            if pw.pw_name in ('nobody', 'nfsnobody'):
                continue
            try:
                token, created = provision_user(pw.pw_name, home_dir=pw.pw_dir)
                status = "created" if created else "exists"
                try:
                    client = RaggerClient(cfg["host"], cfg["port"], token)
                    if client.is_available():
                        client.register_user(pw.pw_name)
                        status += ", registered"
                    else:
                        status += ", pending registration"
                except Exception:
                    status += ", pending registration"
                print(f"  {pw.pw_name}: {status}")
                count += 1
            except Exception as e:
                print(f"  {pw.pw_name}: error ({e})")
        print(f"✓ Processed {count} users")

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
