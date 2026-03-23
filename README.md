# Ragger Memory

**Local-first semantic memory for AI agents and humans.**

Store anything. Find it by meaning. No cloud, no APIs, no subscriptions — everything runs on your machine.

Ragger combines dense vector search with keyword matching for better recall, uses local embeddings (no API calls), and supports pluggable storage backends. It's designed as a long-term memory backend for AI agents but works standalone.

A [C++ port](https://github.com/diskerror/ragger.cpp) is also available with the same HTTP API, database format, and config file.

## Features

- **Local embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (384-dim, ~90MB)
- **Hybrid search** — BM25 keyword + vector cosine similarity (pure Python, configurable blend)
- **Fast vector search** — NumPy cosine similarity (~10-50ms for 50K documents)
- **Pluggable backends** — SQLite (default); abstract base class makes it easy to add more
- **HTTP & MCP servers** — REST API and Model Context Protocol for tool integration
- **Collection filtering** — Organize memories into searchable collections (e.g., `docs`, `reference`, `memory`)
- **Usage tracking** — Per-memory access stats for identifying high-value content
- **Python API** — Reusable `RaggerMemory` class for embedding into your own apps
- **Query logging** — Track searches with timing, scores, and quality metrics
- **Path normalization** — `$HOME` → `~/` for portable, privacy-friendly storage

## Quick Start

```bash
# Install dependencies
cd /path/to/Ragger
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Store a memory
ragger store "The deploy script requires Node 18+"

# Search (semantic — finds by meaning, not just keywords)
ragger search "deployment requirements"

# Import a document (chunked at paragraph boundaries)
ragger import notes.md --collection docs

# Start HTTP server (for OpenClaw plugin or any HTTP client)
ragger serve
```

First run downloads the embedding model (~90MB) to your HuggingFace cache. After that, all operations are offline.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Installation, setup, first run |
| [Configuration](docs/configuration.md) | Config files, settings reference |
| [Collections](docs/collections.md) | Organizing memories into collections |
| [Search & RAG](docs/search-and-rag.md) | How hybrid search works |
| [HTTP API](docs/http-api.md) | REST endpoints, MCP server, auth |
| [Python API](docs/python-api.md) | Library usage, custom backends |
| [Chat Persistence](docs/chat-persistence.md) | Turn storage, summaries, cleanup |
| [Deployment](docs/deployment.md) | Production setup, LaunchDaemon, multi-user |
| [Project Structure](docs/project-structure.md) | Code layout, database schema |
| [OpenClaw Integration](OPENCLAW.md) | Plugin setup for OpenClaw |
| [Agent Guide](README_TO_AGENT.md) | Best practices for AI agents |
| [Roadmap](ROADMAP.md) | Future plans |

## Status

**Version 0.7.0** — Single-user, fully functional.

Multi-user framework is in place (layered config with system/user INI files, SERVER_LOCKED keys, system ceilings on user settings, token auth, persona file hierarchy) but the data layer is still single-user: one database, one user, no user routing. Multi-user data support (users table, per-user DBs, search merging) is planned for a future release.

## Requirements

- Python 3.10+
- ~1GB disk for model + dependencies
- SQLite backend: No extra dependencies (uses Python stdlib)

## How It Works

1. **Store:** Text → 384-dim vector via sentence-transformers → backend document with text, embedding, metadata, and timestamp.

2. **Search:** Query → vector → hybrid scoring (NumPy cosine similarity + BM25 keyword relevance) → results ranked by blended score.

3. **Performance:** 50K vectors × 384 dims ≈ 10-50ms on Apple Silicon. The embedding model stays loaded in server mode for fast repeated queries.

## Python API Example

```python
from ragger_memory import RaggerMemory

with RaggerMemory() as memory:
    memory.store("Some fact", metadata={"source": "notes.md"})
    results = memory.search("related query", limit=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['text']}")
```

## HTTP API Example

```bash
# Store
curl -X POST http://localhost:8432/store \
  -H "Content-Type: application/json" \
  -d '{"text": "Deploy to staging every Friday", "metadata": {"category": "preference"}}'

# Search
curl -X POST http://localhost:8432/search \
  -H "Content-Type: application/json" \
  -d '{"query": "deployment schedule", "limit": 3}'
```

## Command-Line Examples

```bash
# Store a memory
ragger store "The deploy script requires Node 18+"

# Search with filters
ragger search "API authentication" --collections docs --limit 3

# Import files
ragger import notes.md --collection docs
ragger import doc1.md doc2.md doc3.md --collection reference

# Export documents
ragger export docs ./exported/

# Count memories
ragger count

# Rebuild BM25 index
ragger rebuild-bm25

# Run MCP server (JSON-RPC over stdin/stdout)
ragger mcp
```

## Collections

Memories are organized into **collections** — logical groups that let you separate reference material from conversation memories and search the right pool for the right question.

**Built-in:**
- `memory` — Agent-stored memories: facts, decisions, preferences, session summaries (default)

**Example custom collections:**
- `docs` — Project documentation, API references
- `reference` — Technical manuals, specifications
- `notes` — Meeting notes, research, bookmarks

Search specific collections:

```bash
ragger search "API auth" --collections docs reference
```

Or search everything (default):

```bash
ragger search "API auth"
```

See [Collections](docs/collections.md) for best practices and AI agent integration tips.

## License

GPL v3 — See [LICENSE](LICENSE) for details.
