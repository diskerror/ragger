# Ragger Memory

Semantic memory store with local embeddings and pluggable storage backends.
Designed as a long-term memory backend for AI agents (OpenClaw, etc.) but
usable standalone.

No external APIs, no cloud services — everything runs locally.

## Features

- **Local embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (384-dim, ~90MB)
- **Hybrid search** — BM25 keyword + vector cosine similarity (pure Python, configurable blend)
- **Fast vector search** — NumPy cosine similarity (~10-50ms for 50K documents)
- **Pluggable backends** — SQLite (default); abstract base class makes it easy to add more
- **HTTP server** — REST API on localhost for tool integration
- **MCP server** — JSON-RPC over stdin/stdout (Model Context Protocol), with plain text fallback
- **Collection filtering** — Organize memories into searchable collections (e.g. `docs`, `reference`, `memory`)
- **Usage tracking** — Per-memory access stats for identifying high-value content
- **Path normalization** — `$HOME` → `~/` for portable, privacy-friendly storage
- **CLI tools** — Store, search, import files, count memories
- **Python API** — Reusable `RaggerMemory` class
- **Query logging** — Track searches with timing, scores, and quality metrics

## Requirements

- Python 3.10+
- ~1GB disk for model + dependencies
- **SQLite backend:** No extra dependencies (uses Python stdlib)

## Setup

### 1. Install Python dependencies

```bash
cd /path/to/Ragger
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads the embedding model (~90MB) to your HuggingFace cache.
After that, all operations are offline.

### 2. Storage

SQLite is the default backend — zero setup, single-file database at
`~/.ragger/memories.db`. No configuration needed.

The abstract `MemoryBackend` base class makes it straightforward to add
new backends (Postgres, Qdrant, etc.) — see "Writing a Custom Backend" below.

> **Upgrading from v0.4.x:** The database location moved from
> `~/.local/share/ragger/memories.db` to `~/.ragger/memories.db`.
> Copy or move your database file to the new location.

### 3. Test

```bash
./ragger.py --store "Test memory"
./ragger.py --search "test"
./ragger.py --count
```

## Usage

### Command Line

```bash
# Store a memory
./ragger.py --store "The deploy script requires Node 18+"

# Search (semantic — finds by meaning, not keywords)
./ragger.py --search "deployment requirements"

# Search specific collections
./ragger.py --search "API authentication" --collections docs
./ragger.py --search "error handling" --collections docs reference
./ragger.py --search "anything" --collections '*'  # all collections

# Import a text file (paragraph-aware chunking)
./ragger.py --import-file notes.md --chunk-size 500

# Import into a specific collection
./ragger.py --import-file api-docs.md --collection docs

# Import a converted PDF (use docling, pandoc, etc. to get text first)
# docling myfile.pdf --to md -o .
./ragger.py --import-file myfile.md --chunk-size 500

# Count stored memories
./ragger.py --count
```

### File Import Notes

Any text file works: `.md`, `.txt`, `.log`, `.csv`, etc. For PDFs, DOCX,
and other binary formats, convert to text first with a tool like
[docling](https://github.com/DS4SD/docling), pandoc, or pdftotext.

**File size:** No practical limit when using `--chunk-size` — files are
split at paragraph boundaries (`\n\n`) and stored as separate documents.
Without chunking, each file becomes one document. For anything longer than
a page or two, use `--chunk-size`.

**Chunk size:** Smaller chunks (~300-500 chars, roughly one paragraph)
produce better search results — the embedding captures a focused idea
rather than a diluted page. Default minimum is 300. Short paragraphs are
merged until the minimum is reached; paragraphs are never split
mid-sentence. Going much smaller risks losing context (pronouns without
antecedents, etc.).

### HTTP Server

```bash
# Run HTTP server (for OpenClaw plugin or any HTTP client)
./ragger.py --server                        # default: 127.0.0.1:8432
./ragger.py --server --port 9000            # custom port
./ragger.py --server --host 0.0.0.0         # bind to all interfaces

# Run MCP server (JSON-RPC over stdio)
./ragger.py --mcp

# Run MCP server (also accepts plain text queries interactively)
./ragger.py --mcp
```

#### MCP Plain Text Mode

The MCP server accepts both JSON-RPC and plain text input. If a line
isn't JSON, it's treated as a search query and results are returned
as readable plain text:

```
> instrument ranges
1. [score: 0.523] (Orchestration Guide.md) [reference]
   The clarinet has a written range from E3 to C7...

Timing: 12.3ms (10614 chunks)
```

JSON-RPC and plain text can be interleaved freely in the same session.
For one-shot scripting, use `--search` instead (loads model, runs query,
exits).

#### Endpoints

```
GET  /health  — {"status": "ok", "memories": 42}
GET  /count   — {"count": 42}
POST /store   — {"text": "...", "metadata": {...}}  →  {"id": "...", "status": "stored"}
POST /search  — {"query": "...", "limit": 5, "min_score": 0.0, "collections": ["memory"]}
             →  {"results": [...], "timing": {...}}
```

The `collections` parameter filters which memory pools to search. Omit it
to search only the default `memory` collection. Pass `["*"]` to search
everything.

### Python API

```python
from ragger_memory import RaggerMemory

with RaggerMemory() as memory:
    memory.store("Some fact", metadata={"source": "notes.md"})
    results = memory.search("related query", limit=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['text']}")
```

## OpenClaw Integration

Ragger integrates with [OpenClaw](https://github.com/openclaw/openclaw) as a
memory plugin via the HTTP server.

### 1. Install the OpenClaw plugin

Copy the `memory-ragger` plugin directory into `~/.openclaw/extensions/`:

```
~/.openclaw/extensions/memory-ragger/
├── openclaw.plugin.json
└── index.ts
```

### 2. Configure OpenClaw

In `~/.openclaw/openclaw.json`, set the memory plugin slot:

```json
{
  "plugins": {
    "slots": {
      "memory": "memory-ragger"
    },
    "entries": {
      "memory-ragger": {
        "enabled": true,
        "config": {
          "serverUrl": "http://localhost:8432",
          "autoRecall": true,
          "autoCapture": true,
          "serverCommand": "/path/to/Ragger/.venv/bin/python3",
          "serverArgs": ["/path/to/Ragger/ragger.py", "--server"]
        }
      }
    }
  }
}
```

When `serverCommand` is set, the plugin automatically starts the Ragger
HTTP server if it's not already running — and stops it when OpenClaw shuts
down. No separate launchd plist or systemd unit needed. The plugin waits
up to 15 seconds for the server to become ready (model loading takes a
few seconds on first start).

If you prefer to manage the server yourself (launchd, systemd, manual),
just omit `serverCommand` and start it however you like.

This gives the agent three tools:
- **memory_search** — Semantic search over stored memories
- **memory_store** — Save new memories with optional metadata
- **memory_get** — Get the count of stored memories

With `autoRecall` enabled, relevant memories are automatically injected
into context before each agent turn. With `autoCapture`, important user
messages are stored automatically after conversations.

## Collections

Memories are organized into **collections** — logical groups that let you
separate reference material from conversation memories and search the
right pool for the right question.

### Built-in Collections

| Collection | Purpose |
|------------|---------|
| `memory` | Agent-stored memories: facts, decisions, preferences, session summaries (default) |

### Example Custom Collections

| Collection | Purpose |
|------------|---------|
| `docs` | Project documentation, API references |
| `reference` | Technical manuals, specifications |
| `notes` | Meeting notes, research, bookmarks |
| `work` | Work-specific context, procedures |

### Collections vs Categories

**Collections** are the top-level partition — they control which pool of
documents gets searched. Every memory belongs to exactly one collection.

**Categories** are metadata *within* a collection — they describe what
kind of memory it is (fact, decision, preference, session-summary, etc.).
Categories are stored in `metadata.category` and are useful for filtering
or organizing, but don't affect which documents are included in search.

```
memory (collection)
├── fact (category)
├── decision (category)
├── preference (category)
├── lesson (category)
└── session-summary (category)

docs (collection) — imported reference material
orchestration (collection) — imported reference material
```

### How It Works

Every memory has a `collection` field in its metadata. When you search,
you can specify which collections to include:

- **Default (no `collections` param):** Searches all collections.
- **Specific collections:** `--collections docs reference` searches both.
- **Everything (explicit):** `--collections '*'` searches all collections.

Use explicit collection lists when you want to narrow results — for
example, searching only `memory` to find your own notes without noise
from imported reference docs.

### Tagging at Import

```bash
# Import reference docs into a named collection
./ragger.py --import-file api-guide.md --collection docs

# Import without --collection → defaults to "memory"
./ragger.py --import-file meeting-notes.md
```

### Working with Your AI

The real power of collections shows up when an AI agent uses Ragger as
its memory backend. The agent learns what you're working on from context
and searches the right collections automatically:

- Working on your API? The agent includes `docs` in its search.
- Asking about deployment procedures? It searches `reference`.
- General conversation? Just `memory` — no noise from reference docs.

You don't need to tell the agent which collection to use. It figures it
out the same way a good assistant would — by paying attention.

**To customize this for your setup:** Work with your AI to define
collections that match your workflow. Import your reference materials
with `--collection <name>`, and let the agent know what's available.
The agent can then decide which bookshelves to pull from based on what
you're discussing.

This is a collaborative process — your AI learns your collections over
time and gets better at knowing when to reach for reference material
versus conversation history.

## Project Structure

```
Ragger/
├── ragger_memory/              # Python package
│   ├── __init__.py             # Package exports
│   ├── backend.py              # MemoryBackend ABC (NumPy cosine similarity)
│   ├── sqlite_backend.py       # SqliteBackend (SQLite)
│   ├── bm25.py                 # Pure Python BM25 (Okapi) implementation
│   ├── config.py               # Configuration (engine, model, defaults)
│   ├── embedding.py            # Embedder class (sentence-transformers)
│   ├── memory.py               # RaggerMemory facade/factory
│   ├── server.py               # HTTP server
│   ├── mcp_server.py           # MCP JSON-RPC server
│   └── cli.py                  # Command-line interface
├── ragger.py                   # Entry point (chmod +x)
├── requirements.txt            # Python dependencies
└── README.md
```

### Backend Architecture

Storage backends inherit from `MemoryBackend` (in `backend.py`), which
provides the default NumPy-based cosine similarity search. Each backend
implements:

- `store_raw()` — Persist a document with text, embedding, and metadata
- `load_all_embeddings()` — Load all embeddings for brute-force search
- `count()` — Return the number of stored documents
- `close()` — Clean up connections

The base class handles search via `load_all_embeddings()` + NumPy cosine
similarity, plus hybrid BM25 blending, collection filtering, usage
tracking hooks, and query logging.

### Writing a Custom Backend

To add a new storage backend (e.g., Postgres, Qdrant):

1. Create a new file (e.g., `ragger_memory/postgres_backend.py`)
2. Inherit from `MemoryBackend` in `backend.py`
3. Implement the four required methods:

```python
from .backend import MemoryBackend

class PostgresBackend(MemoryBackend):
    def __init__(self, embedder, connection_string=None):
        super().__init__(embedder)
        # Your connection setup here

    def store_raw(self, text, embedding, metadata, timestamp) -> str:
        """Persist a document. Return its ID as a string."""
        ...

    def load_all_embeddings(self) -> tuple[list, list, np.ndarray, list, list]:
        """Return (ids, texts, embeddings_matrix, metadata_list, timestamps)."""
        ...

    def count(self) -> int:
        """Return total document count."""
        ...

    def close(self):
        """Clean up connections."""
        ...
```

4. Register it in `memory.py`'s factory and add config to `config.py`

The base class provides vector search (NumPy cosine similarity), hybrid
BM25 blending, collection filtering, usage tracking hooks, and query
logging — your backend only needs to handle storage and retrieval.

For the C++ port, backends are compiled in rather than dynamically loaded.

## How It Works

1. **Store:** Text → 384-dim vector via sentence-transformers → backend
   document with text, embedding, metadata, and timestamp.

2. **Search:** Query → vector → hybrid scoring (NumPy cosine similarity +
   BM25 keyword relevance) → results ranked by blended score.

3. **Performance:** 50K vectors × 384 dims ≈ 10-50ms on Apple Silicon.
   The embedding model stays loaded in server mode for fast repeated queries.

## RAG Architecture

Ragger uses **hybrid RAG with BM25 + dense retrieval** — combining keyword
matching with semantic vector search for better recall:

1. **Indexing:** Documents are split into paragraph-sized chunks (short
   paragraphs merged to a minimum size, never split mid-sentence). Each
   chunk is embedded into a 384-dimensional vector using a local
   sentence-transformer model, then stored alongside the original text
   and metadata.

2. **Retrieval:** At query time, the query is embedded with the same model.
   Two scores are computed for each document:
   - **Vector score:** Cosine similarity via NumPy (semantic meaning)
   - **BM25 score:** Okapi BM25 keyword relevance (exact term matching)

   Both scores are min-max normalized to [0,1], then blended with
   configurable weights (default: 70% vector, 30% BM25). Top-k results
   are returned ranked by the blended score; the reported score remains
   raw cosine similarity for consistency.

3. **Generation:** Retrieved results are injected into the LLM's context
   window as reference material. The LLM generates its response grounded
   in the retrieved text.

The BM25 implementation is pure Python (no external dependencies). The
BM25 index is persisted in a SQLite table (`bm25_index`) and loaded on
first search. New documents are indexed automatically on store. Use
`--rebuild-bm25` to rebuild the index after migration or bulk changes.

Hybrid search can be toggled and tuned via config:

| Setting | Default | Description |
|---------|---------|-------------|
| `BM25_ENABLED` | `True` | Enable/disable BM25 blending |
| `BM25_WEIGHT` | `0.3` | Weight for BM25 score in blend |
| `VECTOR_WEIGHT` | `0.7` | Weight for vector score in blend |
| `BM25_K1` | `1.5` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 document length normalization |

See [ROADMAP.md](ROADMAP.md) for potential upgrades and future plans.

## Configuration

Edit `ragger_memory/config.py` or set environment variables:

| Setting | Default | Env Var |
|---------|---------|---------|
| Storage engine | `sqlite` | — |
| SQLite path | `~/.ragger/memories.db` | — |
| Embedding model | `all-MiniLM-L6-v2` | — |
| HF cache | `~/.cache/huggingface` | `SENTENCE_TRANSFORMERS_HOME` |
| Server host | `127.0.0.1` | — |
| Server port | `8432` | — |
| BM25 enabled | `True` | — |
| BM25 weight | `0.3` | — |
| Vector weight | `0.7` | — |
| Path normalization | `True` | — |
| Query logging | `True` | — |

## Database Schema

```sql
CREATE TABLE memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL,    -- 384 × float32 = 1536 bytes
    timestamp   TEXT NOT NULL,     -- ISO 8601
    metadata    TEXT              -- JSON (includes "collection" field)
);

CREATE TABLE memory_usage (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL REFERENCES memories(id)
                ON DELETE CASCADE ON UPDATE CASCADE,
    timestamp   TEXT NOT NULL
);

CREATE TABLE bm25_index (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id   INTEGER NOT NULL REFERENCES memories(id)
                ON DELETE CASCADE ON UPDATE CASCADE,
    token       TEXT NOT NULL,
    term_freq   INTEGER NOT NULL
);
```

Imported document chunks include heading context prepended to the text
(full heading chain from the document hierarchy) and a `section` breadcrumb
using `»` separators for deeper nesting.

## Query Logging

Search queries are logged to a separate collection/table for quality
analysis. Each query records timing, result scores, and quality metrics.

```sql
-- query_log table
-- Each row stores a JSON blob with query details and timing
CREATE TABLE query_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,     -- ISO 8601
    query     TEXT NOT NULL,     -- the search query
    results   TEXT,              -- JSON: scores, sources, quality metrics
    timing    TEXT               -- JSON: embedding_ms, search_ms, total_ms, corpus_size
);
```

Example `results` JSON:

```json
{
  "query": "authentication flow",
  "limit": 3,
  "num_results": 3,
  "top_score": 0.6461,
  "score_gap": 0.0415,
  "below_threshold": false,
  "results": [
    {"chunk_id": "42", "score": 0.6461, "source": "API Guide.md", "chunk_size": 415}
  ]
}
```

Query logging can be toggled via `QUERY_LOGGING_ENABLED` in `config.py`.
Logging failures are caught silently — they never break search operations.

## Best Practices for AI Agents

If you're using Ragger as memory for an AI agent (OpenClaw, Claude,
or any framework), these patterns help the agent work effectively
across sessions.

### Read Project Docs First

Before working on any project, the agent should read its documentation
(README, ROADMAP, CLAUDE.md, etc.) — not rely solely on memory search.
Memory stores summaries; project files have the authoritative details.

### Store Decisions Separately

Don't bury technical decisions inside session summaries. Library choices,
architecture decisions, and design rationale should be stored as their own
memory entries with specific tags so they're findable later:

```bash
# Bad: buried in a session summary
"Session 2026-03-17: discussed architecture, chose Crow for HTTP..."

# Good: standalone decision
"Ragger C++ port: chose Crow for HTTP (FetchContent), Eigen for vectors,
ONNX Runtime for embeddings. Rationale: Crow is Flask-like routing,
Eigen replaces NumPy, ONNX avoids Python dependency."
```

### Reference the Source

When storing a decision, include where the details live (file paths,
commit hashes). The memory entry is a pointer; the project file is
the source of truth.

### Usage Scenarios

**Solo developer + AI assistant (local):**
- One Ragger instance at `~/.ragger/memories.db`
- Agent stores conversation context, decisions, and lessons learned
- Import reference docs into named collections (`--collection docs`)
- Agent searches `memory` by default, reaches into `docs`/`reference`
  when the question calls for it

**Solo developer + multiple AI tools:**
- Same Ragger server (port 8432) shared across tools
- OpenClaw, CLI scripts, editor plugins all use the same HTTP API
- Collections separate concerns: `memory` for agent notes, `docs` for
  reference material, `work` for project-specific context

**Team / shared server:**
- Ragger runs as a system service (see ROADMAP.md for multi-user plans)
- Per-user memory via auth token → user isolation
- Shared reference collections available to all users
- Private memories stay private

**Offline / air-gapped:**
- Everything runs locally — no network calls
- Download the embedding model once, then disconnect
- SQLite database is a single file — easy to backup, move, or encrypt

**Development + production split:**
- Dev instance for experimentation, separate prod database
- Export/import via CLI for promoting curated memories
- Same binary, different `--db` paths

### Collection Strategy

Start simple and add collections as needs emerge:

| Stage | Collections |
|-------|-------------|
| Getting started | Just `memory` (the default) |
| Adding reference docs | `memory` + `docs` |
| Multiple doc sources | `memory` + `sibelius` + `orchestration` + ... |
| Team use | Per-user `memory` + shared `reference` |

The agent should know what collections exist and when to search them.
Store this knowledge as a memory entry so it persists across sessions.

## Conversation Memory Lifecycle

AI agents lose context between sessions and during compaction (context
window compression). Ragger can serve as persistent memory, but *how*
conversations get captured matters as much as *that* they're captured.

### The Problem

Raw conversation transcripts are verbose, full of false starts and
filler. Storing them verbatim dilutes search quality. But waiting too
long to summarize risks losing the conversation entirely (session
timeout, compaction, crash).

### A Practical Pattern

1. **Buffer** — Store conversation substance into a `conversation`
   collection as it happens. Keep it lightweight — decisions, questions,
   answers, not "Great question! Let me think about that."

2. **Summarize on pause** — After a period of inactivity (20 minutes
   works well), summarize the buffered conversation. Extract decisions,
   facts, lessons, and action items into proper memory entries in the
   `memory` collection with appropriate categories. Delete the raw
   conversation chunks.

3. **Summarize before compaction** — If the agent's context window is
   getting full, proactively summarize and store before the runtime
   compresses it. Compaction is lossy — it keeps what the summarizer
   thinks matters, which may not be what you care about later.

4. **Store decisions immediately** — Don't wait for summarization.
   Library choices, architecture decisions, design rationale — store
   these as they happen with specific tags. They're the most valuable
   and most easily lost.

### Open Questions

The right approach depends on your use case and is worth experimenting
with:

- **What's the right pause interval?** Too short and you're
  summarizing mid-thought. Too long and compaction gets there first.
  20 minutes is a starting point.

- **What level of detail to keep?** A summary of "we discussed the
  database schema" is useless. "Chose SQLite over Postgres because
  single-file deployment matters more than concurrent writes" is
  valuable. The summarizer needs enough context to know what matters.

- **Who summarizes?** The same agent that had the conversation has
  the best context. A separate summarization job has less context but
  can use a cheaper/faster model. Trade-offs either way.

- **How to handle multi-topic conversations?** A single conversation
  might cover three unrelated projects. Each topic should become its
  own memory entry with its own tags, not one monolithic summary.

- **Conversation collection search policy?** Should `conversation`
  be searched by default, or only explicitly? Raw buffer chunks are
  noisy — excluding them from default search keeps results clean
  while the data is still available when needed.

These are active design questions. If you find patterns that work well,
consider contributing them back.

## License

GPL v3 — See [LICENSE](LICENSE) for details.
