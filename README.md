# Ragger Memory

Semantic memory store with local embeddings and pluggable storage backends.
Designed as a long-term memory backend for AI agents (OpenClaw, etc.) but
usable standalone.

No external APIs, no cloud services — everything runs locally.

## Features

- **Local embeddings** — `all-MiniLM-L6-v2` via sentence-transformers (384-dim, ~90MB)
- **Hybrid search** — BM25 keyword + vector cosine similarity (pure Python, configurable blend)
- **Fast vector search** — NumPy cosine similarity (~10-50ms for 50K documents)
- **Pluggable backends** — MongoDB or SQLite (easy to add more)
- **HTTP server** — REST API on localhost for tool integration
- **MCP server** — JSON-RPC over stdin/stdout (Model Context Protocol)
- **Collection filtering** — Organize memories into searchable collections (e.g. `docs`, `reference`, `memory`)
- **Usage tracking** — Per-memory access stats for identifying high-value content
- **Path normalization** — `$HOME` → `~/` for portable, privacy-friendly storage
- **CLI tools** — Store, search, import files, count memories
- **Python API** — Reusable `RaggerMemory` class
- **Query logging** — Track searches with timing, scores, and quality metrics

## Requirements

- Python 3.10+
- ~1GB disk for model + dependencies
- **MongoDB backend:** MongoDB 6+ (running locally, no auth required)
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

### 2. Choose a storage backend

Edit `ragger_memory/config.py`:

```python
STORAGE_ENGINE = "sqlite"  # or "mongodb"
```

**MongoDB** — requires a running `mongod` instance:
```bash
mongosh --eval "db.runCommand({ping:1})"
```

**SQLite** — zero setup, single-file database at `~/.local/share/ragger/memories.db`.

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
```

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
| `memory` | Conversation memories, agent notes, decisions (default) |

### Example Custom Collections

| Collection | Purpose |
|------------|---------|
| `docs` | Project documentation, API references |
| `reference` | Technical manuals, specifications |
| `notes` | Meeting notes, research, bookmarks |
| `work` | Work-specific context, procedures |

### How It Works

Every memory has a `collection` field in its metadata. When you search,
you specify which collections to include:

- **Default (no `collections` param):** Only searches `memory` — your
  conversation memories and agent-stored notes.
- **Specific collections:** `--collections docs reference` searches both.
- **Everything:** `--collections '*'` searches all collections.

This prevents reference documents (manuals, textbooks) from drowning out
your actual memories in general searches, while still making them available
when you need them.

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
│   ├── bm25.py                 # Pure Python BM25 (Okapi) implementation
│   ├── config.py               # Configuration (engine, URIs, model, defaults)
│   ├── embedding.py            # Embedder class (sentence-transformers)
│   ├── memory.py               # RaggerMemory facade/factory (lazy backend import)
│   ├── server.py               # HTTP server
│   ├── mcp_server.py           # MCP JSON-RPC server
│   ├── cli.py                  # Command-line interface
│   └── backend/                # Storage backends
│       ├── __init__.py
│       ├── base.py             # MemoryBackend ABC (NumPy cosine similarity)
│       ├── mongo.py            # MongoBackend (MongoDB)
│       └── sqlite.py           # SqliteBackend (SQLite)
├── ragger.py                   # Entry point (chmod +x)
├── requirements.txt            # Python dependencies
└── README.md
```

### Backend Architecture

Storage backends inherit from `MemoryBackend` (in `backend/base.py`), which
provides the default NumPy-based cosine similarity search. Each backend
implements:

- `store()` — Persist a document with text, embedding, and metadata
- `get_all_embeddings()` — Load all embeddings for brute-force search
- `count()` — Return the number of stored documents
- `log_query()` — Record search queries for quality analysis

The base class handles search via `get_all_embeddings()` + NumPy cosine
similarity. Backends can override `search()` if they support native vector
search (e.g., MongoDB Atlas `$vectorSearch`).

Adding a new backend (Qdrant, Pinecone, etc.) requires implementing these
four methods and adding a config section in `config.py`.

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
BM25 index is built from cached texts on first search (~230ms for 14K
docs) and reused across subsequent searches.

Hybrid search can be toggled and tuned via config:

| Setting | Default | Description |
|---------|---------|-------------|
| `BM25_ENABLED` | `True` | Enable/disable BM25 blending |
| `BM25_WEIGHT` | `0.3` | Weight for BM25 score in blend |
| `VECTOR_WEIGHT` | `0.7` | Weight for vector score in blend |
| `BM25_K1` | `1.5` | BM25 term frequency saturation |
| `BM25_B` | `0.75` | BM25 document length normalization |

### Potential Upgrades

These are natural next steps if you want to improve retrieval quality:

- **Re-ranking:** After retrieving top-k candidates, re-score them with a
  cross-encoder model (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) for
  more accurate relevance ranking. Slower per query but significantly
  better precision.

- **Query expansion / HyDE:** Generate a hypothetical answer to the query
  using the LLM, then embed *that* for retrieval. Often finds results that
  the original terse query would miss.

- **Memory lifecycle (decay/promotion):** Usage tracking is already in place.
  A natural extension would be to decay memories that are never accessed and
  promote frequently-used ones — mimicking human memory consolidation.

- **Native vector search:** MongoDB's `$vectorSearch` (via mongot) or
  purpose-built vector databases (Qdrant, etc.) would move similarity
  search into the database engine, eliminating the need to load all
  embeddings into Python. Worthwhile for very large corpora (100K+).

None of these are necessary at small scale (<50K chunks), but they
become worthwhile as your corpus grows or retrieval precision becomes
critical.

## Configuration

Edit `ragger_memory/config.py` or set environment variables:

| Setting | Default | Env Var |
|---------|---------|---------|
| Storage engine | `sqlite` | — |
| MongoDB URI | `mongodb://localhost:27017/` | — |
| MongoDB database | `ragger` | — |
| MongoDB collection | `memories` | — |
| SQLite path | `~/.local/share/ragger/memories.db` | — |
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

### MongoDB

```javascript
// ragger.memories
{
  _id: ObjectId,
  text: "The deploy script requires Node 18+",
  embedding: [0.012, -0.034, ...],  // 384-dim float array
  timestamp: ISODate("2026-02-24"),
  metadata: {
    source: "USER.md",
    category: "preference"
  }
}
```

### SQLite

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
    query       TEXT NOT NULL,
    score       REAL NOT NULL,
    timestamp   TEXT NOT NULL
);
```

Imported document chunks include heading context prepended to the text
(full heading chain from the document hierarchy) and a `section` breadcrumb
using `»` separators for deeper nesting.

## Query Logging

Search queries are logged to a separate collection/table for quality
analysis. Each query records timing, result scores, and quality metrics.

```javascript
// MongoDB: ragger.query_log (SQLite: query_log table)
{
  _id: ObjectId,
  timestamp: ISODate("2026-02-26T20:22:39.343Z"),
  query: "authentication flow",
  limit: 3,
  min_score: 0.0,
  num_results: 3,
  top_score: 0.6461,
  score_gap: 0.0415,       // difference between #1 and #2 scores
  below_threshold: false,   // true if top_score < 0.4
  results: [
    { chunk_id: "699f8b41...", score: 0.6461, source: "API Guide.md", chunk_size: 415 },
    { chunk_id: "699f8b41...", score: 0.6046, source: "Auth Reference.md", chunk_size: 309 }
  ],
  timing: {
    embedding_ms: 11.3,     // time to embed the query
    search_ms: 0.3,         // time for cosine similarity
    total_ms: 30.9,         // end-to-end
    corpus_size: 735        // chunks searched
  },
  feedback: null            // reserved for manual relevance rating
}
```

Query logging can be toggled via `QUERY_LOGGING_ENABLED` in `config.py`.
Logging failures are caught silently — they never break search operations.

## License

GPL v3 — See [LICENSE](LICENSE) for details.
