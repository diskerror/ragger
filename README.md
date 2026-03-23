# Ragger Memory

Semantic memory store with local embeddings and pluggable storage backends.
Designed as a long-term memory backend for AI agents (OpenClaw, etc.) but
usable standalone.

No external APIs, no cloud services — **everything runs locally**.

A [C++ port (ragger)](https://github.com/diskerror/ragger.cpp) is also
available with the same HTTP API, database format, and config file.

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

## Status

**Version 0.7.0** — Single-user, fully functional.

Multi-user framework is in place (layered config with system/user INI files,
SERVER_LOCKED keys, system ceilings on user settings, token auth, persona file
hierarchy) but the data layer is still single-user: one database, one user,
no user routing. Multi-user data support (users table, per-user DBs, search
merging) is planned for a future release.

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
ragger store "Test memory"
ragger search "test"
ragger count
```

## Usage

### Command Line

Ragger uses verb-style commands: `ragger <verb> [options] [args]`.
No verb or `help` prints usage.

```bash
# Start HTTP server
ragger serve
ragger serve --host 0.0.0.0 --port 9000

# Store a memory
ragger store "The deploy script requires Node 18+"

# Search (semantic — finds by meaning, not keywords)
ragger search "deployment requirements"

# Search specific collections
ragger search "API authentication" --collections docs
ragger search "error handling" --collections docs reference

# Import a text file (paragraph-aware chunking)
ragger import notes.md --collection docs

# Import multiple files
ragger import doc1.md doc2.md --collection reference

# Import a converted PDF (use docling, pandoc, etc. to get text first)
ragger import myfile.md --min-chunk-size 500

# Export documents
ragger export docs ./exported/ --collection orchestration
ragger export memories ./exported/
ragger export all ./exported/

# Count stored memories
ragger count

# Rebuild BM25 index
ragger rebuild-bm25

# Run MCP server (JSON-RPC over stdin/stdout)
ragger mcp

# Download/update embedding model
ragger update-model

# Help
ragger help
ragger          # no verb = help
```

### File Import Notes

Any text file works: `.md`, `.txt`, `.log`, `.csv`, etc. For PDFs, DOCX,
and other binary formats, convert to text first with a tool like
[docling](https://github.com/DS4SD/docling), pandoc, or pdftotext.

**File size:** No practical limit when using `--min-chunk-size` — files are
split at paragraph boundaries (`\n\n`) and stored as separate documents.
Without chunking, each file becomes one document. For anything longer than
a page or two, use `--min-chunk-size`.

**Chunk size:** Smaller chunks (~300-500 chars, roughly one paragraph)
produce better search results — the embedding captures a focused idea
rather than a diluted page. Default minimum is 300. Short paragraphs are
merged until the minimum is reached; paragraphs are never split
mid-sentence. Going much smaller risks losing context (pronouns without
antecedents, etc.).

### HTTP Server

```bash
# Run HTTP server (for OpenClaw plugin or any HTTP client)
ragger serve                        # default: 127.0.0.1:8432
ragger serve --port 9000            # custom port
ragger serve --host 0.0.0.0         # bind to all interfaces

# Run MCP server (JSON-RPC over stdio)
ragger mcp
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

See [OPENCLAW.md](OPENCLAW.md) for setup and configuration.

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
ragger import api-guide.md --collection docs

# Import without --collection → defaults to "memory"
ragger import meeting-notes.md
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
│   ├── config.py               # Config file loader (INI format)
│   ├── embedding.py            # Embedder class (sentence-transformers)
│   ├── memory.py               # RaggerMemory facade/factory
│   ├── server.py               # HTTP server
│   ├── mcp_server.py           # MCP JSON-RPC server
│   ├── cli.py                  # Verb-style CLI
│   └── lang/                   # i18n language strings
│       ├── __init__.py         # Language selector
│       └── en.py               # English strings
├── example-system.ini          # System config example (hard-linked with C++ project)
├── example-user.ini            # User config example (hard-linked with C++ project)
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
`rebuild-bm25` to rebuild the index after migration or bulk changes.

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

Ragger uses layered INI config files:

1. **System config:** `/etc/ragger.ini` (or `--config=<path>` to override)
2. **User config:** `~/.ragger/ragger.ini` (always read on top)

System config sets infrastructure (host, port, DB path, embedding model,
inference endpoints). User config sets personal preferences (search limits,
chat settings, default model). SERVER_LOCKED keys in system config can't be
overridden by users. System ceilings cap user values (e.g., `max_search_limit`
limits how high a user can set `default_limit`).

If no config exists, first run auto-creates `~/.ragger/ragger.ini` with defaults.

See `example-system.ini` and `example-user.ini` for all options.
The same config format works for both the Python and C++ versions.

```ini
# System config (/etc/ragger.ini) — infrastructure
[server]
host = 127.0.0.1
port = 8432

[storage]
db_path = ~/.ragger/memories.db
default_collection = memory

[embedding]
model = all-MiniLM-L6-v2
dimensions = 384

[search]
default_limit = 5
default_min_score = 0.4
bm25_enabled = true
; Weights are ratios: "3 and 7" = "0.3 and 0.7"
bm25_weight = 3
vector_weight = 7
# max_search_limit = 0    # 0 = no ceiling

[chat]
# System hard limits
max_turn_retention_minutes = 60
max_turns_stored = 100
# max_persona_chars_limit = 0
# max_memory_results_limit = 0
```

```ini
# User config (~/.ragger/ragger.ini) — personal preferences
[search]
default_limit = 10

[inference]
model = qwen/qwen2.5-coder-14b

[chat]
store_turns = true
max_persona_chars = 4000    # limit context for local models
max_memory_results = 2
```

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

Search queries are logged to `~/.ragger/query.log` as single-line JSON
entries with timing, result scores, and quality metrics. Enable/disable
via the `query_log` setting in the config file. Logging failures are
caught silently — they never break search operations.

## AI Agent Best Practices

See [README_TO_AGENT.md](README_TO_AGENT.md) for usage scenarios,
collection strategy, conversation memory lifecycle, and best practices
for AI agents using Ragger as long-term memory.

## Installation

### Per-user install (single user, no sudo)

| Platform | Executable location | Config location |
|----------|-------------------|-----------------|
| macOS    | `~/.local/bin/ragger` | `~/.ragger/ragger.ini` |
| Linux    | `~/.local/bin/ragger` | `~/.ragger/ragger.ini` |
| Windows  | `%LOCALAPPDATA%\ragger\ragger.exe` | `%LOCALAPPDATA%\ragger\ragger.ini` |

On macOS/Linux, ensure `~/.local/bin` is in your `PATH`:
```bash
export PATH="$HOME/.local/bin:$PATH"  # add to ~/.zshrc or ~/.bashrc
```

Install as the default `ragger` command:
```bash
mkdir -p ~/.local/bin
cat > ~/.local/bin/ragger << 'EOF'
#!/bin/bash
RAGGER_PY_DIR="${RAGGER_PY_DIR:-$HOME/PyCharmProjects/Ragger}"
exec python3 "$RAGGER_PY_DIR/ragger_memory/cli.py" "$@"
EOF
chmod +x ~/.local/bin/ragger
```

If the C++ version is also installed, this can coexist as `ragger-py`
while the C++ binary is the default `ragger`.

### System-wide install (future, multi-user)

Reserved for future multi-user support. Will use `/usr/local/bin/ragger`,
`/etc/ragger.ini`, and `/var/ragger/` for data.

## macOS Deployment Note

When running Ragger as a LaunchDaemon (i.e., starting at boot before any user logs in),
be aware that if the user's home directory is on an external or non-default volume, that
volume may not be mounted until a user session starts. In this case, you may need to
enable **System Settings → Users & Groups → Automatically log in as…** for the
relevant user account to ensure the volume is available at boot.

A start script that waits for the volume to mount (with a timeout) can help, but is not
a substitute for the volume actually being mounted by the system.

## License

GPL v3 — See [LICENSE](LICENSE) for details.
