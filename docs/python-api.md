# Python API

Ragger provides a Python library for embedding memory storage and search
into your own applications.

## Basic Usage

```python
from ragger_memory import RaggerMemory

# Initialize with default backend (SQLite)
with RaggerMemory() as memory:
    # Store a memory
    memory.store("The deploy script requires Node 18+", 
                 metadata={"category": "fact", "source": "notes.md"})
    
    # Search
    results = memory.search("deployment requirements", limit=5)
    
    # Print results
    for r in results:
        print(f"[{r['score']:.3f}] {r['text']}")
        print(f"  {r['metadata']}")
```

## RaggerMemory Class

`RaggerMemory` is a facade that loads the configured backend and provides
a unified interface for storing and searching memories.

### Constructor

```python
RaggerMemory(config_path=None)
```

**Parameters:**

- `config_path` (optional) — Path to a custom config file. If not provided,
  uses the standard config search order (`/etc/ragger.ini` → `~/.ragger/ragger.ini`).

### Methods

#### `store(text, metadata=None) -> str`

Store a memory and return its ID.

**Parameters:**

- `text` (required) — Memory content
- `metadata` (optional) — Dictionary of metadata (collection, category, source, etc.)

**Returns:** Document ID as a string.

**Example:**

```python
doc_id = memory.store(
    text="OAuth2 requires a client ID and secret",
    metadata={
        "collection": "docs",
        "category": "reference",
        "source": "api-guide.md"
    }
)
print(f"Stored as {doc_id}")
```

---

#### `search(query, limit=None, min_score=None, collections=None) -> list[dict]`

Search memories using hybrid vector + BM25 search.

**Parameters:**

- `query` (required) — Search query
- `limit` (optional) — Maximum results (defaults to config `default_limit`)
- `min_score` (optional) — Minimum cosine similarity score (defaults to config `default_min_score`)
- `collections` (optional) — List of collection names to search (defaults to all collections)

**Returns:** List of result dictionaries with fields:

- `id` — Document ID
- `text` — Memory content
- `score` — Cosine similarity score (0.0 to 1.0)
- `metadata` — Metadata dictionary
- `timestamp` — ISO 8601 timestamp

**Example:**

```python
results = memory.search(
    query="API authentication",
    limit=3,
    min_score=0.5,
    collections=["docs", "reference"]
)

for r in results:
    print(f"[{r['score']:.3f}] {r['text'][:100]}...")
```

---

#### `count() -> int`

Return the total number of stored memories.

**Example:**

```python
total = memory.count()
print(f"Total memories: {total}")
```

---

#### `close()`

Close the backend connection. Automatically called when using the
`with` statement.

**Example:**

```python
memory = RaggerMemory()
# ... use memory ...
memory.close()
```

Or use context manager:

```python
with RaggerMemory() as memory:
    # ... use memory ...
    pass  # Automatically closed
```

---

## Backend Architecture

Ragger uses a pluggable backend system. All backends inherit from
`MemoryBackend` (defined in `ragger_memory/backend.py`), which provides:

- Brute-force NumPy cosine similarity search
- Hybrid BM25 blending
- Collection filtering
- Usage tracking hooks
- Query logging

Each backend implements four methods:

1. `store_raw(text, embedding, metadata, timestamp) -> str` — Persist a document
2. `load_all_embeddings() -> tuple` — Load all embeddings for search
3. `count() -> int` — Return total document count
4. `close()` — Clean up connections

The base class handles everything else.

### Current Backends

| Backend | Storage | Status |
|---------|---------|--------|
| `SqliteBackend` | SQLite file | Default, production-ready |

### Planned Backends

- Postgres (with pgvector extension)
- Qdrant (vector database)
- Pinecone (cloud vector database)

---

## Writing a Custom Backend

To add a new storage backend:

### 1. Create a Backend Class

Create a new file (e.g., `ragger_memory/postgres_backend.py`):

```python
from .backend import MemoryBackend
import numpy as np

class PostgresBackend(MemoryBackend):
    def __init__(self, embedder, connection_string=None):
        super().__init__(embedder)
        # Your connection setup here
        self.connection_string = connection_string or "postgresql://localhost/ragger"
        self.conn = None  # Initialize your DB connection
    
    def store_raw(self, text, embedding, metadata, timestamp) -> str:
        """
        Persist a document with text, embedding, metadata, and timestamp.
        Return the document ID as a string.
        """
        # Convert embedding to bytes or array format
        # Insert into your database
        # Return the document ID
        pass
    
    def load_all_embeddings(self) -> tuple[list, list, np.ndarray, list, list]:
        """
        Load all documents for brute-force search.
        
        Returns:
            ids: List of document IDs (as strings)
            texts: List of document texts
            embeddings: NumPy array of shape (n_docs, embedding_dim)
            metadata_list: List of metadata dicts
            timestamps: List of ISO 8601 timestamp strings
        """
        # Query all documents from your database
        # Return as a tuple
        pass
    
    def count(self) -> int:
        """Return the total number of documents."""
        # SELECT COUNT(*) FROM your_table
        pass
    
    def close(self):
        """Clean up database connections."""
        if self.conn:
            self.conn.close()
```

### 2. Register the Backend

Edit `ragger_memory/memory.py` to add your backend to the factory:

```python
from .postgres_backend import PostgresBackend

class RaggerMemory:
    def __init__(self, config_path=None):
        # ... existing code ...
        
        backend_type = config.get('storage', 'backend', fallback='sqlite')
        
        if backend_type == 'sqlite':
            from .sqlite_backend import SqliteBackend
            self.backend = SqliteBackend(embedder, db_path=db_path)
        elif backend_type == 'postgres':
            connection_string = config.get('storage', 'connection_string')
            self.backend = PostgresBackend(embedder, connection_string=connection_string)
        else:
            raise ValueError(f"Unknown backend: {backend_type}")
```

### 3. Add Config Support

Edit `ragger_memory/config.py` to add config options for your backend:

```ini
[storage]
backend = postgres
connection_string = postgresql://localhost/ragger
```

### 4. Test

```python
from ragger_memory import RaggerMemory

with RaggerMemory() as memory:
    memory.store("Test memory")
    results = memory.search("test")
    print(results)
```

---

## Vector Database Integration

For large deployments (100K+ documents), consider using a vector database
with approximate nearest neighbor (ANN) search:

### Qdrant Example

```python
from .backend import MemoryBackend
from qdrant_client import QdrantClient

class QdrantBackend(MemoryBackend):
    def __init__(self, embedder, host="localhost", port=6333):
        super().__init__(embedder)
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "memories"
    
    def store_raw(self, text, embedding, metadata, timestamp) -> str:
        # Use Qdrant's upsert API
        pass
    
    def load_all_embeddings(self):
        # For hybrid search, still need to load all embeddings
        # Or implement ANN search in the base class
        pass
```

For ANN search, you'll need to override the `search()` method in the base
class to use the vector database's native search instead of brute-force
NumPy.

---

## Related

- [Search & RAG](search-and-rag.md) — How hybrid search works
- [HTTP API](http-api.md) — REST endpoints for tool integration
- [Configuration](configuration.md) — Config file reference
