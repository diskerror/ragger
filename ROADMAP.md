# Ragger Memory — Roadmap

## Potential Upgrades

These are natural next steps to improve retrieval quality:

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

- **Native vector search:** Purpose-built vector databases (Qdrant, etc.)
  or SQLite vector extensions would move similarity search into the database
  engine, eliminating the need to load all embeddings into Python. Worthwhile
  for very large corpora (100K+).

None of these are necessary at small scale (<50K chunks), but they
become worthwhile as your corpus grows or retrieval precision becomes
critical.

## v0.6.0 (Planned)

- Remove deprecated `--convert` CLI command (migration tooling no longer needed)

## Future

- **C++ port:** Standalone binary — `ragger serve` + `ragger chat`.
  Stack: Crow (HTTP), Eigen (linalg), ONNX Runtime (embeddings),
  SQLite (storage), llama.cpp (LLM). Cross-platform (macOS + Linux).

- **Multi-user support:** Per-user memory in home directories (`~/.ragger/`),
  shared RAG in a common location, filesystem permissions for access control.

- **Thin CLI client:** HTTP request to running server instead of loading
  the embedding model. Sub-100ms queries from the terminal.
