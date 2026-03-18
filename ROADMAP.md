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

- **Native vector search:** Purpose-built vector databases like SQLite vector
  extensions would move similarity search into the database
  engine, eliminating the need to load all embeddings into Python. Worthwhile
  for a very large corpora (100K+).

None of these are necessary at a small scale (<50K chunks), but they
become worthwhile as your corpus grows or retrieval precision becomes
critical.

## v0.6.0 (Planned)

- Remove deprecated `--convert` CLI command (migration tooling no longer needed)

## Future

- **C++ port (`raggerc`):** Standalone binary with two modes:
  - `ragger serve` — memory + LLM backend for OpenClaw or other agent
    frameworks. No workspace files, no agent loop — just HTTP endpoints.
  - `ragger chat` — standalone agent with full tool loop, memory, and LLM.
    Loads workspace MD files from `~/.ragger/` (SOUL.md, USER.md, AGENTS.md,
    MEMORY.md, TOOLS.md) — same conventions as OpenClaw's workspace.
  - Stack: Crow (HTTP, Boost.Asio), Eigen (linalg), ONNX Runtime (embeddings),
    tokenizers-cpp (HuggingFace tokenizer), nlohmann/json, SQLite (storage),
    llama.cpp (LLM). Boost ProgramOptions via c_lib.
  - Cross-platform (macOS + Linux). Same database format as Python version.

- **Multi-user support:** Per-user memory in home directories (`~/.ragger/`),
  shared RAG in a common location, filesystem permissions for access control.

- **Thin CLI client:** HTTP request to running server instead of loading
  the embedding model. Sub-100ms queries from the terminal.
