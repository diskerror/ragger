# Ragger Memory — Roadmap

## Current: v0.6.0

Both Python and C++ versions at feature parity. C++ is production server.

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

## Multi-User Architecture (v0.7.0+)

### System-Wide (`/var/ragger/` or platform equivalent)
- `ragger.ini` — system config (port, host, model path, search params, cache TTL)
- `memories.db` — shared/proprietary documentation. Tagged with `user_id`.
  Contains the `users` table (token → user mapping). Only DB with user
  management schema.
- `models/` — shared ONNX model files
- Logs

### Per-User (`~/.ragger/`)
- Created automatically on user's first run
- `ragger.ini` — personal additions (extends system config, does not override)
- `memories.db` — private conversations, personal memories. No `users` table,
  no `user_id` column — it's implicitly single-user.
- `token` — bearer auth token, permissions 0660
- `USER.md` — about this user (preferences, working style)
- `AGENTS.md` — per-user agent customization
- Workspace files as needed

### Search Behavior
- **Every search queries both DBs** (user + common), same query, same scoring.
- Results merge into a single ranked list — no artificial weighting or bias.
- Common DB (mostly proprietary docs) naturally returns low/no relevance for
  personal memory lookups. The scores do the filtering organically.
- No special flags or opt-in needed. Simple implementation, correct behavior.

### Authentication
- **API clients**: Bearer token in `Authorization` header. Token generated on
  first run, stored in `~/.ragger/token` (0660). Daemon maps token → username
  via `users` table in common DB. The user never sees or manages their user_id.
- **Browser (future `ragger chat` web UI)**: Hashed password + session cookies.
  Separate auth concern from API tokens.
- **Token rotation**: Configurable `token_rotation_minutes` (default 1440 = 24h,
  0 = never). Daemon checks token age on each request; when expired, generates
  new token, writes to `~user/.ragger/token`, updates hash in users table.
  Client re-reads token on 401 and retries once. Prevents tokens in old backups
  from being usable.

### Runtime Model
- **Single process**, single port, single loaded embedding model
- If ONNX embedding becomes a bottleneck, break it out into child processes
- Per-user backend cached with configurable TTL (default 12 hours)
- Multiple concurrent users, each with their own conversation context
- Code laid out for easy llama.cpp integration later

### System Setup
- System-level resources (`/var/ragger/`, `_ragger` user, LaunchDaemon)
  require `sudo` and are handled by the install script (`install-system.sh`).
- User-level resources (`~/.ragger/`) are created automatically on first run.
  No sudo needed.
- Ragger should be as self-contained as possible.

### Collaborative Features
- Users can store to common DB tagged with `user_id` and project (`collection`)
- Project-scoped search: filter by collection for team context
- Move/promote between private and common memory
- Only admin users can modify system-wide files through the agent
- System files also changeable manually with root/ragger user perms

### Permissions
- Daemon runs as `_ragger` (macOS) / `ragger` (Linux)
- System directory owned by `_ragger:ragger`
- Each user's `~/.ragger/` is 0750, owned by user with ragger group
- Project-level permissions (future): who can read/write which collections

## Modes

- **`ragger serve`** — memory service + inference proxy. HTTP endpoints for
  store, search, import, export, and chat completions. For OpenClaw or other
  agent frameworks.
- **`ragger chat`** — thin REPL client. Searches memory for context, injects
  it into the prompt, streams response from configured inference API.
  No tool calling, no agent loop. Simple chat with memory.

Group collaboration happens through existing agent frameworks (e.g.
OpenClaw → Telegram). Users add their agent to a group chat; the agent
uses Ragger for memory.

## Workspace Files

Persona and user customization via markdown files, matching OpenClaw conventions.

**Common** (`/var/ragger/`):
- `SOUL.md` — shared persona. Everyone talks to the same assistant.

**Per-user** (`~/.ragger/`):
- `SOUL.md` — fallback persona (only if no common SOUL.md exists)
- `USER.md` — about this person (name, preferences, working style)
- `AGENTS.md` — how the assistant works for them
- `TOOLS.md` — their specific setup (hosts, devices, etc.)

## Thin CLI Client

HTTP request to running server instead of loading the embedding model.
Sub-100ms queries from the terminal.

## Common Shared Files

Configurable flat file storage location for shared documents, summaries,
and reference material. Common DB stores the filename and which user
pushed it (provenance tracking).

Users can push conversation summaries or project context to common
memory using existing commands (`store --common`) or a dedicated
`push` verb.

## Memory Quality

Bad or obsolete memories are **marked**, not deleted. A memory flagged
as "bad" or "wrong direction" is kept because it may prevent repeating
the same mistake. This mirrors how institutional memory works — failed
experiments are as valuable as successes.
