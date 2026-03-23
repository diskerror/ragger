# Chat Persistence Implementation Summary

## What was implemented

### 1. Delete API (all layers)
- **sqlite_backend.py**: `delete()` and `delete_batch()` methods
- **memory.py**: Exposed delete methods through facade
- **server.py**: `DELETE /memory/<id>` and `POST /delete_batch` endpoints
- **client.py**: HTTP client methods for delete operations

### 2. Metadata Search API
- **sqlite_backend.py**: `search_by_metadata()` method to find memories by category/source
- **memory.py**: Exposed through facade
- **server.py**: `POST /search_by_metadata` endpoint
- **client.py**: HTTP client method

### 3. Three-mode store_turns
Changed `chat_store_turns` from boolean to string with three modes:

- **"true"** (default): Each user+assistant exchange stored as separate memory entry
  - Most granular, best for search
  - Creates more DB entries but enables precise retrieval
  
- **"session"**: One growing memory entry per conversation
  - Single entry updated by delete+re-store
  - Less DB overhead, good for sequential review
  
- **"false"**: No raw turn storage, summaries only
  - Minimal storage, relies entirely on periodic summaries

### 4. System Hard Limits (SERVER_LOCKED)
Added to system config only (users can't override):

- `max_turn_retention_minutes = 60`: Raw turns older than this get summarized and deleted
- `max_turns_stored = 100`: Maximum raw turn entries in DB at any time

### 5. Turn Expiration Logic
Implemented `_expire_old_turns()` in cli.py:

- Finds all memories with `category: conversation` and `source: ragger-chat`
- Checks age against `max_turn_retention_minutes`
- Checks count against `max_turns_stored`
- Batch-summarizes expired turns via LLM
- Stores summary, deletes raw turns
- Runs at:
  - Chat startup (clean up from previous sessions)
  - After each turn storage
  - On quit (before quit summary)

### 6. Configuration Updates
- **config.py**: Added new settings, changed store_turns type, added to SERVER_LOCKED
- **example-system.ini**: Documented new settings with comments
- **example-user.ini**: Explained three modes with detailed comments

## Testing performed
- Config loading verified: `chat_store_turns`, `max_turn_retention_minutes`, `max_turns_stored` all load correctly
- Default values: true, 60, 100

## Git commit
```
commit 545aa68
Chat persistence: three-mode store_turns, system hard limits, turn expiration
```

## What was NOT done (as instructed)
- Did NOT restart services
- Did NOT modify running server
- Did NOT modify C++ code
- Did NOT test actual chat (would require LM Studio running)

## Next steps (if needed)
1. Test chat with LM Studio running: `echo -e "Hi\n/quit" | python3 ragger.py chat`
2. Verify per-turn storage creates separate entries
3. Test session mode (requires config change to `store_turns = session`)
4. Test turn expiration by creating old entries and waiting

## Key design decisions
1. Delete is real deletion, not "mark bad" — raw turns are disposable after summarization
2. Summary replaces raw turns, so no information is lost
3. Session mode uses delete+re-store (simpler than building an update API)
4. Only applies to "true" mode — session and false modes don't trigger expiration
5. Metadata search is simple equality check (not pattern matching)
