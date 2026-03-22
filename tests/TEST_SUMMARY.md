# Test Suite Summary

## New Test Files Created

### 1. test_config.py (17 tests)
Tests for configuration loading and search order.

**Coverage:**
- Config file search order (CLI arg → ~/.ragger/ragger.ini → bootstrap)
- Bootstrap creation of default config on first run
- INI file parsing and value types (int, float, bool, string)
- Custom values overriding defaults
- Module-level config initialization and lazy loading
- Path expansion (~/ to $HOME)
- Backward compatibility via __getattr__

**Test Classes:**
- `TestConfigSearchOrder` - Config discovery priority and bootstrap
- `TestConfigLoading` - INI parsing and fallback defaults
- `TestConfigInitialization` - Module state and auto-init
- `TestExpandPath` - Tilde expansion
- `TestBackwardCompatibility` - Old-style attribute access

### 2. test_import.py (16 tests)
Tests for file import and markdown chunking logic.

**Coverage:**
- Basic import functionality (file existence, empty files)
- Paragraph chunking with configurable minimum size
- Heading preservation in chunks
- Section hierarchy metadata (nested headings with » separator)
- Data cleaning (base64 images, multi-space OCR artifacts, excessive newlines)
- Custom metadata merging
- Sequential chunk numbering and total_chunks consistency

**Test Classes:**
- `TestImportFileBasics` - File handling
- `TestChunkingLogic` - Minimum chunk size behavior
- `TestHeadingPreservation` - Markdown heading handling
- `TestDataCleaning` - Text normalization
- `TestCustomMetadata` - Metadata attachment
- `TestChunkMetadata` - Chunk numbering

### 3. test_server.py (Enhanced - added 3 tests)
Added port collision tests to existing server tests.

**New Coverage:**
- Port already in use (OSError detection)
- _find_free_port() returns bindable port
- Sequential server instances use different ports

**New Test Class:**
- `TestPortCollision` - Port binding and collision handling

## Existing Tests (Verified Working)

### test_bm25.py (24 tests)
Already comprehensive coverage of:
- Tokenizer filtering (stop words, hex strings, base64, short tokens)
- BM25 scoring (IDF, term frequency, k1 parameter)

### test_server.py (Original 10 tests)
Already covered:
- Health, count, store, search endpoints
- Error handling (400, 404)
- Default collection behavior

## Test Suite Statistics

**Total Tests:** 150
- test_config.py: 17 tests ✓
- test_import.py: 16 tests ✓
- test_bm25.py: 24 tests ✓
- test_server.py: 13 tests (10 original + 3 new) ✓
- test_sqlite_backend.py: 33 tests ✓
- Other existing tests: 47 tests ✓

**All tests passing** (verified March 19, 2026)

## Running Tests

```bash
# All new tests
pytest tests/test_config.py tests/test_import.py tests/test_server.py::TestPortCollision -v

# Specific areas
pytest tests/test_config.py -v  # Config loading
pytest tests/test_import.py -v  # Import/chunking
pytest tests/test_bm25.py -v    # BM25/tokenizer
pytest tests/test_server.py -v  # HTTP server

# Full suite
pytest tests/ -v
```

## Notes

- All tests use MockEmbedder from conftest.py for fast execution
- Import tests use MagicMock for RaggerMemory to avoid database dependencies
- Server tests spawn real HTTP servers on random ports for integration testing
- Config tests use temporary directories and monkeypatch for HOME isolation
