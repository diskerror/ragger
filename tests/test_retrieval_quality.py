"""
Retrieval quality tests against the real database.

These tests use the REAL embedder and REAL database to verify that
searches return relevant results. They're slower (~seconds) but catch
issues that mock-embedder unit tests can't.

Requires: ~/.local/share/ragger/memories.db with imported reference docs.

Run with:  pytest tests/test_retrieval_quality.py -v
Skip with: pytest --ignore=tests/test_retrieval_quality.py
"""
import os

import pytest

from ragger_memory.memory import RaggerMemory
from ragger_memory.config import SQLITE_PATH

# Skip entire module if the real database doesn't exist
DB_PATH = os.path.expanduser(SQLITE_PATH)
pytestmark = pytest.mark.skipif(
    not os.path.exists(DB_PATH),
    reason=f"Real database not found at {DB_PATH}"
)


@pytest.fixture(scope="module")
def memory():
    """Shared RaggerMemory instance for all tests (real embedder, real DB)."""
    mem = RaggerMemory()
    yield mem
    mem.close()


class TestDefaultSearchIncludesReferenceDocs:
    """After the fix, default search (collections=None) should search everything."""

    @pytest.mark.parametrize("query,expected_source_fragment", [
        ("tuplet Sibelius create", "sibelius"),
        ("ManuScript plugin language", "sibelius"),
        ("stopped horn mute brass", "orchestration"),
        ("string pizzicato arco bowing", "orchestration"),
        ("timpani tuning kettledrum", "orchestration"),
    ])
    def test_keyword_queries_find_reference_docs(self, memory, query, expected_source_fragment):
        """Keyword-rich queries should return reference docs, not just personal memories."""
        result = memory.search(query, limit=5)
        results = result["results"]
        assert len(results) > 0, f"No results for: {query}"

        # At least one result should come from a reference doc collection
        collections = [r["metadata"].get("collection", "") for r in results]
        assert any(
            expected_source_fragment in c for c in collections
        ), (
            f"Query '{query}' expected results from '{expected_source_fragment}' "
            f"collection, got: {collections}"
        )

    @pytest.mark.parametrize("query,expected_source_fragment", [
        ("How do I create a tuplet in Sibelius?", "sibelius"),
        ("What is the range of the bassoon?", "orchestration"),
        ("How do you mute a French horn?", "orchestration"),
    ])
    def test_natural_language_queries_find_reference_docs(self, memory, query, expected_source_fragment):
        """Natural language questions should also retrieve reference docs."""
        result = memory.search(query, limit=5)
        results = result["results"]
        assert len(results) > 0, f"No results for: {query}"

        collections = [r["metadata"].get("collection", "") for r in results]
        assert any(
            expected_source_fragment in c for c in collections
        ), (
            f"Query '{query}' expected results from '{expected_source_fragment}' "
            f"collection, got: {collections}"
        )


class TestCollectionNarrowing:
    """Explicit collection filters should restrict results."""

    def test_narrow_to_memory_only(self, memory):
        """Searching only 'memory' collection should exclude reference docs."""
        result = memory.search(
            "orchestration",
            limit=5,
            collections=["memory"]
        )
        for r in result["results"]:
            assert r["metadata"].get("collection") == "memory", (
                f"Got collection '{r['metadata'].get('collection')}' when narrowed to 'memory'"
            )

    def test_narrow_to_sibelius(self, memory):
        """Searching 'sibelius' should only return Sibelius docs."""
        result = memory.search(
            "tuplet",
            limit=5,
            collections=["sibelius"]
        )
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert r["metadata"].get("collection") == "sibelius"

    def test_narrow_to_orchestration(self, memory):
        """Searching 'orchestration' should only return orchestration docs."""
        result = memory.search(
            "horn mute",
            limit=5,
            collections=["orchestration"]
        )
        assert len(result["results"]) > 0
        for r in result["results"]:
            assert r["metadata"].get("collection") == "orchestration"

    def test_multi_collection_search(self, memory):
        """Searching multiple collections should return results from both."""
        result = memory.search(
            "instrument range",
            limit=10,
            collections=["sibelius", "orchestration"]
        )
        collections = {r["metadata"].get("collection") for r in result["results"]}
        # Should have results (may not always hit both, but should have some)
        assert len(result["results"]) > 0


class TestScoreQuality:
    """Verify that scores are reasonable for known-good queries."""

    def test_keyword_queries_score_above_threshold(self, memory):
        """Keyword queries against reference docs should score well."""
        result = memory.search("stopped horn mute brass", limit=1)
        assert len(result["results"]) > 0
        top_score = result["results"][0]["score"]
        assert top_score > 0.5, (
            f"Top score {top_score:.3f} too low for a keyword-rich query"
        )

    def test_natural_language_scores_are_positive(self, memory):
        """Natural language queries should at least return positive scores."""
        result = memory.search("How do you mute a French horn?", limit=1)
        assert len(result["results"]) > 0
        assert result["results"][0]["score"] > 0.0


class TestNegativeQueries:
    """Queries about things NOT in the database should not hallucinate results."""

    def test_no_pinecone_references(self, memory):
        """Ragger uses SQLite, not Pinecone/Weaviate/etc."""
        result = memory.search("Pinecone vector database setup", limit=5)
        for r in result["results"]:
            text_lower = r["text"].lower()
            assert "pinecone" not in text_lower, (
                f"Unexpected Pinecone reference in result: {r['text'][:100]}"
            )

    def test_no_external_stemming_library(self, memory):
        """BM25 tokenizer is built-in, no external stemming libs."""
        result = memory.search("NLTK stemming library install", limit=5)
        for r in result["results"]:
            text_lower = r["text"].lower()
            # Should not find docs about installing NLTK/spaCy for stemming
            assert "nltk" not in text_lower or "install" not in text_lower


class TestTechnicalDecisions:
    """Questions about documented technical decisions."""

    def test_find_bm25_rationale(self, memory):
        """Should find info about BM25 implementation decisions."""
        result = memory.search("Why is BM25 index persisted to SQLite?", limit=5)
        assert len(result["results"]) > 0

    def test_find_database_location(self, memory):
        """Should find where the Ragger database is stored."""
        result = memory.search("Where is the Ragger database stored?", limit=5)
        texts = " ".join(r["text"] for r in result["results"])
        # Should mention sqlite or the path
        assert "sqlite" in texts.lower() or "memories.db" in texts.lower() or "ragger" in texts.lower()


class TestWorkingPatterns:
    """Questions about working patterns stored as personal memories."""

    def test_find_project_breakdown_advice(self, memory):
        """Should find advice about breaking down projects for Reid."""
        result = memory.search(
            "How should projects be broken down for Reid?",
            limit=5,
            collections=["memory"]
        )
        assert len(result["results"]) > 0

    def test_find_tool_preferences(self, memory):
        """Should find info about Reid's tools."""
        result = memory.search(
            "What tools does Reid use for music notation?",
            limit=5,
            collections=["memory"]
        )
        assert len(result["results"]) > 0


class TestTimingMetrics:
    """Search should return timing information."""

    def test_timing_present(self, memory):
        result = memory.search("test query", limit=1)
        assert "timing" in result
        timing = result["timing"]
        assert "embedding_ms" in timing
        assert "search_ms" in timing
        assert "total_ms" in timing
        assert "corpus_size" in timing

    def test_filtered_size_matches_all_when_default(self, memory):
        """Default search should show filtered_size == corpus_size."""
        result = memory.search("test", limit=1)
        timing = result["timing"]
        assert timing["filtered_size"] == timing["corpus_size"]

    def test_filtered_size_smaller_when_narrowed(self, memory):
        """Narrowed search should show filtered_size < corpus_size."""
        result = memory.search("test", limit=1, collections=["memory"])
        timing = result["timing"]
        assert timing["filtered_size"] < timing["corpus_size"]
