"""
Tests for BM25 ranking module.
"""
import numpy as np
import pytest

from ragger_memory.bm25 import BM25Index, tokenize, STOP_WORDS


class TestTokenize:
    """Tests for the tokenizer function."""
    
    def test_basic_tokenization(self):
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]
    
    def test_removes_stop_words(self):
        tokens = tokenize("the quick brown fox is a animal")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
    
    def test_splits_on_punctuation(self):
        tokens = tokenize("hello, world! foo-bar")
        assert tokens == ["hello", "world", "foo", "bar"]
    
    def test_keeps_numbers(self):
        tokens = tokenize("Python 3.10 requires version 2")
        assert "python" in tokens
        assert "3" in tokens
        assert "10" in tokens
    
    def test_lowercase(self):
        tokens = tokenize("UPPER Case MiXeD")
        assert tokens == ["upper", "case", "mixed"]
    
    def test_empty_string(self):
        assert tokenize("") == []
    
    def test_only_stop_words(self):
        assert tokenize("the a an is are") == []
    
    def test_unicode_stripped(self):
        # Non-alphanumeric chars are stripped
        tokens = tokenize("café résumé naïve")
        assert "caf" in tokens  # é stripped
        assert "r" in tokens
        assert "sum" in tokens


class TestBM25Index:
    """Tests for the BM25 index."""
    
    def test_build_empty(self):
        idx = BM25Index()
        idx.build([])
        assert not idx.is_built
    
    def test_build_single_doc(self):
        idx = BM25Index()
        idx.build(["hello world"])
        assert idx.is_built
        assert idx._doc_count == 1
    
    def test_build_multiple_docs(self):
        docs = [
            "Python programming language",
            "Java programming language",
            "Rust systems programming",
        ]
        idx = BM25Index()
        idx.build(docs)
        assert idx._doc_count == 3
        assert len(idx._idf) > 0
    
    def test_score_empty_index(self):
        idx = BM25Index()
        idx.build([])
        scores = idx.score("test query")
        assert len(scores) == 0
    
    def test_score_empty_query(self):
        idx = BM25Index()
        idx.build(["hello world"])
        scores = idx.score("")
        assert len(scores) == 1
        assert scores[0] == 0.0
    
    def test_score_matching_doc(self):
        docs = [
            "Python programming language",
            "cooking recipes for dinner",
            "Python snake facts and habitat",
        ]
        idx = BM25Index()
        idx.build(docs)
        scores = idx.score("Python programming")
        # Doc 0 should score highest (both terms match)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]
    
    def test_score_with_indices(self):
        docs = [
            "alpha bravo charlie",
            "delta echo foxtrot",
            "alpha delta golf",
        ]
        idx = BM25Index()
        idx.build(docs)
        # Only score docs 0 and 2
        indices = np.array([0, 2])
        scores = idx.score("alpha", indices)
        assert len(scores) == 2
        assert scores[0] > 0  # "alpha" in doc 0
        assert scores[1] > 0  # "alpha" in doc 2
    
    def test_idf_rare_term_scores_higher(self):
        """A term appearing in fewer docs should have higher IDF."""
        docs = [
            "common word common word",
            "common word rare",
            "common word everyday",
        ]
        idx = BM25Index()
        idx.build(docs)
        assert idx._idf["rare"] > idx._idf["common"]
    
    def test_term_frequency_matters(self):
        """A doc with more occurrences of the query term should score higher."""
        docs = [
            "python",
            "python python python",
        ]
        idx = BM25Index()
        idx.build(docs)
        scores = idx.score("python")
        assert scores[1] > scores[0]
    
    def test_k1_parameter(self):
        """Higher k1 should increase the effect of term frequency."""
        docs = ["word", "word word word word word"]
        
        low_k1 = BM25Index(k1=0.5)
        low_k1.build(docs)
        scores_low = low_k1.score("word")
        
        high_k1 = BM25Index(k1=3.0)
        high_k1.build(docs)
        scores_high = high_k1.score("word")
        
        # With higher k1, the gap between 1 occurrence and 5 should be larger
        gap_low = scores_low[1] - scores_low[0]
        gap_high = scores_high[1] - scores_high[0]
        assert gap_high > gap_low
    
    def test_scores_are_non_negative(self):
        docs = ["foo bar baz", "qux quux corge", "grault garply waldo"]
        idx = BM25Index()
        idx.build(docs)
        scores = idx.score("foo qux nonexistent")
        assert all(s >= 0 for s in scores)
