"""
BM25 ranking for Ragger Memory

Pure Python implementation — no external dependencies.
Built from the same cached texts as vector search; no extra DB reads.
"""
import math
import re
import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Common English stop words (small set — keeps it fast without killing recall)
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "its", "our",
    "their", "not", "no", "so", "if", "then", "than", "too", "very",
    "just", "about", "up", "out", "all", "also", "into", "over", "after",
})

# Regex: split on non-alphanumeric, keep numbers
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, split on non-alphanum, remove stop words."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


class BM25Index:
    """
    BM25 (Okapi BM25) index over a list of documents.
    
    Parameters:
        k1: Term frequency saturation. Higher = more weight to repeated terms.
            Typical range: 1.2–2.0. Default 1.5.
        b:  Length normalization. 0 = no normalization, 1 = full normalization.
            Typical range: 0.5–0.8. Default 0.75.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Built by build()
        self._doc_count: int = 0
        self._avgdl: float = 0.0
        self._doc_lens: Optional[np.ndarray] = None
        self._doc_freqs: Optional[List[dict]] = None  # per-doc term frequencies
        self._idf: Optional[dict] = None  # term -> IDF score
    
    def build(self, texts: List[str]):
        """
        Build the BM25 index from a list of document texts.
        
        Call this once after loading/caching texts, and again
        whenever the corpus changes (cache invalidation).
        """
        self._doc_count = len(texts)
        if self._doc_count == 0:
            self._avgdl = 0.0
            self._doc_lens = np.array([], dtype=np.float32)
            self._doc_freqs = []
            self._idf = {}
            return
        
        # Tokenize all documents
        self._doc_freqs = []
        doc_lens = []
        df = {}  # term -> number of documents containing it
        
        for text in texts:
            tokens = tokenize(text)
            doc_lens.append(len(tokens))
            
            # Term frequency for this doc
            tf = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._doc_freqs.append(tf)
            
            # Document frequency (count each term once per doc)
            for token in tf:
                df[token] = df.get(token, 0) + 1
        
        self._doc_lens = np.array(doc_lens, dtype=np.float32)
        self._avgdl = float(self._doc_lens.mean()) if self._doc_count > 0 else 0.0
        
        # Precompute IDF for all terms: log((N - df + 0.5) / (df + 0.5) + 1)
        self._idf = {}
        for term, freq in df.items():
            self._idf[term] = math.log(
                (self._doc_count - freq + 0.5) / (freq + 0.5) + 1.0
            )
        
        logger.info(
            f"BM25 index built: {self._doc_count} docs, "
            f"{len(self._idf)} unique terms, avgdl={self._avgdl:.1f}"
        )
    
    def score(self, query: str, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Score documents against a query.
        
        Args:
            query: Search query text
            indices: Optional array of document indices to score
                     (for collection filtering). If None, scores all docs.
        
        Returns:
            np.ndarray of BM25 scores, one per document in indices
            (or one per doc if indices is None).
        """
        if self._doc_count == 0:
            return np.array([], dtype=np.float32)
        
        query_tokens = tokenize(query)
        if not query_tokens:
            if indices is not None:
                return np.zeros(len(indices), dtype=np.float32)
            return np.zeros(self._doc_count, dtype=np.float32)
        
        if indices is None:
            indices = np.arange(self._doc_count)
        
        scores = np.zeros(len(indices), dtype=np.float32)
        
        for token in query_tokens:
            idf = self._idf.get(token, 0.0)
            if idf == 0.0:
                continue  # term not in corpus
            
            for i, doc_idx in enumerate(indices):
                tf = self._doc_freqs[doc_idx].get(token, 0)
                if tf == 0:
                    continue
                
                dl = self._doc_lens[doc_idx]
                # BM25 formula
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * numerator / denominator
        
        return scores
    
    @property
    def is_built(self) -> bool:
        return self._doc_freqs is not None and self._doc_count > 0
