"""
Embedder class for text embedding generation
"""
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, MODEL_CACHE_DIR

logger = logging.getLogger(__name__)


class Embedder:
    """Handles embedding model loading and text encoding"""
    
    def __init__(self):
        """Initialize embedder and load model from local snapshot (no network)"""
        self.model = None
        self._load_model()
    
    def _resolve_model_path(self) -> tuple[str | None, str | None]:
        """
        Resolve local model path (logic extracted from old config.py)
        
        Returns:
            (model_local_path, unavailable_reason)
            If model is available, returns (path, None)
            If unavailable, returns (None, reason)
        """
        model_cache_path = Path(MODEL_CACHE_DIR)
        model_hub_dir = (
            model_cache_path / 'hub' / 
            f'models--sentence-transformers--{EMBEDDING_MODEL}' / 'snapshots'
        )
        
        if model_cache_path.is_symlink() and not model_cache_path.exists():
            import os
            return None, (
                f"Model cache directory is a broken symlink:\n"
                f"  {MODEL_CACHE_DIR} -> {os.readlink(str(model_cache_path))}\n"
                f"  The target drive may be offline or unmounted."
            )
        
        if not model_cache_path.exists():
            return None, (
                f"Model cache directory not found: {MODEL_CACHE_DIR}\n"
                f"Run './ragger.py --update-model' to download the embedding model."
            )
        
        if model_hub_dir.is_dir():
            snapshots = list(model_hub_dir.iterdir())
            if snapshots:
                return str(snapshots[0]), None
            else:
                return None, (
                    f"Model '{EMBEDDING_MODEL}' has no snapshots in {model_hub_dir}\n"
                    f"Run './ragger.py --update-model' to download it."
                )
        else:
            return None, (
                f"Model '{EMBEDDING_MODEL}' not found in {MODEL_CACHE_DIR}\n"
                f"Run './ragger.py --update-model' to download it."
            )
    
    def _load_model(self):
        """Load sentence transformer model from local snapshot only (no network)"""
        model_path, error = self._resolve_model_path()
        
        if not model_path:
            raise FileNotFoundError(
                error or f"Model '{EMBEDDING_MODEL}' not available.\n"
                f"Run './ragger.py --update-model' to download it."
            )
        
        try:
            # Force CPU — MPS (Metal) can throw I/O errors in background
            # processes. CPU is fast enough for 384-dim embeddings.
            self.model = SentenceTransformer(model_path, device="cpu")
            logger.info(f"Embedding model loaded: {EMBEDDING_MODEL} (cpu)")
        except Exception as e:
            logger.error(f"Failed to load embedding model from {model_path}: {e}")
            raise
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to embedding vector
        
        Args:
            text: Text to encode
        
        Returns:
            NumPy array of shape (EMBEDDING_DIMENSIONS,)
        """
        return self.model.encode(text)
    
    @staticmethod
    def download_model():
        """Download or update the embedding model from HuggingFace"""
        logger.info(f"Downloading model: {EMBEDDING_MODEL}")
        logger.info(f"Cache directory: {MODEL_CACHE_DIR}")
        model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR)
        logger.info(f"Model ready: {EMBEDDING_MODEL}")
        return model
