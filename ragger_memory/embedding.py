"""
Embedder class for text embedding generation
"""
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, MODEL_CACHE_DIR
from . import lang

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
            return None, lang.ERR_MODEL_CACHE_BROKEN_SYMLINK.format(path=
                f"  {MODEL_CACHE_DIR} -> {os.readlink(str(model_cache_path))}\n"
                f"  The target drive may be offline or unmounted."
            )
        
        if not model_cache_path.exists():
            return None, lang.ERR_MODEL_CACHE_NOT_FOUND.format(path=MODEL_CACHE_DIR)
        
        if model_hub_dir.is_dir():
            snapshots = list(model_hub_dir.iterdir())
            if snapshots:
                return str(snapshots[0]), None
            else:
                return None, lang.ERR_MODEL_NO_SNAPSHOTS.format(
                    model=EMBEDDING_MODEL, path=model_hub_dir)
        else:
            return None, lang.ERR_MODEL_NOT_FOUND.format(
                model=EMBEDDING_MODEL, path=MODEL_CACHE_DIR)
    
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
            # Suppress noisy model load output (progress bars, load reports)
            import os, sys, io
            _prev_tqdm = os.environ.get("TQDM_DISABLE")
            os.environ["TQDM_DISABLE"] = "1"
            for _name in ("sentence_transformers", "mlx", "transformers",
                          "mlx_lm", "huggingface_hub"):
                logging.getLogger(_name).setLevel(logging.WARNING)
            # MLX prints load report via C-level stderr — redirect fd
            _quiet = logging.getLogger().getEffectiveLevel() > logging.DEBUG
            if _quiet:
                _devnull = os.open(os.devnull, os.O_WRONLY)
                _saved_stdout = os.dup(1)
                _saved_stderr = os.dup(2)
                os.dup2(_devnull, 1)
                os.dup2(_devnull, 2)
                os.close(_devnull)
            try:
                self.model = SentenceTransformer(model_path, device="cpu")
            finally:
                if _quiet:
                    os.dup2(_saved_stdout, 1)
                    os.dup2(_saved_stderr, 2)
                    os.close(_saved_stdout)
                    os.close(_saved_stderr)
            if _prev_tqdm is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = _prev_tqdm
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
        return self.model.encode(text, show_progress_bar=False)
    
    @staticmethod
    def download_model():
        """Download or update the embedding model from HuggingFace"""
        logger.info(f"Downloading model: {EMBEDDING_MODEL}")
        logger.info(f"Cache directory: {MODEL_CACHE_DIR}")
        model = SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR)
        logger.info(f"Model ready: {EMBEDDING_MODEL}")
        return model
