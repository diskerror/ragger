"""
Configuration for Ragger Memory
"""
from pathlib import Path
import os

# MongoDB connection (local, no auth needed for dev)
MONGODB_URI = "mongodb://localhost:27017/"
DB_NAME = "ragger"
COLLECTION_NAME = "memories"
QUERY_LOG_COLLECTION = "query_log"
QUERY_LOGGING_ENABLED = True  # Set False to disable query logging

# Embedding model (local, no API key needed)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, ~90MB
EMBEDDING_DIMENSIONS = 384

# Model cache directory (where HuggingFace stores downloaded models)
MODEL_CACHE_DIR = os.environ.get(
    'SENTENCE_TRANSFORMERS_HOME',
    str(Path.home() / '.cache' / 'huggingface')
)

# Resolved local path to the model snapshot
# Loading from the snapshot path bypasses all HuggingFace Hub network calls
_model_hub_dir = Path(MODEL_CACHE_DIR) / 'hub' / f'models--sentence-transformers--{EMBEDDING_MODEL}' / 'snapshots'
MODEL_LOCAL_PATH = None
if _model_hub_dir.is_dir():
    snapshots = list(_model_hub_dir.iterdir())
    if snapshots:
        MODEL_LOCAL_PATH = str(snapshots[0])

# Search defaults
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_MIN_SCORE = 0.0
DEFAULT_CHUNK_SIZE = 500  # characters per chunk — roughly one paragraph
