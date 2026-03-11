"""
Storage backends for Ragger Memory

Backends are imported lazily by memory.py to avoid pulling in
optional dependencies (e.g. pymongo) when not needed.
"""
from .base import MemoryBackend

__all__ = ['MemoryBackend']
