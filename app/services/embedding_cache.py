"""Per-process TTL cache for query embeddings.

검색 hot query에 한해 인코딩 비용을 ~1ms로 단축한다. 다중 인스턴스 환경에서는
Redis로 교체 예정 (후속 PR).
"""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

import numpy as np
from cachetools import TTLCache

from app.core.config import config

_lock = threading.Lock()
_cache: TTLCache = TTLCache(maxsize=config.SEARCH_CACHE_SIZE, ttl=config.SEARCH_CACHE_TTL)


def _key(query: str) -> str:
    normalized = query.strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def get(query: str) -> Optional[np.ndarray]:
    with _lock:
        return _cache.get(_key(query))


def put(query: str, vector: np.ndarray) -> None:
    with _lock:
        _cache[_key(query)] = vector


def clear() -> None:
    with _lock:
        _cache.clear()
