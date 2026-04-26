"""Query embedding 캐시.

Redis가 활성·접속 가능하면 분산 캐시(SETEX/GET 바이트 직렬화), 아니면 in-process
TTLCache로 fallback. 다중 인스턴스 환경에서는 Redis가 권장된다.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

import numpy as np
from cachetools import TTLCache

from app.core.config import config
from app.services import redis_client

_lock = threading.Lock()
_local: TTLCache = TTLCache(maxsize=config.SEARCH_CACHE_SIZE, ttl=config.SEARCH_CACHE_TTL)


def _key(query: str) -> str:
    normalized = query.strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _redis_key(key: str) -> str:
    return f"{config.REDIS_CACHE_NAMESPACE}{key}"


def get(query: str) -> Optional[np.ndarray]:
    key = _key(query)
    client = redis_client.get_client()
    if client is not None:
        try:
            raw = client.get(_redis_key(key))
            if raw is not None:
                return np.frombuffer(raw, dtype=np.float32)
        except Exception:
            pass  # Redis hiccup → fall through to local
    with _lock:
        return _local.get(key)


def put(query: str, vector: np.ndarray) -> None:
    key = _key(query)
    vec = vector.astype(np.float32, copy=False)
    client = redis_client.get_client()
    if client is not None:
        try:
            client.setex(_redis_key(key), config.SEARCH_CACHE_TTL, vec.tobytes())
        except Exception:
            pass
    with _lock:
        _local[key] = vec


def clear() -> None:
    client = redis_client.get_client()
    if client is not None:
        try:
            for k in client.scan_iter(match=f"{config.REDIS_CACHE_NAMESPACE}*", count=500):
                client.delete(k)
        except Exception:
            pass
    with _lock:
        _local.clear()
