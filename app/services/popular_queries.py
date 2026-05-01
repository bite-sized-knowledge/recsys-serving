"""인기 검색어 추적 + prefix 자동완성.

Redis가 있으면 sorted set(ZINCRBY)으로 누적 카운트 + ZSCAN/ZRANGEBYLEX로 prefix 검색.
Redis 없으면 in-process Counter + 단순 prefix scan으로 fallback.

검색 실행 시 service에서 record(query)를 호출하여 카운트 증가.
suggest(prefix, limit)이 popular + prefix 매칭 결과를 반환.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter
from typing import List

from app.core.config import config
from app.services import redis_client

log = logging.getLogger(__name__)

_local_lock = threading.Lock()
_local_counter: Counter[str] = Counter()
_LOCAL_MAX = 5000


def _normalize(query: str) -> str:
    return " ".join(query.strip().lower().split())


def record(query: str) -> None:
    if not config.SEARCH_POPULAR_ENABLED:
        return
    norm = _normalize(query)
    if not norm or len(norm) > 200:
        return
    client = redis_client.get_client()
    if client is not None:
        try:
            pipe = client.pipeline(transaction=False)
            pipe.zincrby(config.REDIS_POPULAR_KEY, 1, norm)
            pipe.expire(config.REDIS_POPULAR_KEY, config.REDIS_POPULAR_TTL)
            pipe.execute()
            return
        except Exception as exc:
            log.warning("Redis zincrby failed, fallback to in-memory: %s", exc)

    with _local_lock:
        _local_counter[norm] += 1
        if len(_local_counter) > _LOCAL_MAX:
            # 가장 적은 항목 50개 제거 (간이 LRU 대용)
            for k, _ in _local_counter.most_common()[-50:]:
                del _local_counter[k]


def suggest(prefix: str, limit: int = 8) -> List[str]:
    """prefix로 시작하는 인기 검색어 top-N 반환 (count desc)."""
    if not config.SEARCH_POPULAR_ENABLED:
        return []
    norm = _normalize(prefix)
    if not norm:
        # 빈 prefix면 전역 인기 검색어
        return top(limit)

    client = redis_client.get_client()
    if client is not None:
        try:
            # ZSCAN으로 모든 멤버 순회 후 prefix 매칭. 데이터 규모가 5000 미만이면
            # 충분히 빠르고, 그 이상이면 Redis Search 모듈로 교체 검토.
            matches: list[tuple[str, float]] = []
            cursor = 0
            pattern = f"{norm}*"
            while True:
                cursor, batch = client.zscan(
                    config.REDIS_POPULAR_KEY, cursor=cursor, match=pattern, count=200
                )
                for member, score in batch:
                    matches.append((member.decode("utf-8"), float(score)))
                if cursor == 0:
                    break
            matches.sort(key=lambda x: x[1], reverse=True)
            return [m for m, _ in matches[:limit]]
        except Exception as exc:
            log.warning("Redis zscan failed, fallback to in-memory: %s", exc)

    with _local_lock:
        items = [(k, c) for k, c in _local_counter.items() if k.startswith(norm)]
    items.sort(key=lambda x: x[1], reverse=True)
    return [k for k, _ in items[:limit]]


def top(limit: int = 8) -> List[str]:
    if not config.SEARCH_POPULAR_ENABLED:
        return []
    client = redis_client.get_client()
    if client is not None:
        try:
            members = client.zrevrange(config.REDIS_POPULAR_KEY, 0, limit - 1)
            return [m.decode("utf-8") for m in members]
        except Exception as exc:
            log.warning("Redis zrevrange failed, fallback to in-memory: %s", exc)

    with _local_lock:
        return [k for k, _ in _local_counter.most_common(limit)]


def clear() -> None:
    """For tests."""
    client = redis_client.get_client()
    if client is not None:
        try:
            client.delete(config.REDIS_POPULAR_KEY)
        except Exception:
            pass
    with _local_lock:
        _local_counter.clear()
