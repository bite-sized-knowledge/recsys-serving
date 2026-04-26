"""Redis 연결 관리. Redis가 비활성/접속 불가면 None을 반환하여 호출 측이
in-process fallback을 사용하도록 한다.
"""

from __future__ import annotations

import logging
from typing import Optional

import redis as redis_pkg

from app.core.config import config

log = logging.getLogger(__name__)

_client: Optional[redis_pkg.Redis] = None
_attempted: bool = False


def get_client() -> Optional[redis_pkg.Redis]:
    global _client, _attempted
    if not config.REDIS_ENABLED:
        return None
    if _client is None and not _attempted:
        _attempted = True
        try:
            client = redis_pkg.from_url(
                config.REDIS_URL,
                socket_timeout=1.0,
                socket_connect_timeout=1.0,
                decode_responses=False,
            )
            client.ping()
            _client = client
            log.info("Redis connected: %s", config.REDIS_URL)
        except Exception as exc:
            log.warning("Redis unavailable, falling back to in-process state: %s", exc)
            _client = None
    return _client


def reset() -> None:
    """For tests."""
    global _client, _attempted
    _client = None
    _attempted = False
