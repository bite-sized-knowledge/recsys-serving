"""recommendation_impression 적재.

응답 직전 동기 INSERT (latency 안 큼: 10 row, single tx). 실패는 graceful skip.
member_id 또는 device_id 둘 중 하나는 NOT NULL (app 단 가드).
feed_request_id 는 같은 응답의 모든 row 가 공유 — impression ↔ click 정확 그룹핑.
was_backfilled / latency_ms 는 같은 feed_request_id 의 모든 row 에 동일 (응답 단위 메트릭).
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def log_impressions(
    db: Session,
    member_id: Optional[int],
    device_id: Optional[str],
    feed_request_id: Optional[str],
    rows: Iterable[Tuple[str, Optional[int], int, Optional[float], bool]],
    latency_ms: Optional[int] = None,
) -> int:
    """rows: (article_id, category_id, position, bandit_theta, was_backfilled). position 은 1..N (1-base)."""
    if member_id is None and not device_id:
        log.warning("impression log: both member_id and device_id missing — skip")
        return 0

    mid = int(member_id) if member_id is not None else None
    did = str(device_id) if device_id else None
    lat = int(latency_ms) if latency_ms is not None else None
    payload: List[dict] = [
        {
            "m": mid,
            "d": did,
            "a": str(article_id),
            "c": int(category_id) if category_id is not None else None,
            "p": int(position),
            "t": float(theta) if theta is not None else None,
            "fr": feed_request_id,
            "bf": 1 if was_backfilled else 0,
            "lat": lat,
        }
        for article_id, category_id, position, theta, was_backfilled in rows
    ]
    if not payload:
        return 0

    try:
        db.execute(
            text(
                """
                INSERT INTO recommendation_impression
                    (member_id, device_id, article_id, category_id, position,
                     feed_request_id, bandit_theta, was_backfilled, latency_ms)
                VALUES (:m, :d, :a, :c, :p, :fr, :t, :bf, :lat)
                """
            ),
            payload,
        )
        db.commit()
        return len(payload)
    except Exception as exc:
        log.warning("impression log failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            log.debug("impression log rollback skipped", exc_info=True)
        return 0
