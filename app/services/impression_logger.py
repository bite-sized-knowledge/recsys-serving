"""recommendation_impression 적재.

응답 직전 동기 INSERT (latency 안 큼: 10 row, single tx). 실패는 graceful skip.
"""
from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


def log_impressions(
    db: Session,
    member_id: int,
    rows: Iterable[Tuple[str, Optional[int], int, Optional[float]]],
) -> int:
    """
    rows: iterable of (article_id, category_id, position, bandit_theta).
    position 은 1..N (1-base).
    Returns inserted row count.
    """
    payload: List[dict] = [
        {
            "m": int(member_id),
            "a": str(article_id),
            "c": int(category_id) if category_id is not None else None,
            "p": int(position),
            "t": float(theta) if theta is not None else None,
        }
        for article_id, category_id, position, theta in rows
    ]
    if not payload:
        return 0

    try:
        db.execute(
            text(
                """
                INSERT INTO recommendation_impression
                    (member_id, article_id, category_id, position, bandit_theta)
                VALUES (:m, :a, :c, :p, :t)
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
