"""POST /feeds/feedback — 실시간 reward 경로.

bite-api 가 user_events insert 후 fire-and-forget 호출.
실패해도 batch reconcile (recommender) 가 다음 cycle 에 정정한다.
"""
from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services import bandit
from app.services.user_vector_lookup import push_event_to_profile

from .schema import FeedbackAck, FeedbackEvent

log = logging.getLogger(__name__)


def _resolve_category(db: Session, article_id: str) -> int | None:
    row = db.execute(
        text("SELECT category_id FROM article WHERE article_id = :a"),
        {"a": article_id},
    ).first()
    if row is None or row[0] is None:
        return None
    return int(row[0])


def handle_feedback(db: Session, ev: FeedbackEvent) -> FeedbackAck:
    et = ev.event_type.lower()

    category_id = _resolve_category(db, ev.article_id)
    if category_id is None:
        return FeedbackAck(accepted=False, reason="category_id not found for article")

    # 1. bandit incremental update
    bandit_updated = False
    try:
        bandit.apply_reward(db, ev.member_id, category_id, et)
        bandit_updated = (et in bandit.REWARD_ALPHA) or (et in bandit.REWARD_BETA)
    except Exception as exc:
        log.warning("bandit apply_reward failed: %s", exc)

    # 2. Phase 2 user_profile EMA push (positive reward 이벤트만 — bandit α 신호와 동일 set)
    user_vector_updated = False
    if et in bandit.REWARD_ALPHA:
        try:
            user_vector_updated = push_event_to_profile(ev.member_id, ev.article_id)
        except Exception as exc:
            log.warning("user_vector push failed: %s", exc)

    return FeedbackAck(
        accepted=True,
        bandit_updated=bandit_updated,
        user_vector_updated=user_vector_updated,
    )
