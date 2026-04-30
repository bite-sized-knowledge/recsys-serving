"""Per-(member, category) Beta Thompson Sampling.

Lazy init: 첫 호출 시 member_interest 기반 prior 채워 INSERT.
실시간 reward update: feedback endpoint에서 incremental.
배치 reconcile (recommender) 가 매일 ground-truth 로 overwrite.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

# Prior 강도 — recommender/config.yml 과 동기화. 변경 시 양쪽 같이.
PRIOR_ALPHA_ONBOARDING = 4.0
PRIOR_BETA_ONBOARDING = 1.0
PRIOR_ALPHA_DEFAULT = 1.0
PRIOR_BETA_DEFAULT = 2.0

# Reward delta per event_type. uninterest 는 β 증가, 그 외는 α 증가.
REWARD_ALPHA: Dict[str, float] = {
    "article_in": 1.0,
    "like": 1.0,
    "archive": 1.0,
    "share": 2.0,
}
REWARD_BETA: Dict[str, float] = {
    "uninterest": 2.0,
}


@dataclass
class BanditRow:
    member_id: int
    category_id: int
    alpha: float
    beta: float


def _fetch_member_interests(db: Session, member_id: int) -> List[int]:
    rows = db.execute(
        text("SELECT interest_id FROM member_interest WHERE member_id = :m"),
        {"m": member_id},
    ).all()
    return [int(r[0]) for r in rows]


def _fetch_pool_categories(db: Session) -> List[int]:
    rows = db.execute(
        text("SELECT DISTINCT category_id FROM recommendation_global WHERE category_id IS NOT NULL")
    ).all()
    return [int(r[0]) for r in rows]


def _fetch_existing_state(db: Session, member_id: int) -> Dict[int, BanditRow]:
    rows = db.execute(
        text("SELECT category_id, alpha, beta FROM member_category_bandit WHERE member_id = :m"),
        {"m": member_id},
    ).all()
    return {
        int(r[0]): BanditRow(
            member_id=member_id, category_id=int(r[0]), alpha=float(r[1]), beta=float(r[2])
        )
        for r in rows
    }


def _lazy_init(db: Session, member_id: int, categories: Iterable[int]) -> Dict[int, BanditRow]:
    """member_category_bandit 에 prior 로 row 채워넣고 반환."""
    onboarding = set(_fetch_member_interests(db, member_id))
    rows: List[Dict] = []
    out: Dict[int, BanditRow] = {}
    for cid in categories:
        if cid in onboarding:
            a, b = PRIOR_ALPHA_ONBOARDING, PRIOR_BETA_ONBOARDING
        else:
            a, b = PRIOR_ALPHA_DEFAULT, PRIOR_BETA_DEFAULT
        rows.append({"m": member_id, "c": int(cid), "a": a, "b": b})
        out[int(cid)] = BanditRow(member_id, int(cid), a, b)

    if rows:
        db.execute(
            text(
                """
                INSERT INTO member_category_bandit (member_id, category_id, alpha, beta)
                VALUES (:m, :c, :a, :b)
                ON DUPLICATE KEY UPDATE alpha = alpha
                """
            ),
            rows,
        )
        db.commit()
    return out


def load_or_init(db: Session, member_id: int) -> Dict[int, BanditRow]:
    """현재 풀에 있는 카테고리 모두에 대해 (member_id, category_id) state 보장."""
    pool_categories = _fetch_pool_categories(db)
    if not pool_categories:
        return {}

    existing = _fetch_existing_state(db, member_id)
    missing = [c for c in pool_categories if c not in existing]
    if missing:
        new_rows = _lazy_init(db, member_id, missing)
        existing.update(new_rows)
    # 풀에 있는 카테고리만 반환 (drop된 카테고리는 무시)
    return {cid: existing[cid] for cid in pool_categories if cid in existing}


def sample_thetas(state: Dict[int, BanditRow], rng: Optional[np.random.Generator] = None) -> Dict[int, float]:
    """Beta(α, β) sample per category."""
    if rng is None:
        rng = np.random.default_rng()
    return {cid: float(rng.beta(row.alpha, row.beta)) for cid, row in state.items()}


def allocate_quota(thetas: Dict[int, float], n_results: int, top_k_categories: int = 6) -> Dict[int, int]:
    """
    softmax(theta) → quota proportions → integer rounding (sum = n_results).

    상위 K 카테고리만 추출해서 quota 분배 (long-tail 다이버시티는 글로벌 풀 빌드 단계에서 보장됨).
    rounding 잔차는 가장 큰 fractional remainder 카테고리에 할당.
    """
    if not thetas or n_results <= 0:
        return {}

    items = sorted(thetas.items(), key=lambda kv: kv[1], reverse=True)[:top_k_categories]
    cats = [c for c, _ in items]
    vals = np.asarray([v for _, v in items], dtype=np.float64)

    # softmax (temperature=1)
    e = np.exp(vals - vals.max())
    probs = e / e.sum()

    raw = probs * n_results
    base = np.floor(raw).astype(int)
    remaining = n_results - int(base.sum())
    if remaining > 0:
        # 가장 큰 fractional remainder 부터 1씩
        frac = raw - base
        order = np.argsort(-frac)
        for i in order[:remaining]:
            base[i] += 1

    # 0 quota 카테고리 drop
    return {cats[i]: int(base[i]) for i in range(len(cats)) if base[i] > 0}


def apply_reward(db: Session, member_id: int, category_id: int, event_type: str) -> None:
    """실시간 incremental update. 이벤트 타입별 α/β delta 적용."""
    et = event_type.lower()
    da = REWARD_ALPHA.get(et, 0.0)
    db_ = REWARD_BETA.get(et, 0.0)
    if da == 0.0 and db_ == 0.0:
        return

    db.execute(
        text(
            """
            INSERT INTO member_category_bandit (member_id, category_id, alpha, beta, clicks)
            VALUES (:m, :c, :pa, :pb, :clk)
            ON DUPLICATE KEY UPDATE
                alpha  = alpha + :da,
                beta   = beta  + :db_,
                clicks = clicks + :clk
            """
        ),
        {
            "m": int(member_id),
            "c": int(category_id),
            # 새로 만들 때 prior + delta (existing이면 ON DUPLICATE 분기)
            "pa": (PRIOR_ALPHA_DEFAULT + da),
            "pb": (PRIOR_BETA_DEFAULT + db_),
            "da": da,
            "db_": db_,
            "clk": 1 if et in REWARD_ALPHA else 0,
        },
    )
    db.commit()
