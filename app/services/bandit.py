"""Per-(member, category) Beta Thompson Sampling.

Lazy init: 첫 호출 시 member_interest 기반 prior 채워 INSERT.
실시간 reward update: feedback endpoint에서 incremental.
배치 reconcile (recommender) 가 매일 ground-truth 로 overwrite.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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
# uninterest β 는 0.5 로 완화 (이전 2.0 — 한 번 누르면 카테고리 거의 죽어서 회복 어려웠음).
REWARD_ALPHA: Dict[str, float] = {
    "article_in": 1.0,
    "like": 1.0,
    "archive": 1.0,
    "share": 2.0,
}
REWARD_BETA: Dict[str, float] = {
    "uninterest": 0.5,
}

# softmax temperature — 높을수록 quota 가 평탄해져 winner-take-all 완화 (serendipity ↑).
QUOTA_TEMPERATURE = 1.5

# 비회원 prior — onboarding 정보 없으니 균등.
DEVICE_PRIOR_ALPHA = 1.0
DEVICE_PRIOR_BETA = 1.0


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


# 글로벌 풀은 매일 1회 swap. 카테고리 set 만 알면 되므로 process-level TTL 캐시.
_POOL_CAT_TTL_SEC = 300
_pool_cat_cache: Tuple[float, List[int]] = (0.0, [])


def _fetch_pool_categories(db: Session) -> List[int]:
    global _pool_cat_cache
    now = time.time()
    cached_at, cached = _pool_cat_cache
    if cached and (now - cached_at) < _POOL_CAT_TTL_SEC:
        return cached
    rows = db.execute(
        text("SELECT DISTINCT category_id FROM recommendation_global WHERE category_id IS NOT NULL")
    ).all()
    fresh = [int(r[0]) for r in rows]
    _pool_cat_cache = (now, fresh)
    return fresh


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


def allocate_quota(
    thetas: Dict[int, float],
    n_results: int,
    top_k_categories: int = 6,
    temperature: float = QUOTA_TEMPERATURE,
) -> Dict[int, int]:
    """
    softmax(theta / T) → quota proportions → integer rounding (sum = n_results).

    상위 K 카테고리만 추출해서 quota 분배 (long-tail 다이버시티는 글로벌 풀 빌드 단계에서 보장됨).
    rounding 잔차는 가장 큰 fractional remainder 카테고리에 할당.
    Temperature > 1 면 quota 가 더 평탄해져 winner-take-all 완화.
    """
    if not thetas or n_results <= 0:
        return {}

    items = sorted(thetas.items(), key=lambda kv: kv[1], reverse=True)[:top_k_categories]
    cats = [c for c, _ in items]
    vals = np.asarray([v for _, v in items], dtype=np.float64) / max(temperature, 1e-6)

    # softmax with temperature
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


@dataclass
class DeviceBanditRow:
    device_id: str
    category_id: int
    alpha: float
    beta: float


def _fetch_existing_device_state(db: Session, device_id: str) -> Dict[int, DeviceBanditRow]:
    rows = db.execute(
        text("SELECT category_id, alpha, beta FROM device_category_bandit WHERE device_id = :d"),
        {"d": device_id},
    ).all()
    return {
        int(r[0]): DeviceBanditRow(
            device_id=device_id, category_id=int(r[0]), alpha=float(r[1]), beta=float(r[2])
        )
        for r in rows
    }


def _lazy_init_device(db: Session, device_id: str, categories: Iterable[int]) -> Dict[int, DeviceBanditRow]:
    """device_category_bandit 에 prior(균등 Beta(1,1))로 row 채워넣고 반환."""
    rows: List[Dict] = []
    out: Dict[int, DeviceBanditRow] = {}
    for cid in categories:
        rows.append({"d": device_id, "c": int(cid), "a": DEVICE_PRIOR_ALPHA, "b": DEVICE_PRIOR_BETA})
        out[int(cid)] = DeviceBanditRow(device_id, int(cid), DEVICE_PRIOR_ALPHA, DEVICE_PRIOR_BETA)

    if rows:
        db.execute(
            text(
                """
                INSERT INTO device_category_bandit (device_id, category_id, alpha, beta)
                VALUES (:d, :c, :a, :b)
                ON DUPLICATE KEY UPDATE alpha = alpha
                """
            ),
            rows,
        )
        db.commit()
    return out


def load_or_init_device(db: Session, device_id: str) -> Dict[int, BanditRow]:
    """비회원 device 의 (device_id, category_id) state 보장. 인터페이스 호환을 위해 BanditRow 형태로 반환."""
    pool_categories = _fetch_pool_categories(db)
    if not pool_categories:
        return {}

    existing = _fetch_existing_device_state(db, device_id)
    missing = [c for c in pool_categories if c not in existing]
    if missing:
        new_rows = _lazy_init_device(db, device_id, missing)
        existing.update(new_rows)

    # member_id 자리에 0 (sentinel) 두고 BanditRow 로 변환 — sample_thetas / allocate_quota 재사용.
    return {
        cid: BanditRow(member_id=0, category_id=cid, alpha=existing[cid].alpha, beta=existing[cid].beta)
        for cid in pool_categories
        if cid in existing
    }


def apply_reward_device(db: Session, device_id: str, category_id: int, event_type: str) -> None:
    """비회원 device 의 incremental reward."""
    et = event_type.lower()
    da = REWARD_ALPHA.get(et, 0.0)
    db_ = REWARD_BETA.get(et, 0.0)
    if da == 0.0 and db_ == 0.0:
        return

    db.execute(
        text(
            """
            INSERT INTO device_category_bandit (device_id, category_id, alpha, beta, clicks)
            VALUES (:d, :c, :pa, :pb, :clk)
            ON DUPLICATE KEY UPDATE
                alpha  = alpha + :da,
                beta   = beta  + :db_,
                clicks = clicks + :clk
            """
        ),
        {
            "d": device_id,
            "c": int(category_id),
            "pa": (DEVICE_PRIOR_ALPHA + da),
            "pb": (DEVICE_PRIOR_BETA + db_),
            "da": da,
            "db_": db_,
            "clk": 1 if et in REWARD_ALPHA else 0,
        },
    )
    db.commit()


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
