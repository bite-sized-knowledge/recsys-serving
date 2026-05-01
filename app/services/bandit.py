"""Per-(member|device, category) Beta Thompson Sampling.

회원 (`member_category_bandit`) 과 비회원 device (`device_category_bandit`) 가
같은 알고리즘을 공유하되 PK 컬럼만 다름 — 공통 helper 로 일반화하고 prior 결정만 분기.

Lazy init: 첫 호출 시 onboarding (회원) 또는 X-Interest-Ids (비회원) 기반 prior 채워 INSERT.
실시간 reward: feedback endpoint 가 incremental UPDATE.
배치 reconcile (recommender) 가 매일 ground-truth 로 overwrite.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple, Union

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

# 비회원 prior — interest_ids 미제공 시 균등.
DEVICE_PRIOR_ALPHA = 1.0
DEVICE_PRIOR_BETA = 1.0
# X-Interest-Ids 헤더로 들어온 카테고리: 회원 onboarding 보다 약간 약한 prior
# (회원: Beta(4,1) — 명시 가입 + 동의 / 비회원: Beta(2,1) — UI 클릭 한번이라 신뢰 약함).
DEVICE_PRIOR_ALPHA_INTEREST = 2.0
DEVICE_PRIOR_BETA_INTEREST = 1.0
DEVICE_PRIOR_ALPHA_NON_INTEREST = 1.0
DEVICE_PRIOR_BETA_NON_INTEREST = 2.0

# 테이블 ↔ PK 컬럼 매핑. 호출자 (서비스/피드백) 가 직접 string 안 쓰도록.
_TABLE_MEMBER = "member_category_bandit"
_TABLE_DEVICE = "device_category_bandit"
_KEY_MEMBER = "member_id"
_KEY_DEVICE = "device_id"


@dataclass(frozen=True)
class BanditRow:
    """member_id (int) 또는 device_id (str) 를 동일 dataclass 로. sample_thetas 가 alpha/beta 만 본다."""
    key: Union[int, str]
    category_id: int
    alpha: float
    beta: float


# Prior decision: (category_id) → (alpha, beta).
PriorFn = Callable[[int], Tuple[float, float]]


# ---------------------------------------------------------------------------
# Generic SQL helpers — 테이블/키 컬럼만 다르고 logic 동일
# ---------------------------------------------------------------------------

def _fetch_existing_bandit(
    db: Session, table: str, key_col: str, key_val: Union[int, str]
) -> Dict[int, BanditRow]:
    """SELECT category_id, alpha, beta FROM <table> WHERE <key_col> = :k"""
    sql = f"SELECT category_id, alpha, beta FROM {table} WHERE {key_col} = :k"
    rows = db.execute(text(sql), {"k": key_val}).all()
    return {
        int(r[0]): BanditRow(key=key_val, category_id=int(r[0]), alpha=float(r[1]), beta=float(r[2]))
        for r in rows
    }


def _insert_bandit_priors(
    db: Session,
    table: str,
    key_col: str,
    key_val: Union[int, str],
    categories: Iterable[int],
    prior_fn: PriorFn,
) -> Dict[int, BanditRow]:
    """카테고리들에 대해 prior INSERT (ON DUPLICATE KEY UPDATE alpha=alpha 로 멱등). 반환은 만들어진 row dict."""
    payload: List[Dict] = []
    out: Dict[int, BanditRow] = {}
    for cid in categories:
        a, b = prior_fn(int(cid))
        payload.append({"k": key_val, "c": int(cid), "a": a, "b": b})
        out[int(cid)] = BanditRow(key=key_val, category_id=int(cid), alpha=a, beta=b)

    if payload:
        sql = (
            f"INSERT INTO {table} ({key_col}, category_id, alpha, beta) "
            f"VALUES (:k, :c, :a, :b) "
            f"ON DUPLICATE KEY UPDATE alpha = alpha"
        )
        db.execute(text(sql), payload)
        db.commit()
    return out


def _apply_reward_generic(
    db: Session,
    table: str,
    key_col: str,
    key_val: Union[int, str],
    category_id: int,
    event_type: str,
    prior_alpha: float,
    prior_beta: float,
) -> None:
    """incremental α/β delta + clicks 카운터. event_type 이 reward 매핑에 없으면 no-op."""
    et = event_type.lower()
    da = REWARD_ALPHA.get(et, 0.0)
    db_ = REWARD_BETA.get(et, 0.0)
    if da == 0.0 and db_ == 0.0:
        return

    sql = (
        f"INSERT INTO {table} ({key_col}, category_id, alpha, beta, clicks) "
        f"VALUES (:k, :c, :pa, :pb, :clk) "
        f"ON DUPLICATE KEY UPDATE "
        f"alpha = alpha + :da, beta = beta + :db_, clicks = clicks + :clk"
    )
    db.execute(
        text(sql),
        {
            "k": key_val,
            "c": int(category_id),
            "pa": prior_alpha + da,
            "pb": prior_beta + db_,
            "da": da,
            "db_": db_,
            "clk": 1 if et in REWARD_ALPHA else 0,
        },
    )
    db.commit()


# ---------------------------------------------------------------------------
# 글로벌 풀 카테고리 (TTL 캐시) + 회원 onboarding 보조 SELECT
# ---------------------------------------------------------------------------

_POOL_CAT_TTL_SEC = 300
_pool_cat_cache: Tuple[float, List[int]] = (0.0, [])


def _fetch_pool_categories(db: Session) -> List[int]:
    """글로벌 풀의 distinct category_id. 풀은 일 1회 swap 이라 process-level TTL 캐시."""
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


def _fetch_member_interests(db: Session, member_id: int) -> List[int]:
    rows = db.execute(
        text("SELECT interest_id FROM member_interest WHERE member_id = :m"),
        {"m": member_id},
    ).all()
    return [int(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# Prior 결정 함수
# ---------------------------------------------------------------------------

def _member_prior_fn(onboarding_set: set) -> PriorFn:
    def fn(cid: int) -> Tuple[float, float]:
        if cid in onboarding_set:
            return PRIOR_ALPHA_ONBOARDING, PRIOR_BETA_ONBOARDING
        return PRIOR_ALPHA_DEFAULT, PRIOR_BETA_DEFAULT
    return fn


def _device_prior_fn(interest_set: Optional[set]) -> PriorFn:
    def fn(cid: int) -> Tuple[float, float]:
        if interest_set is None:
            return DEVICE_PRIOR_ALPHA, DEVICE_PRIOR_BETA
        if cid in interest_set:
            return DEVICE_PRIOR_ALPHA_INTEREST, DEVICE_PRIOR_BETA_INTEREST
        return DEVICE_PRIOR_ALPHA_NON_INTEREST, DEVICE_PRIOR_BETA_NON_INTEREST
    return fn


# ---------------------------------------------------------------------------
# Public API — service.py / feedback.py 가 호출. 시그니처/동작은 phase 1.6 그대로.
# ---------------------------------------------------------------------------

def load_or_init(db: Session, member_id: int) -> Dict[int, BanditRow]:
    """회원: 풀 카테고리 set 에 대해 (member_id, category_id) state 보장."""
    pool_categories = _fetch_pool_categories(db)
    if not pool_categories:
        return {}

    existing = _fetch_existing_bandit(db, _TABLE_MEMBER, _KEY_MEMBER, int(member_id))
    missing = [c for c in pool_categories if c not in existing]
    if missing:
        prior_fn = _member_prior_fn(set(_fetch_member_interests(db, int(member_id))))
        new_rows = _insert_bandit_priors(db, _TABLE_MEMBER, _KEY_MEMBER, int(member_id), missing, prior_fn)
        existing.update(new_rows)
    return {cid: existing[cid] for cid in pool_categories if cid in existing}


def load_or_init_device(
    db: Session,
    device_id: str,
    interest_ids: Optional[Iterable[int]] = None,
) -> Dict[int, BanditRow]:
    """비회원 device: state 보장. interest_ids 는 lazy init prior 결정에만 사용 (이미 row 있으면 무시)."""
    pool_categories = _fetch_pool_categories(db)
    if not pool_categories:
        return {}

    existing = _fetch_existing_bandit(db, _TABLE_DEVICE, _KEY_DEVICE, str(device_id))
    missing = [c for c in pool_categories if c not in existing]
    if missing:
        interest_set = set(int(c) for c in interest_ids) if interest_ids else None
        prior_fn = _device_prior_fn(interest_set)
        new_rows = _insert_bandit_priors(db, _TABLE_DEVICE, _KEY_DEVICE, str(device_id), missing, prior_fn)
        existing.update(new_rows)
    return {cid: existing[cid] for cid in pool_categories if cid in existing}


def apply_reward(db: Session, member_id: int, category_id: int, event_type: str) -> None:
    _apply_reward_generic(
        db, _TABLE_MEMBER, _KEY_MEMBER, int(member_id), category_id, event_type,
        PRIOR_ALPHA_DEFAULT, PRIOR_BETA_DEFAULT,
    )


def apply_reward_device(db: Session, device_id: str, category_id: int, event_type: str) -> None:
    _apply_reward_generic(
        db, _TABLE_DEVICE, _KEY_DEVICE, str(device_id), category_id, event_type,
        DEVICE_PRIOR_ALPHA, DEVICE_PRIOR_BETA,
    )


def migrate_device_to_member(db: Session, device_id: str, member_id: int) -> int:
    """lazy guest 발급 직후 device_category_bandit → member_category_bandit 단일 INSERT...SELECT 이관.

    device 가 ground-truth (impression/click 누적). 회원 row 가 prior 만 있으면 device 값으로 덮어씀.
    이관 후 device 흔적은 보존 (분석용).
    rowcount: ON DUPLICATE KEY UPDATE 의 INSERT 1 / UPDATE 2 추가 규칙 때문에 정확한 source row 수 아님.
    """
    result = db.execute(
        text(
            f"""
            INSERT INTO {_TABLE_MEMBER} (member_id, category_id, alpha, beta, impressions, clicks)
            SELECT :m, category_id, alpha, beta, impressions, clicks
            FROM {_TABLE_DEVICE}
            WHERE device_id = :d
            ON DUPLICATE KEY UPDATE
                alpha = VALUES(alpha),
                beta = VALUES(beta),
                impressions = {_TABLE_MEMBER}.impressions + VALUES(impressions),
                clicks = {_TABLE_MEMBER}.clicks + VALUES(clicks)
            """
        ),
        {"m": int(member_id), "d": device_id},
    )
    db.commit()
    return int(result.rowcount or 0)


# ---------------------------------------------------------------------------
# Sampling / quota — state dataclass key 무시, alpha/beta 만 사용.
# ---------------------------------------------------------------------------

def sample_thetas(
    state: Mapping[int, BanditRow],
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, float]:
    if rng is None:
        rng = np.random.default_rng()
    return {cid: float(rng.beta(row.alpha, row.beta)) for cid, row in state.items()}


def allocate_quota(
    thetas: Dict[int, float],
    n_results: int,
    top_k_categories: int = 6,
    temperature: float = QUOTA_TEMPERATURE,
) -> Dict[int, int]:
    """softmax(theta / T) × N → 카테고리별 quota. T>1 이면 평탄화 (winner-take-all 완화)."""
    if not thetas or n_results <= 0:
        return {}

    items = sorted(thetas.items(), key=lambda kv: kv[1], reverse=True)[:top_k_categories]
    cats = [c for c, _ in items]
    vals = np.asarray([v for _, v in items], dtype=np.float64) / max(temperature, 1e-6)

    e = np.exp(vals - vals.max())
    probs = e / e.sum()

    raw = probs * n_results
    base = np.floor(raw).astype(int)
    remaining = n_results - int(base.sum())
    if remaining > 0:
        frac = raw - base
        order = np.argsort(-frac)
        for i in order[:remaining]:
            base[i] += 1

    return {cats[i]: int(base[i]) for i in range(len(cats)) if base[i] > 0}
