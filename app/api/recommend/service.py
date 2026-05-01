"""
Phase 1+2 추천 서빙 (회원 + 비회원).

흐름 (회원):
  1. bandit state load (or lazy init from member_interest) — member_category_bandit
  2. Beta TS sample → 카테고리별 theta
  3. softmax(theta / T) × N → 카테고리 quota 분배 (winner-take-all 완화)
  4. recommendation_global 에서 카테고리별 후보 가져옴 (lang 필터 + ROW_NUMBER 단일 쿼리)
  5. Phase 2: user_profile 있으면 within-category cosine rerank
  6. quota 부족시 글로벌 풀 weighted random backfill (deterministic 회피)
  7. feed_request_id 발급, dedup, position 부여, recommendation_impression 적재

흐름 (비회원, device_id-only):
  1. device_category_bandit lazy init (균등 prior Beta(1,1))
  2~6. 위와 동일 (단 user_profile 단계 skip — 회원 전용)
  7. impression 에 device_id 로 적재
"""
from __future__ import annotations

import logging
import secrets
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.core.config import config
from app.services import bandit
from app.services.impression_logger import log_impressions
from app.services.qdrant_client import get_client
from app.services.user_vector_lookup import (
    _to_article_point_id,
    cosine,
    get_user_vector,
)

from .schema import RecommendItem

log = logging.getLogger(__name__)

N_RESULTS = 10
TOP_K_CATEGORIES = 6
CANDIDATE_MULTIPLIER = 4  # quota * 4 만큼 카테고리 후보 가져와 rerank
BACKFILL_POOL_SIZE = 50    # 글로벌 풀 backfill 시 weighted random 모집단

# 메트릭 카운터 — admin/diagnostics 에서 노출. 휘발성 (process restart 시 reset).
_request_metrics: Dict[str, int] = {"requests": 0, "anonymous": 0, "backfilled": 0}
_latency_ms: List[float] = []  # 최근 1024 요청
_LATENCY_BUF = 1024


def get_request_metrics() -> Dict[str, float]:
    """미들웨어/diagnostics 가 읽을 가벼운 인메모리 메트릭."""
    snap = dict(_request_metrics)
    if _latency_ms:
        arr = np.asarray(_latency_ms)
        snap["latency_p50_ms"] = float(np.percentile(arr, 50))
        snap["latency_p95_ms"] = float(np.percentile(arr, 95))
        snap["latency_n"] = len(arr)
    return snap


def _record_latency(ms: float) -> None:
    _latency_ms.append(ms)
    if len(_latency_ms) > _LATENCY_BUF:
        del _latency_ms[: len(_latency_ms) - _LATENCY_BUF]


def _gen_feed_request_id() -> str:
    """uuid4 hex 32자 — bite-api 와 bite-web 이 echo 해 user_events.feed_request_id 로 join."""
    return secrets.token_hex(16)


_POOL_BY_CATEGORY_SQL = text(
    """
    SELECT article_id, score, rank_global, category_id
    FROM (
      SELECT
        article_id, score, rank_global, category_id,
        ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY rank_global ASC) AS rn
      FROM recommendation_global
      WHERE category_id IN :cats
        AND (:lang IS NULL OR lang = :lang OR lang IS NULL)
    ) t
    WHERE rn <= :n
    """
).bindparams(bindparam("cats", expanding=True))


def _fetch_pool_for_categories(
    db: Session,
    quota: Dict[int, int],
    multiplier: int,
    lang: Optional[str],
) -> Dict[int, List[Tuple[str, float, float]]]:
    if not quota:
        return {}

    cat_ids = [c for c, q in quota.items() if q > 0]
    if not cat_ids:
        return {}

    max_per_cat = max(quota[c] for c in cat_ids) * max(1, multiplier)
    rows = db.execute(
        _POOL_BY_CATEGORY_SQL,
        {"cats": cat_ids, "n": int(max_per_cat), "lang": lang},
    ).all()

    out: Dict[int, List[Tuple[str, float, float]]] = {c: [] for c in cat_ids}
    for r in rows:
        cid = int(r[3])
        out.setdefault(cid, []).append((str(r[0]), float(r[1]), float(r[2])))
    return out


def _fetch_article_vectors(article_ids: List[str]) -> Dict[str, np.ndarray]:
    if not article_ids:
        return {}
    client = get_client()
    point_ids = [_to_article_point_id(aid) for aid in article_ids]
    id_to_aid = dict(zip(point_ids, article_ids))
    out: Dict[str, np.ndarray] = {}
    try:
        points = client.retrieve(
            collection_name=config.QDRANT_COLLECTION_NAME,
            ids=point_ids,
            with_vectors=True,
            with_payload=False,
        )
    except Exception as exc:
        log.warning("article vector batch retrieve failed: %s", exc)
        return {}
    for p in points:
        if p.vector is None:
            continue
        aid = id_to_aid.get(str(p.id))
        if aid is None:
            continue
        out[aid] = np.asarray(p.vector, dtype=np.float32)
    return out


def _select_within_category(
    candidates: List[Tuple[str, float, float]],
    quota: int,
    user_vec: Optional[np.ndarray],
    article_vecs: Dict[str, np.ndarray],
) -> List[Tuple[str, float]]:
    if not candidates or quota <= 0:
        return []

    if user_vec is None:
        return [(aid, score) for aid, score, _rank in candidates[:quota]]

    scored: List[Tuple[str, float]] = []
    for aid, score, _rank in candidates:
        v = article_vecs.get(aid)
        if v is None:
            scored.append((aid, score * 0.5))
            continue
        sim = cosine(user_vec, v)
        combined = 0.7 * score + 0.3 * (sim + 1.0) / 2.0
        scored.append((aid, combined))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:quota]


def _backfill(
    db: Session,
    needed: int,
    seen_ids: set,
    lang: Optional[str],
    rng: np.random.Generator,
) -> List[Tuple[str, Optional[int]]]:
    """글로벌 풀 top-N 에서 score-weighted random sample (deterministic 회피)."""
    if needed <= 0:
        return []

    rows = db.execute(
        text(
            """
            SELECT article_id, category_id, score
            FROM recommendation_global
            WHERE (:lang IS NULL OR lang = :lang OR lang IS NULL)
            ORDER BY rank_global ASC
            LIMIT :n
            """
        ),
        {"n": BACKFILL_POOL_SIZE + len(seen_ids), "lang": lang},
    ).all()

    pool = [(str(r[0]), (int(r[1]) if r[1] is not None else None), float(r[2])) for r in rows if str(r[0]) not in seen_ids]
    if not pool:
        return []

    weights = np.asarray([s for _, _, s in pool], dtype=np.float64)
    weights = np.clip(weights, 1e-6, None)
    probs = weights / weights.sum()
    take = min(needed, len(pool))
    idxs = rng.choice(len(pool), size=take, replace=False, p=probs)
    return [(pool[i][0], pool[i][1]) for i in idxs]


def _serve(
    db: Session,
    member_id: Optional[int],
    device_id: Optional[str],
    lang: Optional[str],
) -> RecommendItem:
    """회원/비회원 공통 서빙 흐름. anonymous 는 bandit 만 device 테이블."""
    started = time.perf_counter()
    is_anonymous = member_id is None
    rng = np.random.default_rng()
    feed_request_id = _gen_feed_request_id()

    _request_metrics["requests"] += 1
    if is_anonymous:
        _request_metrics["anonymous"] += 1

    # 1. bandit state
    if is_anonymous:
        state = bandit.load_or_init_device(db, device_id)  # type: ignore[arg-type]
    else:
        state = bandit.load_or_init(db, int(member_id))
    if not state:
        log.info("recommendation_global 비어있거나 풀에 카테고리 없음 → 빈 응답")
        _record_latency((time.perf_counter() - started) * 1000)
        return RecommendItem(articles=[])

    # 2. TS sample
    thetas = bandit.sample_thetas(state, rng=rng)

    # 3. quota 분배 (softmax temperature 적용 — bandit.QUOTA_TEMPERATURE)
    quota = bandit.allocate_quota(thetas, N_RESULTS, top_k_categories=TOP_K_CATEGORIES)
    if not quota:
        _record_latency((time.perf_counter() - started) * 1000)
        return RecommendItem(articles=[])

    # 4. Phase 2 user vector — 회원 전용
    user_vec = None if is_anonymous else get_user_vector(int(member_id))
    multiplier = CANDIDATE_MULTIPLIER if user_vec is not None else 1
    by_category = _fetch_pool_for_categories(db, quota, multiplier=multiplier, lang=lang)

    article_vecs: Dict[str, np.ndarray] = {}
    if user_vec is not None:
        all_ids: List[str] = []
        for cands in by_category.values():
            all_ids.extend(aid for aid, _, _ in cands)
        if all_ids:
            article_vecs = _fetch_article_vectors(all_ids)

    # 5. 카테고리별 select
    selected: List[Tuple[str, Optional[int], float]] = []
    seen: set[str] = set()
    for cat_id, q in quota.items():
        chosen = _select_within_category(by_category.get(cat_id, []), q, user_vec, article_vecs)
        for aid, _ in chosen:
            if aid in seen:
                continue
            seen.add(aid)
            selected.append((aid, cat_id, float(thetas.get(cat_id, 0.0))))

    # 6. quota 부족 → weighted random backfill
    if len(selected) < N_RESULTS:
        _request_metrics["backfilled"] += 1
        for aid, cid in _backfill(db, N_RESULTS - len(selected), seen, lang, rng):
            theta = float(thetas.get(cid, 0.0)) if cid is not None else 0.0
            selected.append((aid, cid, theta))
            seen.add(aid)

    # in-response shuffle
    rng.shuffle(selected)

    # 7. impression log (member_id 또는 device_id, feed_request_id)
    impression_rows = [
        (aid, cid, pos + 1, theta)
        for pos, (aid, cid, theta) in enumerate(selected)
    ]
    log_impressions(
        db,
        member_id=member_id,
        device_id=device_id,
        feed_request_id=feed_request_id,
        rows=impression_rows,
    )

    _record_latency((time.perf_counter() - started) * 1000)
    return RecommendItem(articles=[aid for aid, _, _ in selected], feed_request_id=feed_request_id)


def recommend_feeds(
    db: Session,
    member_id: int,
    lang: Optional[str] = None,
) -> RecommendItem:
    return _serve(db, member_id=int(member_id), device_id=None, lang=lang)


def recommend_feeds_anonymous(
    db: Session,
    device_id: str,
    lang: Optional[str] = None,
) -> RecommendItem:
    return _serve(db, member_id=None, device_id=str(device_id), lang=lang)
