"""
Phase 1+2 추천 서빙.

흐름:
  1. bandit state load (or lazy init from member_interest)
  2. Beta TS sample → 각 카테고리 theta
  3. softmax(theta) × N → 카테고리 quota 분배 (top-K 카테고리)
  4. recommendation_global 에서 카테고리별 후보 가져옴 (rank_global ASC, quota * candidate_factor)
  5. Phase 2: user_profile 있으면 within-category rerank by cosine(user_vec, article_vec)
  6. dedup, position 부여, recommendation_impression 적재 후 응답
"""
from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from app.core.config import config
from app.services import bandit
from app.services.impression_logger import log_impressions
from app.services.user_vector_lookup import (
    _to_article_point_id,
    cosine,
    get_user_vector,
)
from app.services.qdrant_client import get_client

from .schema import RecommendItem

log = logging.getLogger(__name__)

N_RESULTS = 10
TOP_K_CATEGORIES = 6
CANDIDATE_MULTIPLIER = 4  # quota * 4 만큼 카테고리 후보 가져와 rerank


_POOL_BY_CATEGORY_SQL = text(
    """
    SELECT article_id, score, rank_global, category_id
    FROM (
      SELECT
        article_id, score, rank_global, category_id,
        ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY rank_global ASC) AS rn
      FROM recommendation_global
      WHERE category_id IN :cats
    ) t
    WHERE rn <= :n
    """
).bindparams(bindparam("cats", expanding=True))


def _fetch_pool_for_categories(
    db: Session,
    quota: Dict[int, int],
    multiplier: int,
) -> Dict[int, List[Tuple[str, float, float]]]:
    """
    카테고리별 후보 [(article_id, score, rank_global)] 단일 쿼리.
    multiplier=1 (rerank 없음, quota 만큼) / >1 (rerank 용 더 가져옴).
    """
    if not quota:
        return {}

    cat_ids = [c for c, q in quota.items() if q > 0]
    if not cat_ids:
        return {}

    max_per_cat = max(quota[c] for c in cat_ids) * max(1, multiplier)
    rows = db.execute(_POOL_BY_CATEGORY_SQL, {"cats": cat_ids, "n": int(max_per_cat)}).all()

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
    """카테고리 후보 중 quota 개 선택. Phase 2 user_vec 있으면 cosine rerank."""
    if not candidates or quota <= 0:
        return []

    if user_vec is None:
        # Phase 1 fallback: 글로벌 score 그대로
        return [(aid, score) for aid, score, _rank in candidates[:quota]]

    scored: List[Tuple[str, float]] = []
    for aid, score, _rank in candidates:
        v = article_vecs.get(aid)
        if v is None:
            # 임베딩 missing — 글로벌 score 그대로 (낮은 우선순위)
            scored.append((aid, score * 0.5))
            continue
        sim = cosine(user_vec, v)
        # 0.7 * 글로벌 + 0.3 * 유사도
        combined = 0.7 * score + 0.3 * (sim + 1.0) / 2.0
        scored.append((aid, combined))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:quota]


def recommend_feeds(db: Session, member_id: int) -> RecommendItem:
    # 1. bandit state
    state = bandit.load_or_init(db, member_id)
    if not state:
        log.info("recommendation_global 비어있음 또는 풀에 카테고리 없음 → 빈 응답")
        return RecommendItem(articles=[])

    # 2. TS sample
    rng = np.random.default_rng()
    thetas = bandit.sample_thetas(state, rng=rng)

    # 3. quota 분배
    quota = bandit.allocate_quota(thetas, N_RESULTS, top_k_categories=TOP_K_CATEGORIES)
    if not quota:
        return RecommendItem(articles=[])

    # 4. Phase 2 user vector — 있으면 within-category rerank 위해 4배 가져오기, 없으면 quota 만큼
    user_vec = get_user_vector(member_id)
    multiplier = CANDIDATE_MULTIPLIER if user_vec is not None else 1
    by_category = _fetch_pool_for_categories(db, quota, multiplier=multiplier)

    article_vecs: Dict[str, np.ndarray] = {}
    if user_vec is not None:
        all_candidate_ids: List[str] = []
        for cands in by_category.values():
            all_candidate_ids.extend(aid for aid, _, _ in cands)
        if all_candidate_ids:
            article_vecs = _fetch_article_vectors(all_candidate_ids)

    # 5. 카테고리별 select
    selected: List[Tuple[str, Optional[int], float]] = []
    seen_ids: set[str] = set()
    for cat_id, q in quota.items():
        cands = by_category.get(cat_id, [])
        chosen = _select_within_category(cands, q, user_vec, article_vecs)
        for aid, _ in chosen:
            if aid in seen_ids:
                continue
            seen_ids.add(aid)
            selected.append((aid, cat_id, float(thetas.get(cat_id, 0.0))))

    # quota 부족 → 글로벌 풀로 backfill (top by rank_global)
    if len(selected) < N_RESULTS:
        fill_n = N_RESULTS - len(selected)
        rows = db.execute(
            text(
                "SELECT article_id, category_id FROM recommendation_global "
                "ORDER BY rank_global ASC LIMIT :n"
            ),
            {"n": fill_n + len(seen_ids)},
        ).all()
        for r in rows:
            aid = str(r[0])
            if aid in seen_ids:
                continue
            cid = int(r[1]) if r[1] is not None else None
            theta_val = float(thetas.get(cid, 0.0)) if cid is not None else 0.0
            selected.append((aid, cid, theta_val))
            seen_ids.add(aid)
            if len(selected) >= N_RESULTS:
                break

    rng.shuffle(selected)

    # 6. impression log (Optional cid 그대로 전달 — DB 컬럼 NULL 허용)
    impression_rows = [
        (aid, cid, pos + 1, theta)
        for pos, (aid, cid, theta) in enumerate(selected)
    ]
    log_impressions(db, member_id, impression_rows)

    return RecommendItem(articles=[aid for aid, _, _ in selected])
