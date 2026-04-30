"""Phase 2: Qdrant `user_profile` collection 에서 user vector retrieve.

배치 (recommender) 가 채우고, feedback endpoint 가 incremental EMA push 한다.
이 모듈은 read-only lookup + EMA push 두 가지 동작.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np

from app.services.qdrant_client import get_client

log = logging.getLogger(__name__)

USER_PROFILE_COLLECTION = "user_profile"
ITEM_COLLECTION = "bite-vectordb"  # article 임베딩 origin
EMA_DECAY = 0.9  # incremental EMA: new = decay * old + (1 - decay) * article_vec
UUID_NAMESPACE = uuid.NAMESPACE_DNS


def _to_article_point_id(article_id: str) -> str:
    return str(uuid.uuid5(UUID_NAMESPACE, article_id))


def get_user_vector(member_id: int) -> Optional[np.ndarray]:
    """user_profile collection 에서 vector retrieve. 없으면 None."""
    client = get_client()
    try:
        points = client.retrieve(
            collection_name=USER_PROFILE_COLLECTION,
            ids=[int(member_id)],
            with_vectors=True,
            with_payload=False,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "not found" in msg or "doesn't exist" in msg:
            return None
        log.warning("user_profile retrieve failed for member %s: %s", member_id, exc)
        return None
    if not points:
        return None
    p = points[0]
    if p.vector is None:
        return None
    return np.asarray(p.vector, dtype=np.float32)


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 0:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def push_event_to_profile(member_id: int, article_id: str) -> bool:
    """
    실시간 incremental EMA. 이 article 임베딩을 user vector 에 섞는다.
    기존 user_profile 있으면 EMA, 없으면 article_vec 그대로.
    """
    client = get_client()

    # 1. fetch article vector
    aid = _to_article_point_id(article_id)
    try:
        art_points = client.retrieve(
            collection_name=ITEM_COLLECTION,
            ids=[aid],
            with_vectors=True,
            with_payload=False,
        )
    except Exception as exc:
        log.warning("article vector retrieve failed (%s): %s", article_id, exc)
        return False
    if not art_points or art_points[0].vector is None:
        return False
    article_vec = np.asarray(art_points[0].vector, dtype=np.float32)

    # 2. fetch existing user vector
    old_vec = get_user_vector(member_id)
    if old_vec is None:
        new_vec = _l2_normalize(article_vec)
    else:
        mixed = EMA_DECAY * old_vec + (1.0 - EMA_DECAY) * article_vec
        new_vec = _l2_normalize(mixed)

    # 3. upsert (ensure collection 은 batch 가 만든다 — 없으면 fail 후 fallback)
    from qdrant_client.http import models as qm
    try:
        client.upsert(
            collection_name=USER_PROFILE_COLLECTION,
            points=[
                qm.PointStruct(
                    id=int(member_id),
                    vector=new_vec.tolist(),
                    payload={"member_id": int(member_id)},
                )
            ],
        )
        return True
    except Exception as exc:
        msg = str(exc).lower()
        if "doesn't exist" in msg or "not found" in msg:
            log.info("user_profile collection 미존재 — recommender 배치가 만들 때까지 skip")
            return False
        log.warning("user_profile upsert failed (member %s): %s", member_id, exc)
        return False


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
