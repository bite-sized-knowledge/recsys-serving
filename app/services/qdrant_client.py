"""Qdrant 클라이언트 + dense search + filter 변환 + payload index 부트스트랩."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from app.core.config import config

log = logging.getLogger(__name__)

_client: Optional[QdrantClient] = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY or None,
            prefer_grpc=False,
            timeout=10,
        )
    return _client


@dataclass
class SearchFilters:
    category_id: Optional[int] = None
    lang: Optional[str] = None
    blog_id: Optional[int] = None
    published_after: Optional[float] = None
    published_before: Optional[float] = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (
                self.category_id,
                self.lang,
                self.blog_id,
                self.published_after,
                self.published_before,
            )
        )


def to_qdrant_filter(f: SearchFilters) -> Optional[qm.Filter]:
    if f.is_empty():
        return None
    must: list[qm.FieldCondition] = []
    if f.category_id is not None:
        must.append(qm.FieldCondition(key="category_id", match=qm.MatchValue(value=f.category_id)))
    if f.lang is not None:
        must.append(qm.FieldCondition(key="lang", match=qm.MatchValue(value=f.lang)))
    if f.blog_id is not None:
        must.append(qm.FieldCondition(key="blog_id", match=qm.MatchValue(value=f.blog_id)))
    if f.published_after is not None or f.published_before is not None:
        must.append(
            qm.FieldCondition(
                key="published_at",
                range=qm.Range(gte=f.published_after, lte=f.published_before),
            )
        )
    return qm.Filter(must=must)


def dense_search(
    vector: np.ndarray,
    filters: SearchFilters,
    top_n: int,
) -> list[tuple[str, float]]:
    """Returns [(article_id, score)] sorted by score desc."""
    client = get_client()
    result = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=vector.tolist(),
        query_filter=to_qdrant_filter(filters),
        limit=top_n,
        with_payload=["article_id"],
        with_vectors=False,
    )
    out: list[tuple[str, float]] = []
    for point in result.points:
        article_id = point.payload.get("article_id") if point.payload else None
        if article_id is None:
            continue
        out.append((str(article_id), float(point.score)))
    return out


_PAYLOAD_INDEXES: tuple[tuple[str, qm.PayloadSchemaType], ...] = (
    ("category_id", qm.PayloadSchemaType.INTEGER),
    ("lang", qm.PayloadSchemaType.KEYWORD),
    ("blog_id", qm.PayloadSchemaType.INTEGER),
    ("published_at", qm.PayloadSchemaType.FLOAT),
)


def ensure_payload_indexes() -> None:
    """Idempotent payload index creation. 실패는 경고 후 진행."""
    client = get_client()
    for field_name, schema in _PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=schema,
            )
            log.info("Qdrant payload index created: %s", field_name)
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "exists" in msg:
                log.debug("Qdrant payload index already exists: %s", field_name)
            else:
                log.warning("Qdrant payload index creation failed for %s: %s", field_name, exc)
