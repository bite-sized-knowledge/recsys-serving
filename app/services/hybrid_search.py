"""BM25 (MySQL FULLTEXT) + Dense (Qdrant) hybrid search with RRF fusion."""

from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Iterable, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import config
from app.services import embedding_cache
from app.services.embedder import QueryEncoder
from app.services.qdrant_client import SearchFilters, dense_search

log = logging.getLogger(__name__)


class SearchMode(str, Enum):
    HYBRID = "hybrid"
    DENSE = "dense"
    FULLTEXT = "fulltext"


_BM25_SQL = text(
    """
    SELECT a.article_id
    FROM article a
    WHERE MATCH(a.title, a.description, a.keywords) AGAINST (:q IN BOOLEAN MODE)
      AND (:category_id IS NULL OR a.category_id = :category_id)
      AND (:lang IS NULL OR a.lang = :lang)
      AND (:blog_id IS NULL OR a.blog_id = :blog_id)
      AND (:published_after IS NULL OR a.published_at >= FROM_UNIXTIME(:published_after))
      AND (:published_before IS NULL OR a.published_at <= FROM_UNIXTIME(:published_before))
    ORDER BY MATCH(a.title, a.description, a.keywords) AGAINST (:q IN BOOLEAN MODE) DESC,
             a.published_at DESC
    LIMIT :limit
    """
)


def _sanitize_phrase(query: str) -> str:
    """BOOLEAN MODE 메타문자를 제거하고 phrase로 감싼다.

    bite-api `internal/article/repository.go:361-365` 로직을 그대로 포팅.
    """
    cleaned = query
    for ch in ('"', "+", "-", "*", "~"):
        cleaned = cleaned.replace(ch, " ")
    cleaned = cleaned.strip()
    return f'"{cleaned}"' if cleaned else ""


def _bm25_query(
    db: Session,
    query: str,
    filters: SearchFilters,
    top_n: int,
) -> list[str]:
    phrase = _sanitize_phrase(query)
    if not phrase:
        return []
    rows = db.execute(
        _BM25_SQL,
        {
            "q": phrase,
            "category_id": filters.category_id,
            "lang": filters.lang,
            "blog_id": filters.blog_id,
            "published_after": filters.published_after,
            "published_before": filters.published_before,
            "limit": top_n,
        },
    ).all()
    return [str(r[0]) for r in rows]


def rrf_fuse(*rankings: Iterable[str], k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


async def _encode_query(query: str) -> Optional[list[float]]:
    cached = embedding_cache.get(query)
    if cached is not None:
        return cached
    encoder = QueryEncoder.instance()
    vector = await asyncio.to_thread(encoder.encode, query)
    embedding_cache.put(query, vector)
    return vector


async def hybrid_search(
    db: Session,
    query: str,
    filters: SearchFilters,
    limit: int,
    offset: int = 0,
    mode: SearchMode = SearchMode.HYBRID,
) -> list[str]:
    if not query or not query.strip():
        return []

    bm25_top_n = config.SEARCH_BM25_TOP_N
    dense_top_n = config.SEARCH_DENSE_TOP_N

    bm25_ids: list[str] = []
    dense_ids: list[str] = []

    if mode in (SearchMode.HYBRID, SearchMode.FULLTEXT):
        bm25_task = asyncio.to_thread(_bm25_query, db, query, filters, bm25_top_n)
    else:
        bm25_task = None

    if mode in (SearchMode.HYBRID, SearchMode.DENSE):
        vector = await _encode_query(query)
        dense_task = asyncio.to_thread(dense_search, vector, filters, dense_top_n)
    else:
        dense_task = None

    if bm25_task is not None and dense_task is not None:
        bm25_ids, dense_results = await asyncio.gather(bm25_task, dense_task)
        dense_ids = [aid for aid, _ in dense_results]
    elif bm25_task is not None:
        bm25_ids = await bm25_task
    elif dense_task is not None:
        dense_results = await dense_task
        dense_ids = [aid for aid, _ in dense_results]

    if mode is SearchMode.FULLTEXT:
        fused = bm25_ids
    elif mode is SearchMode.DENSE:
        fused = dense_ids
    else:
        fused = rrf_fuse(bm25_ids, dense_ids, k=config.SEARCH_RRF_K)

    return fused[offset : offset + limit]
