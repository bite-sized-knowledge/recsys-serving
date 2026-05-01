"""Search service entry point.

Cursor pagination은 stateless snapshot 방식. 첫 호출에서 hybrid_search이 fused
ranking을 max_pool개까지 계산하여 cursor에 통째로 (zlib + base64) 인코딩한다.
다음 페이지 요청은 cursor에서 snapshot을 꺼내 slice만 수행하므로 데이터 변경에
관계없이 동일한 검색 세션 안에서 결과가 결정적(deterministic)으로 유지된다.
query가 바뀌면 query_hash가 불일치하여 fresh fetch가 트리거된다.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
import zlib
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.core.config import config
from app.services import popular_queries
from app.services.hybrid_search import SearchMode, hybrid_search
from app.services.qdrant_client import SearchFilters

MAX_QUERY_LEN = 200


def _query_hash(query: str) -> str:
    # MUST stay in sync with bite-api/internal/event/service.go normalizeAndHashQuery
    # (lower + strip + sha1 → hex 12). 한쪽만 바뀌면 검색 분석 join이 silent하게 깨짐.
    return hashlib.sha1(query.strip().lower().encode("utf-8")).hexdigest()[:12]


def _new_query_id() -> str:
    return uuid.uuid4().hex


def _encode_cursor(snapshot: list[str], offset: int, query_hash: str, query_id: str) -> str:
    payload = {"f": snapshot, "o": offset, "q": query_hash, "i": query_id}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return base64.urlsafe_b64encode(compressed).decode("ascii")


def _decode_cursor(cursor: Optional[str]) -> dict:
    if not cursor:
        return {}
    try:
        compressed = base64.urlsafe_b64decode(cursor.encode("ascii"))
        raw = zlib.decompress(compressed)
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError("잘못된 cursor입니다.") from exc
    if not isinstance(payload, dict):
        raise ValueError("잘못된 cursor 페이로드입니다.")
    return payload


async def search_articles(
    db: Session,
    query: str,
    limit: int,
    cursor: Optional[str] = None,
    category_id: Optional[int] = None,
    lang: Optional[str] = None,
    blog_id: Optional[int] = None,
    published_after: Optional[float] = None,
    published_before: Optional[float] = None,
    mode: SearchMode = SearchMode.HYBRID,
) -> Tuple[List[str], Optional[str], str]:
    """Returns (article_ids, next_cursor, query_id).

    query_id는 같은 query로 시작된 검색 세션 전체에서 동일(snapshot cursor 안에 echo).
    클라이언트가 후속 노출/클릭 이벤트에 첨부하면 (query_id, clicked_position) 단위
    분석이 가능하다.
    """
    if not query or not query.strip():
        raise ValueError("검색어를 제공해야 합니다.")
    if len(query) > MAX_QUERY_LEN:
        raise ValueError(f"검색어는 {MAX_QUERY_LEN}자 이하여야 합니다.")

    normalized_q = query.strip()
    fresh_hash = _query_hash(normalized_q)

    payload = _decode_cursor(cursor)
    snapshot: list[str] = list(payload.get("f") or [])
    offset = int(payload.get("o", 0) or 0)
    cached_hash = payload.get("q", "")
    cached_query_id = payload.get("i") or ""

    # Snapshot이 stale(다른 query) 또는 부재면 fresh hybrid 호출.
    if not snapshot or cached_hash != fresh_hash or not cached_query_id:
        filters = SearchFilters(
            category_id=category_id,
            lang=lang,
            blog_id=blog_id,
            published_after=published_after,
            published_before=published_before,
        )
        snapshot = await hybrid_search(
            db=db,
            query=normalized_q,
            filters=filters,
            max_pool=config.SEARCH_MAX_POOL,
            mode=mode,
        )
        offset = 0
        cached_query_id = _new_query_id()
        # 인기 검색어 카운트는 fresh fetch 시점에만 (cursor follow-up은 같은 검색)
        if snapshot:
            popular_queries.record(normalized_q)

    page = snapshot[offset : offset + limit]

    next_offset = offset + limit
    next_cursor: Optional[str] = None
    if next_offset < len(snapshot):
        next_cursor = _encode_cursor(snapshot, next_offset, fresh_hash, cached_query_id)

    return page, next_cursor, cached_query_id
