"""Search service — Hybrid (BM25 + Dense) + RRF fusion entry point.

Cursor pagination은 1차에서 단순 offset(opaque base64 인코딩)으로 처리한다.
RRF의 stable ordering은 보장되지 않지만 동일 query에 대해 같은 BM25/dense top-N이
들어오면 결과는 deterministic이므로 실용상 안정적이다.
"""

from __future__ import annotations

import base64
import json
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.services.hybrid_search import SearchMode, hybrid_search
from app.services.qdrant_client import SearchFilters

MAX_QUERY_LEN = 200
MAX_OFFSET = 1000


def _decode_cursor(cursor: Optional[str]) -> int:
    if not cursor:
        return 0
    try:
        decoded = base64.urlsafe_b64decode(cursor.encode("ascii"))
        payload = json.loads(decoded)
        offset = int(payload.get("o", 0))
    except (ValueError, json.JSONDecodeError):
        raise ValueError("잘못된 cursor입니다.")
    if offset < 0 or offset > MAX_OFFSET:
        raise ValueError("cursor offset 범위를 벗어났습니다.")
    return offset


def _encode_cursor(offset: int) -> str:
    payload = json.dumps({"o": offset}, separators=(",", ":")).encode("ascii")
    return base64.urlsafe_b64encode(payload).decode("ascii")


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
) -> Tuple[List[str], Optional[str]]:
    if not query or not query.strip():
        raise ValueError("검색어를 제공해야 합니다.")
    if len(query) > MAX_QUERY_LEN:
        raise ValueError(f"검색어는 {MAX_QUERY_LEN}자 이하여야 합니다.")

    offset = _decode_cursor(cursor)
    filters = SearchFilters(
        category_id=category_id,
        lang=lang,
        blog_id=blog_id,
        published_after=published_after,
        published_before=published_before,
    )

    article_ids = await hybrid_search(
        db=db,
        query=query.strip(),
        filters=filters,
        limit=limit,
        offset=offset,
        mode=mode,
    )

    next_cursor: Optional[str] = None
    if len(article_ids) == limit:
        next_offset = offset + limit
        if next_offset <= MAX_OFFSET:
            next_cursor = _encode_cursor(next_offset)

    return article_ids, next_cursor
