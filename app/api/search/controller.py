from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm.session import Session

from app.api.search.schema import (
    SearchResponse,
    SuggestResponse,
    UnderstandRequest,
    UnderstandResponse,
)
from app.api.search.service import search_articles
from app.core.auth import verify_api_key
from app.db import get_db
from app.services import popular_queries
from app.services.hybrid_search import SearchMode

router = APIRouter()


@router.get("", response_model=SearchResponse, tags=["search"], status_code=status.HTTP_200_OK)
async def get_search_results(
    query: str,
    limit: int = Query(20, ge=1, le=100, description="검색 결과 상한 (최대 100, 기본 20)"),
    cursor: Optional[str] = Query(None, description="opaque cursor (이전 응답의 next 값)"),
    category_id: Optional[int] = Query(None),
    lang: Optional[str] = Query(None, max_length=10),
    blog_id: Optional[int] = Query(None),
    published_after: Optional[float] = Query(None, description="epoch 초"),
    published_before: Optional[float] = Query(None, description="epoch 초"),
    mode: SearchMode = Query(
        SearchMode.HYBRID,
        description="hybrid(default) | hybrid_rerank | dense | fulltext",
    ),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
) -> SearchResponse:
    try:
        article_ids, next_cursor, query_id = await search_articles(
            db=db,
            query=query,
            limit=limit,
            cursor=cursor,
            category_id=category_id,
            lang=lang,
            blog_id=blog_id,
            published_after=published_after,
            published_before=published_before,
            mode=mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="검색 중 서버 오류가 발생했습니다.",
        )

    return SearchResponse(articles=article_ids, next=next_cursor, query_id=query_id)


@router.post(
    "/understand",
    response_model=UnderstandResponse,
    tags=["search"],
    status_code=status.HTTP_200_OK,
)
async def understand_query(
    body: UnderstandRequest,
    _api_key: str = Depends(verify_api_key),
) -> UnderstandResponse:
    """Natural-language query → 구조화된 의도/필터 힌트.

    Deterministic regex 기반(이전 LLM 의존 제거). 응답 schema는 backward
    compatible. category_hint는 heuristic 정확도 부족으로 항상 null —
    추후 작은 intent classifier로 대체 검토.
    """
    if not body.query or not body.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query가 필요합니다.",
        )

    from app.services.query_understanding import analyze, recency_to_published_after

    parsed = analyze(body.query.strip())
    return UnderstandResponse(
        search_keywords=parsed.get("search_keywords"),
        lang=parsed.get("lang") if parsed.get("lang") in ("ko", "en") else None,
        recency=parsed.get("recency"),
        category_hint=parsed.get("category_hint"),
        intent=parsed.get("intent"),
        published_after=recency_to_published_after(parsed.get("recency")),
    )


@router.get(
    "/suggest",
    response_model=SuggestResponse,
    tags=["search"],
    status_code=status.HTTP_200_OK,
)
async def suggest_queries(
    q: str = Query("", description="prefix. 비어 있으면 전역 인기 검색어"),
    limit: int = Query(8, ge=1, le=20),
    _api_key: str = Depends(verify_api_key),
) -> SuggestResponse:
    suggestions = popular_queries.suggest(q, limit=limit)
    return SuggestResponse(suggestions=suggestions)
