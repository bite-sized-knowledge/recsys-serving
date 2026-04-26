import logging
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
from app.core.config import config
from app.db import get_db
from app.services import popular_queries
from app.services.hybrid_search import SearchMode

router = APIRouter()
log = logging.getLogger(__name__)


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
        article_ids, next_cursor = await search_articles(
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

    return SearchResponse(articles=article_ids, next=next_cursor)


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

    LLM 호출이라 ~500ms 걸린다. 검색 latency 직격을 피하기 위해 별도 호출.
    클라이언트가 결과를 받아 /search 호출에 필터로 적용한다.
    """
    if not config.QU_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="query understanding이 비활성화되어 있습니다.",
        )
    if not body.query or not body.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="query가 필요합니다.",
        )

    try:
        from app.services.query_understanding import (
            QueryUnderstander,
            recency_to_published_after,
        )

        understander = QueryUnderstander.instance()
        parsed = understander.analyze(body.query.strip())
    except FileNotFoundError as exc:
        log.warning("QU model missing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="query understanding model이 준비되지 않았습니다.",
        ) from exc
    except Exception as exc:
        log.warning("QU failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="query understanding 호출 중 오류가 발생했습니다.",
        ) from exc

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
