from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm.session import Session

from app.api.search.schema import SearchResponse
from app.api.search.service import search_articles
from app.core.auth import verify_api_key
from app.db import get_db
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
    mode: SearchMode = Query(SearchMode.HYBRID, description="hybrid | dense | fulltext"),
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
