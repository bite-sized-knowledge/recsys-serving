from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm.session import Session

from app.db import get_db
from app.api.search.schema import SearchResponse
from app.api.search.service import search_articles

router = APIRouter()


@router.get("", response_model=SearchResponse, tags=["search"], status_code=status.HTTP_200_OK)
async def get_search_results(
    query: str,
    limit: int = Query(
        20,
        ge=1,
        le=100,
        description="검색 결과 상한 (최대 100, 기본 20)",
    ),
    db: Session = Depends(get_db),
) -> SearchResponse:
    try:
        article_ids = await search_articles(
            db=db,
            query=query,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)) from exc
    except Exception as exc:  
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 중 서버 오류가 발생했습니다. : {exc}",
        ) from exc

    return SearchResponse(articles=article_ids)
