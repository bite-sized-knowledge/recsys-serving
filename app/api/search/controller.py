from fastapi import APIRouter, Depends, HTTPException, Query, status
from app.api.search.dependencies import get_collection_name, get_qdrant_searcher
from app.api.search.qdrant_searcher import QdrantSearcher
from app.api.search.schema import SearchResponse
from app.api.search.service import search_articles

router = APIRouter()


@router.get("", response_model=SearchResponse, tags=["search"], status_code=status.HTTP_200_OK)
async def get_search_results(
    query: str,
    limit: int = Query(
        QdrantSearcher.DEFAULT_RESULT_LIMIT,
        ge=1,
        le=QdrantSearcher.DEFAULT_RESULT_LIMIT,
        description="검색 결과 상한 (최대 100, 기본 100)",
    ),
    searcher: QdrantSearcher = Depends(get_qdrant_searcher),
    collection_name: str = Depends(get_collection_name),
) -> SearchResponse:
    try:
        article_ids = await search_articles(
            searcher=searcher,
            query=query,
            collection_name=collection_name,
            limit=limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="검색 중 서버 오류가 발생했습니다.",
        ) from exc

    return SearchResponse(articles=article_ids)
