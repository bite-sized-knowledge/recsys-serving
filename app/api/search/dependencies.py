from functools import lru_cache
from fastapi import HTTPException, status
from app.api.search.qdrant_searcher import QdrantSearcher
from app.core.config import config


@lru_cache
def get_collection_name() -> str:
    return config.QDRANT_COLLECTION_NAME


@lru_cache
def get_qdrant_searcher() -> QdrantSearcher:
    try:
        return QdrantSearcher(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
            collection_name=config.QDRANT_COLLECTION_NAME,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QdrantSearcher 초기화 실패: {exc}"
        ) from exc
