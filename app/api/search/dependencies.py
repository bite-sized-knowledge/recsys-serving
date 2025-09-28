import os
from functools import lru_cache
from fastapi import HTTPException, status
from dotenv import load_dotenv
from app.api.search.qdrant_searcher import QdrantSearcher

load_dotenv()


@lru_cache
def get_collection_name() -> str:
    collection_name = os.getenv("QDRANT_COLLECTION_NAME")
    if not collection_name:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="QDRANT_COLLECTION_NAME 환경 변수가 설정되지 않았습니다."
        )
    return collection_name


@lru_cache
def get_qdrant_searcher() -> QdrantSearcher:
    try:
        return QdrantSearcher()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"QdrantSearcher 초기화 실패: {exc}"
        ) from exc
