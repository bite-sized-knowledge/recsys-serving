from typing import List
from app.api.search.qdrant_searcher import QdrantSearcher


async def search_articles(
    searcher: QdrantSearcher,
    query: str,
    collection_name: str,
    limit: int = QdrantSearcher.DEFAULT_RESULT_LIMIT,
) -> List[str]:
    if not query:
        raise ValueError("검색어를 제공해야 합니다.")

    results = await searcher.search(
        query=query,
        collection_name=collection_name,
        limit=limit,
    )

    return [str(result["article_id"]) for result in results if result.get("article_id")]
