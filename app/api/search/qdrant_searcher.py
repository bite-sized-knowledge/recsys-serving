from typing import Dict, List, Optional
from qdrant_client import QdrantClient
from app.api.search.embedder import TextEmbeddings

class QdrantSearcher:
    """
    Qdrant 벡터 DB에서 의미 기반 검색을 수행하는 클래스
    """
    DEFAULT_RESULT_LIMIT = 100

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        collection_name: str,
        client: Optional[QdrantClient] = None,
        embedder: Optional[TextEmbeddings] = None,
    ):
        if not url:
            raise ValueError("QDRANT_URL 환경 변수가 설정되지 않았습니다.")
        if not api_key:
            raise ValueError("QDRANT_API_KEY 환경 변수가 설정되지 않았습니다.")
        if not collection_name:
            raise ValueError("QDRANT_COLLECTION_NAME 환경 변수가 설정되지 않았습니다.")

        self.client = client or QdrantClient(
            url=url,
            api_key=api_key,
            check_compatibility=False,
        )
        self.embedder = embedder or TextEmbeddings()
        self.collection_name = collection_name
        print("QdrantSearcher 초기화 완료. Qdrant 클라이언트 연결됨.")

    async def search(
        self,
        *,
        query: str,
        collection_name: Optional[str] = None,
        limit: Optional[int] = None,
        dimensions: int = 512 
    ) -> List[Dict]:
        """
        주어진 텍스트 쿼리로 벡터 검색을 수행하고 결과를 반환
        """
        if not query:
            raise ValueError("검색어를 제공해야 합니다.")

        resolved_collection = collection_name or self.collection_name
        resolved_limit = (
            self.DEFAULT_RESULT_LIMIT
            if limit is None
            else min(limit, self.DEFAULT_RESULT_LIMIT)
        )

        if resolved_limit <= 0:
            raise ValueError("limit 파라미터는 1 이상의 정수여야 합니다.")

        # 1. 검색어를 임베딩 벡터로 변환
        query_vector = await self.embedder.embed_text(
            text=query, 
            dimensions=dimensions
        )
        print(f"Search Query : {query}, embedding done")

        # 2. Qdrant에 벡터 검색 실행
        search_result = self._execute_query(
            collection_name=resolved_collection,
            query_vector=query_vector,
            limit=resolved_limit,
        )
        print("Qdrant Search Completed")

        # 3. 결과를 가공하여 반환
        processed_results = []
        for result in search_result:
            payload = result.payload
            processed_results.append({
                "article_id": payload.get('article_id', 'N/A'),
                "category": payload.get('category', 'N/A'),
                "score": result.score
            })
        
        return processed_results
    
    # 컬렉션의 총 개수를 가져오는 메서드 추가
    async def get_collection_count(self, collection_name: str):
        try:
            # count() 메소드를 사용하여 컬렉션 내 포인트(값)의 개수를 얻음
            count_result = self.client.count(collection_name=collection_name, exact=True)
            return count_result.count
        except Exception as e:
            print(f"컬렉션 개수를 가져오는 중 오류 발생: {e}")
            return None

    def _execute_query(self, *, collection_name: str, query_vector: List[float], limit: int):
        query_kwargs = dict(
            collection_name=collection_name,
            limit=limit,
            with_payload=True,
        )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                query=query_vector,
                **query_kwargs,
            )
            return response.points if hasattr(response, "points") else response

        # 구 버전 클라이언트 호환
        return self.client.search(
            query_vector=query_vector,
            **query_kwargs,
        )
