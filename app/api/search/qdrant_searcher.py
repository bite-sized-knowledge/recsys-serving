import os
from dotenv import load_dotenv
from typing import List, Dict
from qdrant_client import QdrantClient
from embedder import TextEmbeddings

class QdrantSearcher:
    """
    Qdrant 벡터 DB에서 의미 기반 검색을 수행하는 클래스
    """
    def __init__(self):
        load_dotenv()
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_URL 또는 QDRANT_API_KEY가 .env 파일에 설정되지 않았습니다.")

        self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.embedder = TextEmbeddings()
        print("✅ QdrantSearcher 초기화 완료. Qdrant 클라이언트 연결됨.")

    async def search(
        self,
        query: str,
        collection_name: str = "bite-vectordb",
        limit: int = 5,
        dimensions: int = 512 
    ) -> List[Dict]:
        """
        주어진 텍스트 쿼리로 벡터 검색을 수행하고 결과를 반환
        """
        # 1. 검색어를 임베딩 벡터로 변환
        query_vector = await self.embedder.embed_text(
            text=query, 
            dimensions=dimensions
        )
        print("✅ 검색어 임베딩 완료")

        # 2. Qdrant에 벡터 검색 실행
        search_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        print("✅ Qdrant 벡터 검색 완료")

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