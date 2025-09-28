import os
from fastapi import FastAPI, HTTPException
from typing import List, Optional
from dotenv import load_dotenv
from qdrant_searcher import QdrantSearcher

# 1. 환경 변수 로드
load_dotenv()
collection_name = os.getenv("QDRANT_COLLECTION_NAME")

# 2. FastAPI 앱 및 QdrantSearcher 인스턴스 생성
app = FastAPI()
try:
    searcher = QdrantSearcher()
except ValueError as e:
    print(f"오류: QdrantSearcher 초기화 실패 - {e}")

# 3. 검색 API 엔드포인트 정의
@app.get("/search", response_model=List[str])
async def search_articles(query: str, points: int = 10):
    """
    주어진 검색어(query)로 Qdrant DB에서 유사한 article_id를 검색
    데이터는 URL 쿼리 파라미터(Query Parameters)를 통해 전달
    - **query**: 검색할 텍스트 (필수)
    - **points**: 반환할 결과의 최대 개수 (기본값: 10)
    """
    if not query:
        raise HTTPException(status_code=400, detail="검색어를 제공해야 합니다.")

    if not searcher:
        raise HTTPException(status_code=500, detail="서버가 Qdrant 클라이언트에 연결되지 않았습니다.")

    print(f"🔍 검색 요청: '{query}', 포인트: {points}개")
    try:
        results = await searcher.search(
            query=query,
            collection_name=collection_name,
            limit=points
        )
        article_ids = [result['article_id'] for result in results]
        print(f"✅ 검색 완료. 총 {len(article_ids)}개의 결과 반환.")
        return article_ids
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=f"검색 중 서버 오류 발생: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)