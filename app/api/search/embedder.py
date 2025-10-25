import os
import json
import boto3
import asyncio
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from numpy.linalg import norm

class TextEmbeddings:
    """
    AWS Bedrock의 Titan Embedding 모델을 사용하여 텍스트를 임베딩하는 클래스.
    긴 텍스트는 자동으로 분할(chunking)하여 평균 임베딩을 계산
    """
    accept = "application/json"
    content_type = "application/json"

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "ap-northeast-2",
        chunk_size: int = None,
        chunk_overlap: int = 200
    ):
        """
        클래스 생성자: Bedrock 클라이언트와 텍스트 분할기(chunker)를 초기화
        """
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 5000))
        self.chunk_overlap = chunk_overlap
        self.chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def _embed_once(self, text: str, dimensions: int, normalize: bool) -> List[float]:
        """
        [내부 함수] 단일 API 호출로 텍스트 조각을 임베딩
        """
        body = json.dumps({
            "inputText": text,
            "dimensions": dimensions,
            "normalize": normalize
        })
        try:
            response = self.bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                accept=self.accept,
                contentType=self.content_type
            )
            resp_body = json.loads(response["body"].read())
            return resp_body["embedding"]
        except Exception as e:
            print(f"[TitanEmbedding] Failed to embed: {e}")
            raise

    async def embed_text(self, text: str, dimensions: int, normalize: bool = True) -> List[float]:
        """
        주어진 텍스트를 임베딩 벡터로 변환
        """
        chunks = self.chunker.split_text(text)

        if len(chunks) == 1:
            return await asyncio.to_thread(self._embed_once, text, dimensions, normalize)

        # 텍스트가 여러 조각으로 나뉜 경우, 각 조각을 병렬로 임베딩하고 평균을 계산
        tasks = [
            asyncio.to_thread(self._embed_once, chunk, dimensions, False)
            for chunk in chunks
        ]
        embeddings = await asyncio.gather(*tasks)

        avg = [
            sum(values) / len(embeddings)
            for values in zip(*embeddings)
        ]
        if normalize:
            l2 = norm(avg)
            avg = [v / l2 for v in avg]

        return avg