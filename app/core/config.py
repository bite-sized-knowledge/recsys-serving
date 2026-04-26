from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config(BaseSettings):
    ENVIRONMENT: str = "prod"
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_USER: str = "bite-dev"
    DB_PASSWORD: str
    DB_NAME: str = "bite"

    RECSYS_API_KEY: str = ""

    QDRANT_URL: str = "http://127.0.0.1:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "bite-vectordb"

    EMBEDDING_MODEL_DIR: str = str(PROJECT_ROOT / "models" / "qwen3-embed-onnx")
    EMBEDDING_MODEL_FILENAME: str = "model.onnx"
    EMBEDDING_DIM: int = 1024
    EMBEDDING_MAX_LEN: int = 512

    RERANKER_MODEL_DIR: str = str(PROJECT_ROOT / "models" / "bge-reranker-base-onnx")
    # 기본 fp32. CrossEncoder ranking과 정확히 일치 (int8은 미세한 ordering 흔들림).
    # 디스크 절감을 위해 RERANKER_MODEL_FILENAME=model_int8.onnx로 override 가능.
    RERANKER_MODEL_FILENAME: str = "model.onnx"
    # max_len 128 + 짧은 passage = attention O(N²) 부담 ↓.
    # 1500자 passage(token~500) → 11s, 240자 passage(token~100) → 2.5s,
    # 목표: 100자 passage(token~40) → ~500ms.
    RERANKER_MAX_LEN: int = 128
    RERANKER_TOP_N: int = 20
    RERANKER_BATCH_SIZE: int = 16
    RERANKER_ENABLED: bool = True
    RERANKER_TIMEOUT_SEC: float = 1.5
    RERANKER_PASSAGE_TITLE_CHARS: int = 80
    RERANKER_PASSAGE_DESC_CHARS: int = 120
    RERANKER_PASSAGE_KEYWORDS_CHARS: int = 50

    QU_ENABLED: bool = True
    QU_MODEL_PATH: str = str(
        PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct-gguf" / "qwen2.5-0.5b-instruct-q4_0.gguf"
    )
    QU_N_CTX: int = 2048
    QU_N_THREADS: int = 4
    QU_MAX_TOKENS: int = 200

    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    REDIS_ENABLED: bool = True
    REDIS_CACHE_NAMESPACE: str = "search:emb:"
    REDIS_POPULAR_KEY: str = "search:popular"
    REDIS_POPULAR_TTL: int = 60 * 60 * 24 * 30  # 30 days

    SEARCH_CACHE_SIZE: int = 10000
    SEARCH_CACHE_TTL: int = 3600

    SEARCH_BM25_TOP_N: int = 50
    SEARCH_DENSE_TOP_N: int = 50
    SEARCH_RRF_K: int = 60
    SEARCH_MAX_POOL: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_config():
    return Config()

config = get_config()
