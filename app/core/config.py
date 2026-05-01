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
    RERANKER_MODEL_FILENAME: str = "model.onnx"
    # token 길이가 latency를 지배(attention O(N²)). max_len 96 + passage char 축소
    # = 30 pair · 100자 passage(token~40) ≈ 200~300ms 목표. timeout으로 cap.
    RERANKER_MAX_LEN: int = 96
    RERANKER_TOP_N: int = 15
    RERANKER_BATCH_SIZE: int = 32
    RERANKER_ENABLED: bool = True
    RERANKER_TIMEOUT_SEC: float = 1.0
    RERANKER_PASSAGE_TITLE_CHARS: int = 60
    RERANKER_PASSAGE_DESC_CHARS: int = 80
    RERANKER_PASSAGE_KEYWORDS_CHARS: int = 40

    REDIS_URL: str = "redis://127.0.0.1:6379/0"
    REDIS_ENABLED: bool = True
    REDIS_CACHE_NAMESPACE: str = "search:emb:"
    REDIS_POPULAR_KEY: str = "search:popular"
    REDIS_POPULAR_TTL: int = 60 * 60 * 24 * 30  # 30 days

    # Drift monitoring — 색인 측 SentenceTransformer ↔ 쿼리 측 ONNX parity check
    DRIFT_ENABLED: bool = True
    DRIFT_REFERENCE_MODEL_ID: str = "Qwen/Qwen3-Embedding-0.6B"
    DRIFT_INTERVAL_SEC: int = 24 * 60 * 60  # 24h
    DRIFT_INITIAL_DELAY_SEC: int = 5 * 60  # 5min after startup
    DRIFT_COS_THRESHOLD: float = 0.99

    SEARCH_CACHE_SIZE: int = 10000
    SEARCH_CACHE_TTL: int = 3600

    SEARCH_BM25_TOP_N: int = 50
    SEARCH_DENSE_TOP_N: int = 50
    SEARCH_RRF_K: int = 60
    SEARCH_MAX_POOL: int = 100
    # Drop dense neighbors below this cosine score before fusion. Without
    # this, gibberish queries still pull random nearest articles by vector
    # proximity and BM25 partial substring hits — surfacing low-relevance
    # results to users. Tune downward if recall drops on legitimate niche
    # queries.
    SEARCH_DENSE_MIN_SCORE: float = 0.45

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_config():
    return Config()

config = get_config()
