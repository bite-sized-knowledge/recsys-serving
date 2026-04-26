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

    SEARCH_CACHE_SIZE: int = 10000
    SEARCH_CACHE_TTL: int = 3600

    SEARCH_BM25_TOP_N: int = 50
    SEARCH_DENSE_TOP_N: int = 50
    SEARCH_RRF_K: int = 60

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_config():
    return Config()

config = get_config()
