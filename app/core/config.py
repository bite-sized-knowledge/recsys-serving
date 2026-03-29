from pydantic_settings import BaseSettings
from functools import lru_cache

class Config(BaseSettings):
    ENVIRONMENT: str = "prod"
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 3306
    DB_USER: str = "bite-dev"
    DB_PASSWORD: str = "qkdlqm!"
    DB_NAME: str = "bite"

    QDRANT_URL: str = "http://127.0.0.1:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "bite-vectordb"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_config():
    return Config()

config = get_config()
