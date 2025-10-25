from pydantic_settings import BaseSettings
from functools import lru_cache

class Config(BaseSettings):
    ENVIRONMENT: str
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    DYNAMODB_REGION: str
    DYNAMODB_ENDPOINT_URL: str

    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_NAME: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache
def get_config():
    return Config()

config = get_config()