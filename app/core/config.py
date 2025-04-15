from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

if os.environ.get("ENVIRONMENT") == "dev":
    load_dotenv()

class Settings(BaseSettings):
    # Database settings
    DB_NAME: str = str(os.getenv("DB_NAME"))
    DB_HOST: str = str(os.getenv("DB_HOST"))
    DB_USER: str = str(os.getenv("DB_USER"))
    DB_PASSWORD: str = str(os.getenv("DB_PASSWORD"))
    DB_PORT: int = int(os.getenv("DB_PORT", 3306))

    ENVIRONMENT: str = str(os.getenv("ENVIRONMENT"))

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Instantiate the settings object
settings = Settings()
