from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

if os.environ.get("ENVIRONMENT") == "dev":
    load_dotenv()

class Settings(BaseSettings):
    # Database settings
    DB_NAME: str
    DB_HOST: str
    DB_USER: str
    DB_PASSWORD: str
    DB_PORT: int  

    ENVIRONMENT: str 

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False
    )


# Instantiate the settings object
settings = Settings()
