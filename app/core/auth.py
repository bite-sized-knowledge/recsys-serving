from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from app.core.config import config

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Depends(_api_key_header)) -> str:
    if not config.RECSYS_API_KEY:
        return ""  # auth disabled when key is not configured
    if not api_key or api_key != config.RECSYS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key
