# app/api/__init__.py
from fastapi import APIRouter
from app.api.health import router as health_router
from .recommend import controller as recommend
from .search import controller as search

api = APIRouter()

api.include_router(health_router, prefix="/health", tags=["Health"])
api.include_router(recommend.router, prefix="/feeds", tags=["Feeds"])
api.include_router(search.router, prefix="/search", tags=["Search"])
