# app/api/__init__.py
from fastapi import APIRouter
from app.api.health import router as health_router
from app.api.recommend import router as recommend_router

api = APIRouter()

api.include_router(health_router, prefix="/health", tags=["Health"])
api.include_router(recommend_router, prefix="/feeds", tags=["Feeds"])