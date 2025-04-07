from fastapi import FastAPI
from app.api import health
from app.api import recommend
from .db import Connection

app = FastAPI(
    title="Bites Recommender API",
    description="API for recommending feeds to users based on their interests.",
    version="1.0.0"
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(recommend.router, prefix="/feeds", tags=["feeds"])

# 기본 엔드포인트
@app.get("/")
async def root():
    return {"message": "Bites Recommender API is running!"}