from fastapi import APIRouter, Depends 
from sqlalchemy.orm.session import Session
from sqlalchemy import text
from app.db import get_db 
from .schema import RecommendItem

router = APIRouter()

@router.get("", response_model=RecommendItem, tags=["feeds"])
async def recommend_feeds(
    member_id: int,
    db: Session = Depends(get_db),
):
    result = db.execute(
        text(f"""
        SELECT 
            article_id 
        FROM 
            article
        ORDER BY RAND()
        LIMIT 10;
        """)
    ).fetchall()
    articles = [row[0] for row in result]

    return RecommendItem(articles=articles)
