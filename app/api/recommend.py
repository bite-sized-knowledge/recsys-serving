from fastapi import APIRouter, Body, Depends 
from sqlalchemy.orm.session import Session
from sqlalchemy import text
from app.db import get_db 

router = APIRouter()

@router.get("", tags=["feeds"])
async def recommend_feeds(
    db: Session = Depends(get_db),
):
    result = db.execute(
        text(f"""
        SELECT count(*) FROM article;
        """)
    ).fetchone()[0]


    return {
        "articles" : [
            result
        ]
    }