from fastapi import APIRouter, Depends 
from sqlalchemy.orm.session import Session
from sqlalchemy import text
from app.db import get_db 

router = APIRouter()

@router.get("", tags=["feeds"])
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
        ORDER BY RAND({member_id})
        LIMIT 10;
        """)
    ).fetchall()


    return {
        "articles" : [
            row[0] for row in result
        ]
    }