from fastapi import APIRouter, Depends, status
from sqlalchemy.orm.session import Session
from app.db import get_db, get_dynamo
from .schema import RecommendItem
from .service import recommend_feeds  


router = APIRouter()

@router.get("", response_model=RecommendItem, tags=["feeds"], status_code=status.HTTP_200_OK)
async def get_recommend_feeds(
    member_id: int,
    db: Session = Depends(get_db),
    dynamo=Depends(get_dynamo)
):
    return recommend_feeds(db, dynamo, member_id)