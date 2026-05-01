from fastapi import APIRouter, Depends, status
from sqlalchemy.orm.session import Session

from app.core.auth import verify_api_key
from app.db import get_db

from .feedback import handle_feedback
from .schema import FeedbackAck, FeedbackEvent, RecommendItem
from .service import recommend_feeds

router = APIRouter()


@router.get("", response_model=RecommendItem, tags=["feeds"], status_code=status.HTTP_200_OK)
async def get_recommend_feeds(
    member_id: int,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    return recommend_feeds(db, member_id)


@router.post(
    "/feedback",
    response_model=FeedbackAck,
    tags=["feeds"],
    status_code=status.HTTP_200_OK,
)
async def post_feedback(
    event: FeedbackEvent,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    return handle_feedback(db, event)
