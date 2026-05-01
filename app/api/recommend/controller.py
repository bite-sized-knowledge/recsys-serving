from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm.session import Session

from app.core.auth import verify_api_key
from app.db import get_db

from .feedback import handle_feedback
from .schema import FeedbackAck, FeedbackEvent, RecommendItem
from .service import recommend_feeds, recommend_feeds_anonymous

router = APIRouter()


@router.get("", response_model=RecommendItem, tags=["feeds"], status_code=status.HTTP_200_OK)
async def get_recommend_feeds(
    member_id: int | None = None,
    device_id: str | None = None,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    if member_id is None and not device_id:
        raise HTTPException(status_code=400, detail="member_id or device_id is required")
    if member_id is None:
        # device_id-only — phase 1+2 personalization 입력이 없어 글로벌 풀 fallback.
        return recommend_feeds_anonymous(db)
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
