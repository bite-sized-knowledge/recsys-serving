from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm.session import Session

from app.core.auth import verify_api_key
from app.db import get_db
from app.services import bandit

from .feedback import handle_feedback
from .schema import FeedbackAck, FeedbackEvent, RecommendItem
from .service import recommend_feeds, recommend_feeds_anonymous


router = APIRouter()

HEADER_INTEREST_IDS = "X-Interest-Ids"


def _parse_interest_ids(raw: Optional[str]) -> Optional[List[int]]:
    """X-Interest-Ids CSV → [int]. invalid 항목은 무시. 결과 비면 None."""
    if not raw:
        return None
    out: List[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            continue
    return out or None


@router.get("", response_model=RecommendItem, tags=["feeds"], status_code=status.HTTP_200_OK)
async def get_recommend_feeds(
    member_id: int | None = None,
    device_id: str | None = None,
    lang: str | None = None,
    x_interest_ids: str | None = Header(default=None, alias=HEADER_INTEREST_IDS),
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    if member_id is None and not device_id:
        raise HTTPException(status_code=400, detail="member_id or device_id is required")
    if member_id is None:
        return recommend_feeds_anonymous(
            db, device_id, lang=lang, interest_ids=_parse_interest_ids(x_interest_ids)
        )
    return recommend_feeds(db, member_id, lang=lang)


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


class MigrateDeviceRequest(BaseModel):
    member_id: int = Field(..., gt=0)
    device_id: str = Field(..., min_length=1)


class MigrateDeviceAck(BaseModel):
    migrated: int


@router.post(
    "/migrate-device",
    response_model=MigrateDeviceAck,
    tags=["feeds"],
    status_code=status.HTTP_200_OK,
)
async def post_migrate_device(
    payload: MigrateDeviceRequest,
    db: Session = Depends(get_db),
    _api_key: str = Depends(verify_api_key),
):
    """비회원 device 가 lazy guest 로 회원 전환된 직후 bandit state 이관.

    bite-api LazyGuest middleware 가 발급 직후 fire-and-forget 호출.
    멱등 (재호출 시 device 값으로 덮어씀, impressions/clicks 는 누적).
    """
    n = bandit.migrate_device_to_member(db, payload.device_id, payload.member_id)
    return MigrateDeviceAck(migrated=n)
