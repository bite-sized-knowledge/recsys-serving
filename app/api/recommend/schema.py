from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class RecommendItem(BaseModel):
    """응답 contract — bite-api 가 소비.

    feed_request_id 는 옵션 (구버전 호환). bite-api 가 echo 해 bite-web 의
    다음 user_events.feed_request_id 로 흘러가서 impression ↔ click 정확 그룹핑.
    """
    articles: List[str]
    feed_request_id: Optional[str] = None


class FeedbackEvent(BaseModel):
    """POST /feeds/feedback — bite-api fire-and-forget.

    회원이면 member_id, 비회원이면 device_id. 둘 다 있으면 member_id 우선.
    """
    member_id: Optional[int] = None
    device_id: Optional[str] = None
    article_id: str
    event_type: str = Field(..., description="article_in / like / archive / share / uninterest 등")

    @model_validator(mode="after")
    def _require_identifier(self):
        if self.member_id is None and not self.device_id:
            raise ValueError("member_id or device_id is required")
        return self


class FeedbackAck(BaseModel):
    accepted: bool
    bandit_updated: bool = False
    user_vector_updated: bool = False
    reason: Optional[str] = None
