from typing import List, Optional

from pydantic import BaseModel, Field


class RecommendItem(BaseModel):
    """응답 contract — bite-api 가 소비. 변경 시 cross-service 영향."""
    articles: List[str]


class FeedbackEvent(BaseModel):
    """POST /feeds/feedback 요청.

    bite-api 가 user_events insert 후 fire-and-forget 으로 호출.
    실패해도 batch reconcile (recommender) 가 다음 cycle 에 정정.
    """
    member_id: int
    article_id: str
    event_type: str = Field(..., description="article_in / like / archive / share / uninterest 등")


class FeedbackAck(BaseModel):
    accepted: bool
    bandit_updated: bool = False
    user_vector_updated: bool = False
    reason: Optional[str] = None
