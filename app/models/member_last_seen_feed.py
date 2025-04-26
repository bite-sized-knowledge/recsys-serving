from sqlalchemy import Column, BigInteger, CHAR, TIMESTAMP
from .base import Base

class MemberLastSeenFeed(Base):
    __tablename__ = "member_last_seen_feed"
    member_id = Column(BigInteger, primary_key=True)
    article_id = Column(CHAR(27), nullable=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)