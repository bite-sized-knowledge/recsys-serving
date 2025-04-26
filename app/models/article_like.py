from sqlalchemy import Column, BigInteger, CHAR, Boolean, TIMESTAMP
from .base import Base

class ArticleLike(Base):
    __tablename__ = "article_like"
    article_like_id = Column(BigInteger, primary_key=True, autoincrement=True)
    article_id = Column(CHAR(27), nullable=False)
    member_id = Column(BigInteger, nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)