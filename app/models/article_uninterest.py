from sqlalchemy import Column, BigInteger, CHAR, TIMESTAMP
from .base import Base

class ArticleUninterest(Base):
    __tablename__ = "article_uninterest"
    article_uninterest_id = Column(BigInteger, primary_key=True, autoincrement=True)
    article_id = Column(CHAR(27), nullable=False)
    member_id = Column(BigInteger, nullable=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)