from sqlalchemy import Column, BigInteger, CHAR, TIMESTAMP
from .base import Base

class ArticleShare(Base):
    __tablename__ = "article_share"
    article_share_id = Column(BigInteger, primary_key=True, autoincrement=True)
    article_id = Column(CHAR(27), nullable=False)
    member_id = Column(BigInteger, nullable=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)