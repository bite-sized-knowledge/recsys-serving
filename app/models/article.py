from sqlalchemy import Column, BigInteger, String, CHAR, Text, TIMESTAMP

from .base import Base

class Article(Base):
    __tablename__ = "article"
    article_id = Column(CHAR(27), primary_key=True)
    blog_id = Column(BigInteger)
    url = Column(String(500))
    title = Column(String(255))
    thumbnail = Column(String(500))
    description = Column(String(1000))
    keywords = Column(String(255))
    category_id = Column(BigInteger)
    content = Column(Text)
    content_length = Column(BigInteger)
    lang = Column(String(10))
    like_count = Column(BigInteger, nullable=False, default=0)
    share_count = Column(BigInteger, nullable=False, default=0)
    bookmark_count = Column(BigInteger, nullable=False, default=0)
    published_at = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)