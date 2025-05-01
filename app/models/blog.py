from sqlalchemy import Column, BigInteger, String, TIMESTAMP

from .base import Base

class Blog(Base):
    __tablename__ = "blog"
    blog_id = Column(BigInteger, primary_key=True, autoincrement=True)
    platform_id = Column(BigInteger)
    title = Column(String(255))
    url = Column(String(500))
    rss_url = Column(String(500), nullable=False)
    favicon = Column(String(255))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)