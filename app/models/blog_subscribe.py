from sqlalchemy import Column, BigInteger, Boolean, TIMESTAMP
from .base import Base

class BlogSubscribe(Base):
    __tablename__ = "blog_subscribe"
    blog_subscribe_id = Column(BigInteger, primary_key=True, autoincrement=True)
    blog_id = Column(BigInteger, nullable=False)
    member_id = Column(BigInteger, nullable=False)
    is_deleted = Column(Boolean, nullable=False, default=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)