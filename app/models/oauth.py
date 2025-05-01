from sqlalchemy import Column, BigInteger, String, TIMESTAMP

from .base import Base

class Oauth(Base):
    __tablename__ = "oauth"
    oauth_id = Column(BigInteger, primary_key=True, autoincrement=True)
    member_id = Column(BigInteger, nullable=False)
    provider = Column(String(10), nullable=False)
    provider_member_id = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)