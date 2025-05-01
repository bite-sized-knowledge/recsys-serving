from sqlalchemy import Column, BigInteger, ForeignKey, TIMESTAMP
from .base import Base

class MemberInterest(Base):
    __tablename__ = "member_interest"
    member_interest_id = Column(BigInteger, primary_key=True, autoincrement=True)
    member_id = Column(BigInteger, nullable=False)
    interest_id = Column(BigInteger, nullable=False)
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)