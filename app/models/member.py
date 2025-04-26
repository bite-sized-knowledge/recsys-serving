from sqlalchemy import Column, BigInteger, String, Integer, TIMESTAMP
from .base import Base

class Member(Base):
    __tablename__ = "member"
    member_id = Column(BigInteger, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True)
    password = Column(String(255))
    name = Column(String(255))
    birth = Column(Integer)
    gender = Column(String(10))
    status = Column(String(10))
    role = Column(String(20))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)