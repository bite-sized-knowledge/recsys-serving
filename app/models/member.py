from sqlalchemy import (
    Column,
    BigInteger,
    String,
    Integer,
    TIMESTAMP,
    UniqueConstraint,
)
from .base import Base

class Member(Base):
    __tablename__ = "member"
    __table_args__ = (
        UniqueConstraint("email", name="uq_member_email"),
        {"comment": "유저 정보"},
    )

    member_id = Column(BigInteger, primary_key=True, autoincrement=True, nullable=False)
    email = Column(String(255), unique=True, nullable=True)
    password = Column(String(255), nullable=True)
    name = Column(String(255), nullable=True)
    birth = Column(Integer, nullable=True)
    gender = Column(String(10), nullable=True)
    status = Column(String(10), nullable=True)
    role = Column(String(20), nullable=True)
    created_at = Column(
        TIMESTAMP, server_default="CURRENT_TIMESTAMP", nullable=True
    )
    updated_at = Column(
        TIMESTAMP,
        server_default="CURRENT_TIMESTAMP",
        server_onupdate="CURRENT_TIMESTAMP",
        nullable=True,
    )