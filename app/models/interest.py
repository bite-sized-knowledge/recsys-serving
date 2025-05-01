from sqlalchemy import Column, BigInteger, String, TIMESTAMP
from .base import Base

class Interest(Base):
    __tablename__ = "interest"
    interest_id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False)
    image = Column(String(255))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)