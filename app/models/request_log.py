from sqlalchemy import BigInteger, Column, DateTime, Integer, SmallInteger, String
from sqlalchemy.sql import func

from app.models.base import Base


class RecsysRequestLog(Base):
    __tablename__ = "recsys_request_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    request_id = Column(String(36), nullable=False, unique=True)
    endpoint = Column(String(32), nullable=False)
    member_id = Column(BigInteger, nullable=True)
    status_code = Column(SmallInteger, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    result_count = Column(Integer, nullable=True)
    query_text = Column(String(255), nullable=True)
    source = Column(String(50), nullable=True)
    error_class = Column(String(64), nullable=True)
    occurred_at = Column(
        DateTime(timezone=False),
        server_default=func.current_timestamp(3),
    )
