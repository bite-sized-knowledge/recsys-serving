from sqlalchemy import (
    Column,
    BigInteger,
    CHAR,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
    func,
)
from .base import Base


class Recommendation(Base):
    __tablename__ = "recommendation"
    __table_args__ = (
        UniqueConstraint("member_id", "article_id", name="uq_member_article"),
        Index("idx_member_id_recommendation_id", "member_id", "recommendation_id"),
        {"comment": "추천 테이블"},
    )

    recommendation_id = Column(
        BigInteger, primary_key=True, autoincrement=True, comment="PK"
    )
    member_id = Column(
        BigInteger,
        ForeignKey("member.member_id", ondelete="CASCADE"),
        nullable=False,
        comment="추천 대상 멤버 ID",
    )
    article_id = Column(
        CHAR(27, collation="utf8mb4_bin"),
        ForeignKey("article.article_id", ondelete="CASCADE"),
        nullable=False,
        comment="추천된 아티클 ID",
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="추천 생성 시각 (UTC)",
    )