from sqlalchemy import Column, BigInteger, CHAR, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.sql import func
from .base import Base

class Recommendation(Base):
    __tablename__ = "recommendation"
    __table_args__ = (
        UniqueConstraint("member_id", "article_id", name="uq_member_article"),
        Index("idx_member_id_recommendation_id", "member_id", "recommendation_id"),
    )

    recommendation_id = Column(BigInteger, primary_key=True, autoincrement=True)
    member_id = Column(BigInteger, ForeignKey("member.member_id"), nullable=False)
    article_id = Column(CHAR(27), ForeignKey("article.article_id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Recommendation(id={self.recommendation_id}, member_id={self.member_id}, article_id='{self.article_id}')>"