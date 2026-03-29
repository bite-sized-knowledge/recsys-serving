from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.article import Article

from .schema import RecommendItem


def recommend_feeds(db: Session, member_id: int) -> RecommendItem:
    del member_id

    articles = (
        db.query(Article.article_id)
        .filter(Article.article_id.isnot(None))
        .order_by(func.rand())
        .limit(10)
        .all()
    )

    return RecommendItem(articles=[row[0] for row in articles])
