from sqlalchemy.orm import Session
from .schema import RecommendItem
from .utils import get_last_seen_article_id

from app.models.recommendation import Recommendation
from app.models.article import Article


def recommend_feeds(db: Session, dynamo, member_id: int) -> RecommendItem:
    last_seen_article_id = get_last_seen_article_id(db, member_id)

    # recommendation_id 커서 값 찾기 (없으면 0)
    last_seen_rec_id = 0
    if last_seen_article_id:
        rec_row = db.query(Recommendation.recommendation_id)\
            .filter(Recommendation.member_id == member_id)\
            .filter(Recommendation.article_id == last_seen_article_id)\
            .first()
        last_seen_rec_id = rec_row.recommendation_id if rec_row else 0

    # 추천 쿼리
    q = (
        db.query(Recommendation.article_id)\
        .join(Article, Recommendation.article_id == Article.article_id)\
        .filter(Recommendation.member_id == member_id)\
        .filter(Recommendation.recommendation_id > last_seen_rec_id)\
    )

    articles = [row[0] for row in q.order_by(Recommendation.recommendation_id.asc()).limit(10).all()]

    return RecommendItem(articles=articles)