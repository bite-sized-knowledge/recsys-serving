from sqlalchemy.orm import Session
from .schema import RecommendItem
from .utils import get_last_seen_article_id
from .fallback import fallback_recommend_exploration
from boto3.resources.base import ServiceResource

from app.models.recommendation import Recommendation
from app.models.article import Article


def recommend_feeds(db: Session, dynamo: ServiceResource, member_id: int) -> RecommendItem:
    """
    유저에게 추천할 feed 리스트(최대 10개)를 반환한다.
    1. last_seen_article_id를 기준으로 커서 이후의 추천 pool(recommendation)에서 조회
    2. 추천 pool이 비었으면 fallback pool을 먼저 생성 후 재조회
    """

    last_seen_article_id = get_last_seen_article_id(db, member_id)

    # 1. last_seen_rec_id 찾기 (없으면 0)
    if last_seen_article_id is not None:
        rec_row = (
            db.query(Recommendation.recommendation_id)
            .filter(Recommendation.member_id == member_id)
            .filter(Recommendation.article_id == last_seen_article_id)
            .first()
        )
        last_seen_rec_id = rec_row.recommendation_id if rec_row else 0
    else:
        last_seen_rec_id = 0
        # 커서가 없으면 recommendation pool이 비었을 확률 높음, 바로 fallback
        fallback_recommend_exploration(db, dynamo, member_id)

    # 2. 추천 pool에서 article_id 10개 추출 (커서 이후)
    q = (
        db.query(Recommendation.article_id)
        .join(Article, Recommendation.article_id == Article.article_id)
        .filter(Recommendation.member_id == member_id)
        .filter(Recommendation.recommendation_id >= last_seen_rec_id)
        .order_by(Recommendation.recommendation_id.asc())
    )
    articles = [row[0] for row in q.limit(10).all()]

    # 3. 추천 pool이 모두 소진됐다면 fallback 다시 생성 후 재조회
    if not articles or len(articles) < 10:
        fallback_recommend_exploration(db, dynamo, member_id)
        # fallback 후 다시 쿼리
        q = (
            db.query(Recommendation.article_id)
            .join(Article, Recommendation.article_id == Article.article_id)
            .filter(Recommendation.member_id == member_id)
            .filter(Recommendation.recommendation_id > last_seen_rec_id)
            .order_by(Recommendation.recommendation_id.asc())
        )
        articles = [row[0] for row in q.limit(10).all()]

    return RecommendItem(articles=articles)