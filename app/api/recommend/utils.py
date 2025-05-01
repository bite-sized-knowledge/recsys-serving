from sqlalchemy.orm import Session
from app.models.member_last_seen_feed import MemberLastSeenFeed
from app.models.recommendation import Recommendation
from app.models.article import Article
from app.models.member_interest import MemberInterest
from datetime import datetime
from sqlalchemy import func
import collections
from boto3.dynamodb.conditions import Key
from boto3.resources.base import ServiceResource


def get_today_recommendation_article_ids(db: Session, member_id: int):
    # 오늘 추천 pool에 있었던 article_id
    today = datetime.now().date()
    ids = db.query(Recommendation.article_id)\
        .filter(
            Recommendation.member_id == member_id,
            func.date(Recommendation.created_at) == today
        ).all()
    return {row[0] for row in ids}

def get_latest_articles(db: Session, exclude_ids, limit):
    # 최신글 가져오기
    return db.query(Article.article_id)\
        .filter(
            ~Article.article_id.in_(exclude_ids),
            Article.keywords.isnot(None),
            Article.category_id.isnot(None),
            Article.published_at.isnot(None)
        )\
        .order_by(Article.published_at.desc())\
        .limit(limit).all()

def get_user_interest_categories(db: Session, member_id: int):
    rows = db.query(MemberInterest.interest_id)\
        .filter(MemberInterest.member_id == member_id)\
        .all()

    return [row[0] for row in rows]

def get_interest_category_articles(db: Session, member_id: int, exclude_ids, limit):
    # 유저의 관심 카테고리 id를 구하는 함수 필요
    user_category_ids = get_user_interest_categories(db, member_id)
    EXCLUDE_CATEGORY_ID = 14

    return db.query(Article.article_id)\
        .filter(
            Article.category_id.in_(user_category_ids),
            ~Article.article_id.in_(exclude_ids),
            Article.keywords.isnot(None),
            Article.category_id != EXCLUDE_CATEGORY_ID,
        )\
        .order_by(func.rand())\
        .limit(limit).all()

def get_diversity_articles(db: Session, member_id:int, exclude_ids, limit):
    user_category_ids = get_user_interest_categories(db, member_id)
    # 다양성을 위해 keyword/카테고리/블로그 기준 랜덤 추출

    return db.query(Article.article_id)\
        .filter(
            ~Article.article_id.in_(exclude_ids),
            ~Article.category_id.in_(user_category_ids),
            Article.keywords.isnot(None)
        )\
        .order_by(func.rand())\
        .limit(limit).all()

def get_popular_article_ids(dynamo: ServiceResource, limit:int=30, days:int=3, exclude_ids=None):
    """
    DynamoDB GSI(Query) 기반 최근 3일 내 인기글 집계 (event_type in [article_in, like, archive])
    :param dynamo: boto3 DynamoDB resource
    :param limit: 반환할 인기글 개수
    :param days: 최근 N일 기준
    :param exclude_ids: 제외할 article_id set
    :return: 인기글 article_id 리스트 (내림차순)
    """
    if exclude_ids is None:
        exclude_ids = set()
    table = dynamo.Table('event')
    now = int(datetime.now().timestamp())
    since = now - days * 24 * 60 * 60

    event_types = ["article_in", "like", "archive"]
    counter = collections.Counter()

    for etype in event_types:
        # 첫 페이지 쿼리
        response = table.query(
            IndexName='event_type-timestamp-index',
            KeyConditionExpression=Key('event_type').eq(etype) & Key('timestamp').gte(since),
            ProjectionExpression='target_id'
        )

        for item in response['Items']:
            aid = item.get("target_id")
            if aid and aid not in exclude_ids:
                counter[aid] += 1

        # 페이징 처리 (1MB 이상 결과일 경우)
        while 'LastEvaluatedKey' in response:
            response = table.query(
                IndexName='event_type-timestamp-index',
                KeyConditionExpression=Key('event_type').eq(etype) & Key('timestamp').gte(since),
                ProjectionExpression='target_id',
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            for item in response['Items']:
                aid = item.get("target_id")
                if aid and aid not in exclude_ids:
                    counter[aid] += 1

    ranked = [aid for aid, _ in counter.most_common()][:limit]
    return ranked

def get_recently_viewed_article_ids(dynamo: ServiceResource, member_id: int) -> set:
    # 최근 3일동안 노출됐던 articles
    table = dynamo.Table('event')
    now = int(datetime.now().timestamp())
    three_days_ago = now - 3 * 24 * 60 * 60

    response = table.query(
        KeyConditionExpression="member_id = :m AND #ts >= :start_ts",
        FilterExpression="event_type = :e",
        ExpressionAttributeNames={"#ts": "timestamp"},
        ExpressionAttributeValues={
            ":m": member_id,
            ":start_ts": three_days_ago,
            ":e": "f_imp",
        }
    )
    return set(item["target_id"] for item in response["Items"])

def get_last_seen_article_id(db: Session, member_id: int) -> str:
    row = db.query(MemberLastSeenFeed)\
        .filter(MemberLastSeenFeed.member_id == member_id)\
        .first()

    if row:
        return str(row.article_id)
    return None

def filter_existing_article_ids(db: Session, article_ids):
    """
    주어진 article_ids 중 실제 article 테이블에 존재하는 것만 리턴 (ORM 방식).
    """
    if not article_ids:
        return []
    rows = db.query(Article.article_id)\
        .filter(Article.article_id.in_(article_ids))\
        .all()
    # [Row('abc',), ...] 형태이므로
    return [row[0] for row in rows]