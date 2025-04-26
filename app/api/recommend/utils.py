from sqlalchemy.orm import Session

from app.models.recommendation import Recommendation
from app.models.member_last_seen_feed import MemberLastSeenFeed
from app.models.article import Article

from datetime import datetime, timezone


def get_recently_viewed_article_ids(dynamo: Session, member_id: int):
    table = dynamo.Table('event')
    now = int(datetime.now(timezone.utc).timestamp())
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
    row = db.query(MemberLastSeenFeed).filter(MemberLastSeenFeed.member_id == member_id).first()
    if row:
        return str(row.article_id)
    return None
