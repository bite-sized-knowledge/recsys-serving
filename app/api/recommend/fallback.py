from sqlalchemy.orm import Session
from datetime import datetime, timezone
import random
from app.models.recommendation import Recommendation
from .utils import (
    get_today_recommendation_article_ids,
    get_recently_viewed_article_ids,
    get_latest_articles,
    get_interest_category_articles,
    get_diversity_articles,
    get_popular_article_ids,
    filter_existing_article_ids
)
from boto3.resources.base import ServiceResource

def _fill_article_ids(article_ids, candidates, n_want):
    """중복 없이 최대 n_want개만 추가"""
    new_ids = []
 
    for value in candidates:
        if value is None:
            continue
        if hasattr(value, '__getitem__') and not isinstance(value, str):
            value = value[0]
        new_ids.append(value.strip())

    for aid in new_ids:
        if len(article_ids) >= n_want:
            break
        article_ids.add(aid)
    return article_ids

def fallback_recommend_exploration(
        db: Session,
        dynamo: ServiceResource,
        member_id: int,
        limit_total=100
    ) -> None:
    """
    지정된 유저(member_id)에 대해 탐색 기반(fallback) 추천 pool을 생성하여, 
    추천 후보 아티클 100개를 랜덤하게 선정한 뒤 recommendation 테이블에 일괄 삽입하는 함수.

    추천 후보는 다음과 같이 pool별 비율로 분배됨:
        - 최신글 30%
        - 유저의 관심 카테고리 글 30%
        - 다양성(랜덤) 10%
        - 인기글(DynamoDB event 기준) 30%

    주요 제외 기준:
        - 최근 3일간 유저에게 노출된 글
        - 오늘 추천 pool에 이미 포함된 글
        - 중복 아티클 및 keywords가 없는 아티클

    pool별 추출 결과를 합친 뒤, 최종 100개 후보를 무작위로 셔플하여 DB에 저장.

    Args:
        db (Session): SQLAlchemy 세션 (RDBMS)
        dynamo: boto3 DynamoDB 리소스 객체
        member_id (int): 추천할 유저의 ID
        limit_total (int): 최종 추천 pool 크기 (기본 100)

    Returns:
        None
    """

    print("FallBack Logic Executed")

    # 1. 제외 article set
    exclude_ids = set()
    exclude_ids |= get_recently_viewed_article_ids(dynamo, member_id)
    exclude_ids |= get_today_recommendation_article_ids(db, member_id)

    # 2. Pool별 비율 정의 및 준비
    pool_plan = [
        ("latest",    0.3),
        ("interest",  0.3),
        ("diversity", 0.1),
        ("popular",   0.3)
    ]
    counts = [int(limit_total * r) for _, r in pool_plan]
    counts[-1] = limit_total - sum(counts[:-1])  # 마지막 popular pool은 남은 몫 모두

    # 3. Pool별 추출 및 합치기
    article_ids = set()
    cur_n = 0
    for (plan, _), n_pick in zip(pool_plan, counts):
        # pool_func은 (db, exclude_ids, n) → id list
        if plan == "popular":
            candidates = get_popular_article_ids(dynamo, n_pick*2, exclude_ids=exclude_ids)
        elif plan == "latest":
            candidates = get_latest_articles(db, exclude_ids, n_pick*2)
        elif plan == "interest":
            candidates = get_interest_category_articles(db, member_id, exclude_ids, n_pick*2)
        else:
            candidates = get_diversity_articles(db, member_id, exclude_ids, n_pick*2)


        article_ids = _fill_article_ids(article_ids, candidates, cur_n + n_pick)
        cur_n = len(article_ids)
        exclude_ids |= article_ids


    # 4. 셔플 + 최종 100개
    article_ids = list(article_ids)
    article_ids = filter_existing_article_ids(db, article_ids)
    random.shuffle(article_ids)
    article_ids = article_ids[:limit_total]

    # 5. recommendation 테이블에 일괄 insert
    now = datetime.now(timezone.utc)
    db.bulk_save_objects([
        Recommendation(
            member_id=member_id,
            article_id=aid,
            created_at=now,
        ) for aid in article_ids
    ])
    db.commit()

    return