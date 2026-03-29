from typing import List

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.models.article import Article


async def search_articles(
    db: Session,
    query: str,
    limit: int = 20,
) -> List[str]:
    if not query:
        raise ValueError("검색어를 제공해야 합니다.")

    normalized = query.strip()
    results = (
        db.query(Article.article_id)
        .filter(
            or_(
                Article.title.ilike(f"%{normalized}%"),
                Article.description.ilike(f"%{normalized}%"),
            )
        )
        .order_by(Article.published_at.desc().nullslast(), Article.created_at.desc().nullslast())
        .limit(limit)
        .all()
    )

    if not results:
        results = (
            db.query(Article.article_id)
            .filter(Article.article_id.isnot(None))
            .order_by(func.rand())
            .limit(limit)
            .all()
        )

    return [str(result[0]) for result in results if result[0]]
