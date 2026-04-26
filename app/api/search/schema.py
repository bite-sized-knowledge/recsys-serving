from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class SearchResponse(BaseModel):
    articles: List[str]
    next: Optional[str] = None


class UnderstandRequest(BaseModel):
    query: str


class UnderstandResponse(BaseModel):
    search_keywords: Optional[str] = None
    lang: Optional[str] = None
    recency: Optional[str] = None
    category_hint: Optional[str] = None
    intent: Optional[str] = None
    published_after: Optional[float] = None


class SuggestResponse(BaseModel):
    suggestions: List[str]
