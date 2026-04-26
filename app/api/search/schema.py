from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class SearchResponse(BaseModel):
    articles: List[str]
    next: Optional[str] = None
