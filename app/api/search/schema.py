from typing import List
from pydantic import BaseModel


class SearchResponse(BaseModel):
    article_ids: List[str]
