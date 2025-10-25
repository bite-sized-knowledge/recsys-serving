from typing import List
from pydantic import BaseModel


class SearchResponse(BaseModel):
    articles: List[str]
