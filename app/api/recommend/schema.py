from pydantic import BaseModel
from typing import List

class RecommendItem(BaseModel):
    articles : List[str]
