from pydantic import BaseModel, Field
from typing import List

class SearchRequest(BaseModel):
    query: str = Field(..., description="User query")
    k: int = Field(10, ge=1, le=100, description="Top-K to return")
    rerank: bool = Field(False, description="Enable cross-encoder reranking (if configured)")

class SearchHit(BaseModel):
    id: int
    score: float
    content: str
    recipe_source: str

class SearchResponse(BaseModel):
    query: str
    k: int
    hits: List[SearchHit]
    took_ms: float
    index_size: int
    model: str