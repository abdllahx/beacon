from datetime import datetime

from pydantic import BaseModel, Field


class RawArticle(BaseModel):
    source: str = Field(description="Origin source: 'gdelt' or 'newsapi'")
    url: str
    title: str | None = None
    content: str | None = None
    language: str | None = None
    published_at: datetime | None = None
