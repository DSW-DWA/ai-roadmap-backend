from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class Resource(BaseModel):
    title: str
    url: Optional[str] = None
    type: Literal["article", "doc", "video", "course", "exercise", "book", "repo"] = "article"


class Milestone(BaseModel):
    id: str = Field(..., description="Стабильный идентификатор")
    title: str
    summary: str
    topics: List[str] = Field(default_factory=list)
    resources: List[Resource] = Field(default_factory=list)
    estimated_hours: int = Field(4, ge=1, le=60)
    tags: List[str] = Field(default_factory=list)


class Roadmap(BaseModel):
    title: str = "SQL Roadmap"
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    total_estimated_hours: int = 0
    milestones: List[Milestone] = Field(default_factory=list)
    notes: Optional[str] = None


class RewriteRequest(BaseModel):
    roadmap: Roadmap
    prompt: str = Field(..., min_length=1, description="Инструкция, как изменить роадмап")
