from pydantic import BaseModel

from app.llm_pipelines.models import KnowledgeMap


class RewriteRequest(BaseModel):
    knowledge_map: KnowledgeMap
    user_query: str
