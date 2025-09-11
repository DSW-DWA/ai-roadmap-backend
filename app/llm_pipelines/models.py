from pydantic import BaseModel


class Concept(BaseModel):
    """
    A single concept in the knowledge map.

    Represents an individual learning concept with its metadata including
    relationships to other concepts and source attribution.
    """

    title: str
    description: str | None
    related: list[str] | None
    source: list[str] | None

    consist_of: list['Concept'] | None


class KnowledgeMap(BaseModel):
    """
    Complete knowledge map containing all concepts and their relationships.

    The root container for a structured representation of educational content
    organized into hierarchical concepts with cross-references and source links.
    """

    concepts: list[Concept]