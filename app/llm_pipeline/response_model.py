from inspect import cleandoc

from pydantic import BaseModel, Field


class Concept(BaseModel):
    """A Concept is a distinct unit of knowledge that represents an idea, skill, or topic which can be taught, learned, or connected to other concepts."""

    name: str = Field(
        description=cleandoc("""
            Every Concept should have a name so it can be easily identified and referenced
        """)
    )
    consist_of: list['Concept'] | None = Field(
        default=None,
        description=cleandoc("""
            Concept may stand alone or be composed of sub-concepts, forming part of a larger knowledge structure.
        """),
    )


class KnowledgeMap(BaseModel):
    """
    A KnowledgeMap is a structured representation of Concepts and the relationships between them.

    It models both hierarchical and relational dependencies among Concepts.
    """

    concepts: list[Concept] = Field(
        description="""The set of all Concepts included in this KnowledgeMap"""
    )

    prerequisites: dict[str, str] = Field(
        description=cleandoc("""
            A mapping of prerequisite relationships between Concepts.
            Each key is the name of a Concept that requires prior knowledge,
            and each value is the name of the Concept that must be known first.
            This defines the partial order of learning dependencies.
            These relationships are directed.
        """)
    )

    related_concepts: dict[str, str] = Field(
        description=cleandoc("""
            A mapping of non-hierarchical relationships between Concepts.
            Each key and value represent two Concept names that are related,
            but not in a prerequisite order.
            This models lateral associations such as similarity, complementarity,
            or thematic linkage.
            These relationships are undirected.
        """)
    )
