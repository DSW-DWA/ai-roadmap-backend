from inspect import cleandoc
from pydantic import BaseModel, Field

type Concept = dict[str, list[Concept | str]]


class KnowledgeMap(BaseModel):
    """
    A KnowledgeMap is a structured representation of Concepts and the relationships between them.

    It models both hierarchical and relational dependencies among Concepts.

    A Concept is a distinct unit of knowledge that represents an idea, skill, or topic which can be taught, learned, or connected to other concepts.
    Every Concept must have an unique name so it can be clearly identified and referenced.
    """

    hierarchy: list[Concept] = Field(
        description=cleandoc("""
            A mapping of hierarchical relationships between Concepts.
            Each key is the name of the concept
            and each value is sub-concept that forms part of the current concept.
            This is recursive:
                - Concept can be composed of sub-concepts (object), forming part of a larger knowledge structure.
                - or may stand alone (base case, string)

            Note: All concepts existing in knowledge map must be present here.
        """)
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
