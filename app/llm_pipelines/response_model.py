from pydantic import BaseModel, Field



class KnowledgeMap(BaseModel):
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
