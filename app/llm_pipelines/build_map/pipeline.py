from inspect import cleandoc

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from app.llm_pipelines.build_map.prompts import (
    step_one_hierarchy_prompt,
    step_three_related_concepts_prompt,
)

type ConceptsHierarchyNode = dict[str, list[ConceptsHierarchyNode | str]]

class ConceptsHierarchy(BaseModel):
    """Concepts organized into hierarchical structure"""

    hierarchy: ConceptsHierarchyNode = Field(
        description=cleandoc("""
            A mapping of hierarchical relationships between concepts.  
            Each key is the name of a concept and each value is a list of sub-concepts.  

            This is recursive:  
            - A concept can be composed of sub-concepts (objects), forming part of a larger knowledge structure.  
            - Or it may stand alone (base case, string).
            - If concepts is leaf node, write it as string, do not add empty list of childs
        """)
    )

class ConceptPrerequisites(BaseModel):
    """Concepts with their corresponding prerequisites"""

    prerequisites: dict[str, list[str]] = Field(
        description=cleandoc("""
            A mapping of prerequisites for each concept, where each key represents a concept and its values represent the concepts that must be understood first.
            If two concepts are already in a hierarchical relationship, they should not be connected as prerequisites.
        """)
    )

class ConceptSources(BaseModel):
    link: str = Field(
        description=cleandoc("Link in the format file-path#Header-1/Header-2"),
        examples=["docs/tutorial.md#Quick-start/Setup/Requirements"]
    )

class RelatedConcepts(BaseModel):
    """Concepts which are related to each other"""
    
    related: dict[str, list[str]] = Field(
        description=cleandoc("""
            A mapping, where each key is a concept and value is concepts related to it.
            This is used to mark cross-hierarchical relationship.
        """)
    )


def _flatten_hierarchy(node: dict|list|str):
    match node:
        case list():
            for item in node:
                yield from _flatten_hierarchy(item)
        case dict():
            for key, value in node.items():
                yield key
                yield from _flatten_hierarchy(value)
        case str():
            yield node

#def _get_children(node: dict|list|str):
#    match node:
#        case list():
#        case dict():
#            return {node.:
#        case str():
#            return {node: []}
#
#def get_strict_prerequisites_schema(allowed_pairs: dict[str, set[str]]):
#    pass


class BuildMapPipeline:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
    ):
        self._client: AsyncOpenAI = client
        self._model: str = model

    async def _build_hierarchy(
        self,
        material: dict[str, str],
        language: str,
    ) -> tuple[list[ChatCompletionMessageParam], ConceptsHierarchy]:
        messages = step_one_hierarchy_prompt(
            material=material, language=language,
            response_model=ConceptsHierarchy,
        )

        response = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=ConceptsHierarchy,
            temperature=0.0,
            seed=42
        )

        message = response.choices[0].message
        assert message.parsed
        assert message.content
        
        messages += [{'role': 'assistant', 'content': message.content}]

        return (
            messages, # pyright: ignore[reportReturnType]
            message.parsed
        )

    async def _link_related(
        self,
        messages: list[ChatCompletionMessageParam]
    ) -> tuple[list[ChatCompletionMessageParam], ConceptPrerequisites]:
        messages += step_three_related_concepts_prompt(json_schema=RelatedConcepts.model_json_schema())

        response = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=RelatedConcepts,
            temperature=0.0,
            seed=42
        )

        message = response.choices[0].message
        assert message.parsed
        assert message.content
        
        messages += [{'role': 'assistant', 'content': message.content}]

        return (
            messages, # pyright: ignore[reportReturnType]
            message.parsed
        )
        ...

    async def build(
        self, material: dict[str, str],
        language: str = 'ru'
    ):
        messages, hierarchy = await self._build_hierarchy(material, language=language)
        
        flattened = set(_flatten_hierarchy(hierarchy.hierarchy))
        # parents = 
        allowed_prerequisite_pairs = {
            key: flattened - {key} for key in flattened
        }


        _, prerequisites = await self._link_related(messages) 
        return hierarchy, prerequisites

# async def build_map(
#     client: AsyncOpenAI, model: str, material: dict[str, str], language: str = 'ru'
# ) -> KnowledgeMap:
#     """Build knowledge map from list of educational materials using OpenAI model"""
# 
