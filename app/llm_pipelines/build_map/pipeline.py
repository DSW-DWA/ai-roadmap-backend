import re
from collections.abc import Iterable
from inspect import cleandoc
from itertools import chain

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from app.llm_pipelines.build_map.prompts import (
    step_one_hierarchy_prompt,
    step_three_related_concepts_prompt,
    step_two_add_sources,
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
    sources: dict[str, str] = Field(
        description=cleandoc("""
            A mapping between concepts and sources, where key is unique concept name,
            and value is a link in the format file-path#Header-1/Header-2
        """),
    )


def get_markdown_header_paths(markdown_text: str) -> list[str]:
    """
    Parses a markdown string to find all header lines and returns a list of slash-separated paths for the leaf headers.

    A header is considered a "leaf" if it is not immediately followed by a
    header of a deeper level.
    """

    header_pattern = re.compile(r'^(#{1,6})\s+(.*)', re.MULTILINE)
    matches = header_pattern.finditer(markdown_text)

    headers = []
    for match in matches:
        level = len(match.group(1))
        text = match.group(2).strip().replace(' ', '-')
        headers.append({'level': level, 'text': text})

    if not headers:
        return []

    result_paths = []
    current_path = []

    for i, header in enumerate(headers):
        level = header['level']
        text = header['text']

        while len(current_path) >= level:
            current_path.pop()
        current_path.append(text)

        is_leaf = False
        is_last_header = i == len(headers) - 1

        if is_last_header:
            is_leaf = True
        else:
            next_header_level = headers[i + 1]['level']
            if next_header_level <= level:
                is_leaf = True

        if is_leaf:
            result_paths.append('/'.join(current_path))

    return result_paths


def get_allowed_sources(material):
    values = list(material.keys())
    for name, content in material.items():
        for header_path in get_markdown_header_paths(content):
            values.append(f'{name}#{header_path}')

    return values


class RelatedConcepts(BaseModel):
    """Concepts which are related to each other"""

    related: dict[str, list[str]] = Field(
        description=cleandoc("""
            A mapping, where each key is a concept and value is concepts related to it.
            This is used to mark cross-hierarchical relationship.
        """)
    )


def flatten_hierarchy(node: dict | list | str):
    match node:
        case list():
            for item in node:
                yield from flatten_hierarchy(item)
        case dict():
            for key, value in node.items():
                yield key
                yield from flatten_hierarchy(value)
        case str():
            yield node


def get_children(node: dict | str, acc=None):
    acc = acc or {}
    match node:
        case str():
            return acc, [node]

        case dict():
            children = []
            for k, v in node.items():
                children.append(k)
                acc[k] = list(chain.from_iterable(get_children(item, acc)[1] for item in v))
                children.extend(acc[k])

            return acc, children


def get_parents(node: dict | str, path=None, acc=None):
    path = path or []
    acc = acc or {}
    match node:
        case str():
            acc[node] = path
            return acc
        case dict():
            for k, v in node.items():
                acc[k] = path
                for item in v:
                    get_parents(item, path + [k], acc)
            return acc


def clean_related_concepts(hierarchy: ConceptsHierarchy, related: RelatedConcepts):
    existing = set(flatten_hierarchy(hierarchy.hierarchy))
    children, _ = get_children(hierarchy.hierarchy)
    parents = get_parents(hierarchy.hierarchy)

    for key, values in list(related.related.items()):
        new_values = filter(
            lambda v: v in existing,
            set(values or []) - set(children.get(key, [])) - set(parents.get(key, [])),
        )
        new_values = list(new_values) or None
        if not new_values:
            related.related.pop(key)
        else:
            related.related[key] = new_values


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
            material=material,
            language=language,
            response_model=ConceptsHierarchy,
        )

        response = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=ConceptsHierarchy,
            temperature=0.0,
            seed=42,
        )

        message = response.choices[0].message
        assert message.parsed
        assert message.content

        messages += [{'role': 'assistant', 'content': message.content}]

        return (
            messages,  # pyright: ignore[reportReturnType]
            message.parsed,
        )

    async def _add_sources(
        self,
        messages: list[ChatCompletionMessageParam],
        concepts: Iterable[str],
        material: dict[str, str],
    ) -> tuple[list[ChatCompletionMessageParam], ConceptSources]:
        messages += step_two_add_sources(response_model=ConceptSources)

        response = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=ConceptSources,
            temperature=0.0,
            seed=42,
        )

        message = response.choices[0].message
        assert message.parsed
        assert message.content

        sources = message.parsed
        allowed_sources = get_allowed_sources(material)
        for concept in list(sources.sources):
            if concept not in concepts or sources.sources[concept] not in allowed_sources:
                sources.sources.pop(concept)

        messages += [{'role': 'assistant', 'content': sources.model_dump_json()}]

        return (
            messages,  # pyright: ignore[reportReturnType]
            message.parsed,
        )

    async def _link_related(
        self, hierarchy: ConceptsHierarchy, messages: list[ChatCompletionMessageParam]
    ) -> tuple[list[ChatCompletionMessageParam], RelatedConcepts]:
        messages += step_three_related_concepts_prompt(response_model=RelatedConcepts)

        response = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=RelatedConcepts,
            temperature=0.0,
            seed=42,
        )

        message = response.choices[0].message
        assert message.parsed
        assert message.content
        related = message.parsed
        clean_related_concepts(hierarchy, related)

        messages += [{'role': 'assistant', 'content': related.model_dump_json()}]

        return (
            messages,  # pyright: ignore[reportReturnType]
            message.parsed,
        )

    async def build(self, material: dict[str, str], language: str = 'ru'):
        messages, hierarchy = await self._build_hierarchy(material, language=language)
        concepts = set(flatten_hierarchy(hierarchy.hierarchy))
        messages, sources = await self._add_sources(messages, concepts, material)
        messages, related = await self._link_related(hierarchy, messages)

        def build_concepts(node: str | dict) -> list[Concept]:
            match node:
                case str():
                    return [
                        Concept(
                            title=node,
                            description=None,
                            related=related.related.get(node),
                            source=sources.sources.get(node),
                            consist_of=None,
                        )
                    ]
                case dict():
                    return [
                        Concept(
                            title=k,
                            description=None,
                            related=related.related.get(k),
                            source=sources.sources.get(k),
                            consist_of=list(
                                chain.from_iterable(build_concepts(v) for v in node[k])
                            ),
                        )
                        for k in node
                    ]

        return KnowledgeMap(concepts=build_concepts(hierarchy.hierarchy))


class Concept(BaseModel):
    title: str
    description: str | None
    related: list[str] | None
    source: str | None

    consist_of: list['Concept'] | None


class KnowledgeMap(BaseModel):
    concepts: list[Concept]
