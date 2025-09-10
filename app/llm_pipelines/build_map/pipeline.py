import re
from inspect import cleandoc
from itertools import chain

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from app.llm_pipelines.build_map.prompts import (
    hierarchy_with_sources_prompt,
    # step_two_add_sources,
)

type ConceptHierarchy = dict[str, 'ConceptHierarchyNode']


class ConceptHierarchyNode(BaseModel):
    """Single item in concepts hierarhy"""

    sources: list[str] = Field(
        description=cleandoc("""
            Each value is a link in the format file-path or file-path#Header-1/Header-2
        """),
    )

    consists_of: ConceptHierarchy | None = Field(
        default=None,
        description=cleandoc(""" 
            Is a dictionary, with concepts as keys and their data as values.

            This makes model recustive:
            - A concept can be composed of sub-concepts, making one step into recursion. In this case value of consist_of will be a dictionary with sub-concepts.
            - Or it may stand alone (base case) as a leaf node. If concepts is leaf node, i.e has no sub-concepts, consists_of is null.
        """),
    )


class ConceptHierarchyModel(BaseModel):
    """Model for hierarchically-organized concepts"""

    hierarchy: ConceptHierarchy = Field(
        description=cleandoc("""
            Mapping for concepts, key is a concepts name, and value is data for this concept.
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


class RelatedConcepts(BaseModel):
    """Concepts which are related to each other"""

    related: dict[str, list[str]] = Field(
        description=cleandoc("""
            A mapping, where each key is a concept and value is concepts related to it.
            This is used to mark cross-hierarchical relationship.
        """)
    )


def format_header(header: str) -> str:
    """Convert header into slug for source"""
    return header.replace(' ', '').replace('(', '').replace(')', '')


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
            result_paths.append('/'.join(map(format_header, current_path)))

    return result_paths


def get_allowed_sources(material):
    """Get all possible sources from material, both in {filename} and {filename}#{header_path} format"""
    values = list(material.keys())
    for name, content in material.items():
        for header_path in get_markdown_header_paths(content):
            values.append(f'{name}#{header_path}')

    return values


def clear_sources(hierarchy: ConceptHierarchy, allowed_sources: list[str]) -> ConceptHierarchy:
    """Remove non-existing sources"""
    allowed_sources_lookup = set(allowed_sources)
    new_hierarchy = hierarchy.copy()

    def _clear(hierarchy: ConceptHierarchy):
        for concept in hierarchy.values():
            if not concept.consists_of:
                return
            concept.sources = list(
                filter(lambda s: s in allowed_sources_lookup, concept.sources)
            )
            _clear(concept.consists_of)

    _clear(new_hierarchy)

    return new_hierarchy


def flatten_hierarchy(hierarchy: ConceptHierarchy) -> set[str]:
    """Extract all unique concept names from a ConceptHierarchy."""
    concepts = set()
    for concept_name, node in hierarchy.items():
        concepts.add(concept_name)
        if node.consists_of:
            concepts.update(flatten_hierarchy(node.consists_of))
    return concepts


def get_parents_map(hierarchy: ConceptHierarchy) -> dict[str, list[str]]:
    """
    Generates a map where each key is a concept name and the value is a list of its parent concepts, tracing back to the root.

    Parameters
    ----------
        hierarchy: The top-level concepts hierarchy dictionary.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each concept to its list of parents.
    """
    parents_map: dict[str, list[str]] = {}

    def _traverse(sub_hierarchy: ConceptHierarchy, current_path: list[str]):
        if not sub_hierarchy:
            return

        for concept_name, node_data in sub_hierarchy.items():
            # The current path represents the parents of this concept.
            parents_map[concept_name] = current_path

            # If this node has sub-concepts, recurse deeper.
            if node_data.consists_of:
                # The new path for the children includes the current concept.
                new_path = current_path + [concept_name]
                _traverse(node_data.consists_of, new_path)

    _traverse(hierarchy, [])
    return parents_map


def get_children_map(hierarchy: ConceptHierarchy) -> dict[str, list[str]]:
    """
    Generates a map where each key is a concept name and the value is a flat list of all its descendant concepts (children, grandchildren, etc.).

    Parameters
    ----------
        hierarchy: The top-level concepts hierarchy dictionary.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each concept to its flat list of descendants.
    """
    children_map: dict[str, list[str]] = {}

    def _traverse(sub_hierarchy: ConceptHierarchy) -> list[str]:
        # This list will include all concepts at the current level and below.
        all_concepts_in_subtree: list[str] = []

        if not sub_hierarchy:
            return all_concepts_in_subtree

        for concept_name, node_data in sub_hierarchy.items():
            # Add the current concept to the list for this subtree.
            all_concepts_in_subtree.append(concept_name)

            # If the node has children, process them recursively.
            if node_data.consists_of:
                # The returned list contains all descendants of the current node.
                descendants = _traverse(node_data.consists_of)
                children_map[concept_name] = descendants
                all_concepts_in_subtree.extend(descendants)
            else:
                # Leaf nodes have no children.
                children_map[concept_name] = []

        return all_concepts_in_subtree

    _traverse(hierarchy)
    return children_map


class BuildMapPipeline:
    """
    LLM pipeline for constructing a knowledge maps.

    Works by orchestrating multiple micro-agents. Each micro-agent is responsible for a specific step in the pipeline, such as:
    - Extracting concepts and organizing them hierarchically (combined)
    - Restoring sources
    - Linking related concepts
    - Generating short descriptions
    """

    def __init__(self, client: AsyncOpenAI, model: str, model_lite: str):
        self._client: AsyncOpenAI = client
        self._model: str = model
        self._model_lite: str = model_lite

    async def _build_hierarchy(
        self,
        material: dict[str, str],
        language: str,
    ) -> ConceptHierarchyModel:
        sources = get_allowed_sources(material)
        messages = hierarchy_with_sources_prompt(
            material=material,
            allowed_sources=sources,
            language=language,
            response_model=ConceptHierarchyModel,
        )

        response = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=ConceptHierarchyModel,
            temperature=0.0,
            seed=42,
            max_tokens=4096,
        )

        message = response.choices[0].message
        assert message.parsed
        assert message.content

        hierarchy_with_sources = message.parsed
        hierarchy_with_sources.hierarchy = clear_sources(
            hierarchy_with_sources.hierarchy, allowed_sources=sources
        )

        return hierarchy_with_sources

    # async def _add_sources(
    #    self,
    #    messages: list[ChatCompletionMessageParam],
    #    concepts: Iterable[str],
    #    material: dict[str, str],
    # ) -> ConceptSources:
    #    messages += step_two_add_sources(response_model=ConceptSources)

    #    response = await self._client.chat.completions.parse(
    #        model=self._model,
    #        messages=messages,
    #        response_format=ConceptSources,
    #        temperature=0.0,
    #        seed=42,
    #    )

    #    message = response.choices[0].message
    #    assert message.parsed
    #    assert message.content

    #    sources = message.parsed
    #    print(sources)
    #    allowed_sources = get_allowed_sources(material)
    #    print(allowed_sources)
    #    for concept in list(sources.sources):
    #        source = sources.sources[concept].replace(' ', '-')
    #        if concept not in concepts or source not in allowed_sources:
    #            sources.sources.pop(concept)
    #        else:
    #            sources.sources[concept] = source
    #    print(sources)
    #    return sources

    async def _link_related(
        self, hierarchy: ConceptHierarchy, messages: list[ChatCompletionMessageParam]
    ) -> RelatedConcepts:
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

        return message.parsed

    async def build(self, material: dict[str, str], language: str = 'ru'):
        """
        Build a complete knowledge map from educational material.

        Parameters
        ----------
        material : dict[str, str]
            Dictionary mapping source names to their content
        language : str, default='ru'
            Language code for processing

        Returns
        -------
        KnowledgeMap
            Complete knowledge map with hierarchical concepts, sources, and relationships
        """
        hierarchy = await self._build_hierarchy(material, language=language)
        # concepts = set(flatten_hierarchy(hierarchy.hierarchy))
        # sources = await self._add_sources(messages, concepts, material)
        # related = await self._link_related(hierarchy, messages)
        return hierarchy

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
    """
    A single concept in the knowledge map.

    Represents an individual learning concept with its metadata including
    relationships to other concepts and source attribution.
    """

    title: str
    description: str | None
    related: list[str] | None
    source: str | None

    consist_of: list['Concept'] | None


class KnowledgeMap(BaseModel):
    """
    Complete knowledge map containing all concepts and their relationships.

    The root container for a structured representation of educational content
    organized into hierarchical concepts with cross-references and source links.
    """

    concepts: list[Concept]
