import re
from inspect import cleandoc

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.llm_pipelines.build_map.prompts import (
    hierarchy_with_sources_prompt,
    related_concepts_prompt,
    # step_two_add_sources,
)

type ConceptHierarchy = dict[str, 'ConceptHierarchyNode']


class ConceptHierarchyNode(BaseModel):
    """Единичный элемент в иерархии концептов"""

    sources: list[str] | None = Field(
        description=cleandoc("""
            Каждое значение — это ссылка в формате file-path или file-path#Header-1/Header-2.
            В списке обязательно должна быть одна ссылка
        """),
    )

    consists_of: ConceptHierarchy | None = Field(
        default=None,
        description=cleandoc(""" 
            Представляет собой словарь, где ключи — это концепты, а значения — данные для них.

            Делает модель рекурсивной:
            - Концепт может состоять из подконцептов, формируя один шаг рекурсии. В этом случае значение consists_of будет словарём с подконцептами.
            - Либо он может быть самостоятельным (базовый случай) — листовым узлом. Если концепт является листовым узлом, то есть не имеет подконцептов, то consists_of = null.
        """),
    )


class ConceptHierarchyModel(BaseModel):
    """Модель для иерархически организованных концептов"""

    hierarchy: ConceptHierarchy = Field(
        description=cleandoc("""
            Отображение концептов: ключ — это название концепта, значение — данные для этого концепта.
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


def preprocess_hierarchy(
    hierarchy: ConceptHierarchy, allowed_sources: list[str]
) -> ConceptHierarchy:
    """Normalize sources and concept names format, clear hallucinated or mistaken sources"""
    new_hierarchy = hierarchy.copy()

    def _fix_source_name(src: str):
        file, _, headers = src.partition('#')
        headers = '/'.join(format_header(h) for h in headers.split('/'))
        return f'{file}#{headers}'

    def _fix_hallucinated_header(src: str, allowed_sources):
        # if file name is exists but header does not, truncate to file-name only
        if src in allowed_sources:
            return src

        file, _, _ = src.partition('#')
        if file in allowed_sources:
            return file

    def _prepocess(hierarchy: ConceptHierarchy) -> ConceptHierarchy:
        new_hierarchy = {}
        for name, concept in hierarchy.items():
            new_sources = (
                list(
                    map(
                        lambda s: _fix_hallucinated_header(
                            _fix_source_name(s), allowed_sources=set(allowed_sources)
                        ),
                        concept.sources,
                    ),
                )
                if concept.sources
                else None
            )

            new_hierarchy[name.removeprefix('_')] = ConceptHierarchyNode(
                sources=new_sources,
                consists_of=_prepocess(concept.consists_of) if concept.consists_of else None,
            )
        return new_hierarchy

    new_hierarchy = _prepocess(hierarchy)

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


def preprocess_related(
    related: RelatedConcepts, hierarchy: ConceptHierarchy
) -> RelatedConcepts:
    concepts = flatten_hierarchy(hierarchy)
    ancestors = get_parents_map(hierarchy)
    descendats = get_children_map(hierarchy)

    new_related = related.related.copy()

    for key, value in list(new_related.items()):
        new_related[key] = (
            list(
                filter(
                    lambda concept: concept in concepts
                    and concept not in ancestors.get(key, [])
                    and concept not in descendats.get(key, []),
                    value,
                )
            )
            or None
        )

    return related.model_copy(update={'related': new_related})


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
        hierarchy_with_sources.hierarchy = preprocess_hierarchy(
            hierarchy_with_sources.hierarchy, allowed_sources=sources
        )

        return hierarchy_with_sources

    async def _link_related(self, hierarchy: ConceptHierarchy) -> RelatedConcepts:
        concepts = flatten_hierarchy(hierarchy)

        messages = related_concepts_prompt(
            concepts=list(concepts), response_model=RelatedConcepts
        )

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
        related = preprocess_related(message.parsed, hierarchy)

        return related

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
        related = await self._link_related(hierarchy.hierarchy)

        def _convert_hierarchy_to_concepts(
            nodes: ConceptHierarchy, related_map: dict[str, list[str] | None]
        ) -> list[Concept]:
            concept_list = []
            if not nodes:
                return []

            for title, node_data in nodes.items():
                # Recursively process any sub-concepts.
                sub_concepts = None
                if node_data.consists_of:
                    sub_concepts = _convert_hierarchy_to_concepts(
                        node_data.consists_of, related_map
                    )

                # Create the new Concept, transforming data to the target format.
                concept = Concept(
                    title=title,
                    description=None,  # This field is not generated by the current pipeline.
                    related=related_map.get(title),
                    # Take the first source if multiple are present.
                    source=node_data.sources,
                    consist_of=sub_concepts,
                )
                concept_list.append(concept)

            return concept_list

        # Step 3: Convert the intermediate structures into the final KnowledgeMap.
        final_concepts = _convert_hierarchy_to_concepts(hierarchy.hierarchy, related.related)

        return KnowledgeMap(concepts=final_concepts)


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
