from openai import AsyncOpenAI

# Предполагается, что эти модели и утилиты находятся в указанных путях
from app.llm_pipelines.edit_map.prompts import (
    edit_map_prompt,
)
from app.llm_pipelines.models import Concept, KnowledgeMap
from app.llm_pipelines.utils import format_header, get_allowed_sources

# TODO: refactor code duplication


def flatten_map(concepts: list[Concept] | None) -> set[str]:
    """Рекурсивно извлекает все уникальные названия концептов из списка концептов."""
    titles = set()
    if not concepts:
        return titles

    for concept in concepts:
        titles.add(concept.title)
        if concept.consist_of:
            titles.update(flatten_map(concept.consist_of))

    return titles


def get_parents_map_from_list(concepts: list[Concept] | None) -> dict[str, list[str]]:
    parents_map: dict[str, list[str]] = {}

    def _traverse(sub_concepts: list[Concept] | None, current_path: list[str]):
        if not sub_concepts:
            return

        for concept in sub_concepts:
            parents_map[concept.title] = current_path
            if concept.consist_of:
                new_path = current_path + [concept.title]
                _traverse(concept.consist_of, new_path)

    _traverse(concepts, [])
    return parents_map


def get_children_map_from_list(concepts: list[Concept] | None) -> dict[str, list[str]]:
    children_map: dict[str, list[str]] = {}

    def _traverse(sub_concepts: list[Concept] | None) -> list[str]:
        all_titles_in_subtree: list[str] = []
        if not sub_concepts:
            return all_titles_in_subtree

        for concept in sub_concepts:
            all_titles_in_subtree.append(concept.title)

            if concept.consist_of:
                descendants = _traverse(concept.consist_of)
                children_map[concept.title] = descendants
                all_titles_in_subtree.extend(descendants)
            else:
                children_map[concept.title] = []

        return all_titles_in_subtree

    _traverse(concepts)
    return children_map


def preprocess_edited_map(edited_map: KnowledgeMap, allowed_sources: list[str]) -> KnowledgeMap:
    all_titles = flatten_map(edited_map.concepts)
    parents_map = get_parents_map_from_list(edited_map.concepts)
    children_map = get_children_map_from_list(edited_map.concepts)
    allowed_sources_set = set(allowed_sources)

    def _fix_source_name(src: str) -> str:
        """Нормализует формат заголовков в строке источника."""
        file, _, headers = src.partition('#')
        if not headers:
            return file

        formatted_headers = '/'.join(format_header(h) for h in headers.split('/') if h)
        return f'{file}#{formatted_headers}' if formatted_headers else file

    def _fix_hallucinated_source(src: str) -> str | None:
        """Проверяет источник. Если заголовок неверный, возвращает только имя файла."""
        if src in allowed_sources_set:
            return src

        file, _, _ = src.partition('#')
        if file in allowed_sources_set:
            return file

        return None

    def _traverse_and_process(concepts: list[Concept] | None) -> list[Concept] | None:
        if not concepts:
            return None

        processed_concepts: list[Concept] = []
        for concept in concepts:
            new_sources = None
            if concept.source:
                processed_src = [
                    _fix_hallucinated_source(_fix_source_name(s)) for s in concept.source
                ]
                new_sources = [s for s in processed_src if s] or None

            new_related = None
            if concept.related:
                ancestors = set(parents_map.get(concept.title, []))
                descendants = set(children_map.get(concept.title, []))

                filtered_related = [
                    rel_title
                    for rel_title in concept.related
                    if (
                        rel_title in all_titles
                        and rel_title != concept.title
                        and rel_title not in ancestors
                        and rel_title not in descendants
                    )
                ]
                new_related = filtered_related or None

            processed_children = _traverse_and_process(concept.consist_of)

            processed_concept = concept.model_copy(
                update={
                    'source': new_sources,
                    'related': new_related,
                    'consist_of': processed_children,
                }
            )
            processed_concepts.append(processed_concept)

        return processed_concepts

    final_processed_concepts = _traverse_and_process(edited_map.concepts)

    return KnowledgeMap(concepts=final_processed_concepts or [])


class EditMapPipeline:
    def __init__(self, client: AsyncOpenAI, model: str):
        self._client: AsyncOpenAI = client
        self._model: str = model

    async def edit(
        self,
        material: dict[str, str],
        knowledge_map: KnowledgeMap,
        user_query: str,
        language: str = 'ru',
    ) -> KnowledgeMap:
        allowed_sources = get_allowed_sources(material)

        messages = edit_map_prompt(
            material=material,
            knowledge_map=knowledge_map,
            user_query=user_query,
            allowed_sources=allowed_sources,
            language=language,
            response_model=KnowledgeMap,
        )

        response = await self._client.chat.completions.parse(
            model=self._model,
            messages=messages,
            response_format=KnowledgeMap,
            temperature=0.0,
            seed=42,
            max_tokens=4096,
        )

        message = response.choices[0].message
        assert message.parsed

        edited_map = preprocess_edited_map(
            edited_map=message.parsed, allowed_sources=allowed_sources
        )

        return edited_map
