"""Prompts for knowledge map generation pipeline"""

import pathlib

from iso639 import Language
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

here = pathlib.Path(__file__).parent.resolve()
jinja_env = Environment(loader=FileSystemLoader(str(here)), undefined=StrictUndefined)


def hierarchy_with_sources_prompt(
    *,
    material: dict[str, str],
    language: str,
    allowed_sources: list[str],
    response_model: type[BaseModel],
) -> list[ChatCompletionMessageParam]:
    """Creates prompt for hierarchical concepts extraction."""

    system_template = jinja_env.get_template('system.md.jinja')
    user_template = jinja_env.get_template('hierarchy-with-sources.md.jinja')
    language_name = Language.match(language).name
    json_schema = response_model.model_json_schema()

    return [
        {
            'role': 'system',
            'content': system_template.render(language=language_name),
        },
        {
            'role': 'user',
            'content': user_template.render(
                material=material, json_schema=json_schema, allowed_sources=allowed_sources
            ),
        },
    ]


def related_concepts_prompt(
    *, concepts: list[str], response_model: type[BaseModel], language: str
) -> list[ChatCompletionMessageParam]:
    """Creates prompt for linking related concepts"""
    system_template = jinja_env.get_template('system.md.jinja')
    user_template = jinja_env.get_template('related-concepts.md.jinja')
    language_name = Language.match(language).name
    return [
        {
            'role': 'system',
            'content': system_template.render(language=language_name),
        },
        {
            'role': 'user',
            'content': user_template.render(
                concepts=concepts, json_schema=response_model.model_json_schema()
            ),
        },
    ]


def add_description_prompt(
    *,
    material: dict[str, str],
    concept: str,
    parent_concepts: list[str] | None,
    related_concepts: list[str] | None,
    language: str,
) -> list[ChatCompletionMessageParam]:
    """Creates prompt for generating concepts description"""
    system_template = jinja_env.get_template('system.md.jinja')
    user_template = jinja_env.get_template('add-description.md.jinja')
    language_name = Language.match(language).name
    return [
        {
            'role': 'system',
            'content': system_template.render(language=language_name),
        },
        {
            'role': 'user',
            'content': user_template.render(
                material=material,
                concept=concept,
                related_concepts=related_concepts,
                parent_concepts=parent_concepts,
            ),
        },
    ]


__all__ = ['hierarchy_with_sources_prompt', 'related_concepts_prompt', 'add_description_prompt']
