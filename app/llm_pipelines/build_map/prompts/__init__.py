"""Prompts for knowledge map generation pipeline"""

import pathlib
from typing import Any

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
    step_one_template = jinja_env.get_template('hierarchy-with-sources.md.jinja')
    language_name = Language.match(language).name
    json_schema = response_model.model_json_schema()

    return [
        {
            'role': 'system',
            'content': system_template.render(language=language_name),
        },
        {
            'role': 'user',
            'content': step_one_template.render(
                material=material, json_schema=json_schema, allowed_sources=allowed_sources
            ),
        },
    ]


def step_two_related_concepts_prompt(
    *, response_model: type[BaseModel]
) -> list[ChatCompletionMessageParam]:
    """Creates prompt for linking related concepts"""
    step_three_template = jinja_env.get_template('step-3-related-concepts.md.jinja')
    return [
        {
            'role': 'user',
            'content': step_three_template.render(
                json_schema=response_model.model_json_schema()
            ),
        }
    ]


__all__ = ['hierarchy_with_sources_prompt', 'step_three_related_concepts_prompt']
