"""Prompts for knowledge map generation pipeline"""

import pathlib
from typing import Any

from iso639 import Language
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

here = pathlib.Path(__file__).parent.resolve()
jinja_env = Environment(loader=FileSystemLoader(str(here)), undefined=StrictUndefined)


def step_one_hierarchy_prompt(
    *, material: dict[str, str], language: str, response_model: type[BaseModel]
) -> list[ChatCompletionMessageParam]:
    """Creates prompt for hierarchical concepts extraction."""
    
    system_template = jinja_env.get_template('system.md.jinja')
    step_one_template = jinja_env.get_template('step-1-hierarchy.md.jinja')
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
                material=material, json_schema=json_schema
            )
        },
    ]

def step_three_related_concepts_prompt(*, json_schema: dict[str, Any]) -> list[ChatCompletionMessageParam]:
    """Creates prompt for linking related concepts"""
    step_two_template = jinja_env.get_template('step-3-related-concepts.md.jinja')
    return [
        {
            'role': 'user',
            'content': step_two_template.render(json_schema=json_schema)
        }
    ]


__all__ = [
    'step_one_hierarchy_prompt', 'step_three_related_concepts_prompt'
]
