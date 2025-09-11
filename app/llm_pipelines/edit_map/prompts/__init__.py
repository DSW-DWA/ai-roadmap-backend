"""Prompts for knowledge map generation pipeline"""

import pathlib

from iso639 import Language
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from app.llm_pipelines.models import KnowledgeMap

here = pathlib.Path(__file__).parent.resolve()
jinja_env = Environment(loader=FileSystemLoader(str(here)), undefined=StrictUndefined)


def edit_map_prompt(
    *,
    material: dict[str, str],
    knowledge_map: KnowledgeMap,
    user_query: str,
    language: str,
    allowed_sources: list[str],
    response_model: type[BaseModel],
) -> list[ChatCompletionMessageParam]:
    """Creates prompt for hierarchical concepts extraction."""

    system_template = jinja_env.get_template('system.md.jinja')
    user_template = jinja_env.get_template('edit-map.md.jinja')
    knowledge_map_json = knowledge_map.model_dump_json(indent=4)
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
                material=material,
                knowledge_map=knowledge_map_json,
                user_query=user_query,
                json_schema=json_schema,
                allowed_sources=allowed_sources,
            ),
        },
    ]


__all__ = ['edit_map_prompt']
