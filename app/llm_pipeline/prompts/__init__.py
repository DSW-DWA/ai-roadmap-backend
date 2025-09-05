"""Prompts for knowledge map generation pipeline"""

import pathlib

from iso639 import Language
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai.types.chat import ChatCompletionMessageParam

from ..response_model import KnowledgeMap

here = pathlib.Path(__file__).parent.resolve()
jinja_env = Environment(loader=FileSystemLoader(str(here)), undefined=StrictUndefined)


def build_map_prompt(
    material: dict[str, str], language: str
) -> list[ChatCompletionMessageParam]:
    """Create prompt for knowledge map building in OpenAI chat-completions format"""
    system_template = jinja_env.get_template('build_map/system.md.jinja')
    user_template = jinja_env.get_template('build_map/user.md.jinja')
    language_name = Language.match(language).name

    return [
        {
            'role': 'system',
            'content': system_template.render(
                json_schema=KnowledgeMap.model_json_schema(), language=language_name
            ),
        },
        {'role': 'user', 'content': user_template.render(material=material)},
    ]


__all__ = ['build_map_prompt']
