from openai import AsyncOpenAI

from app.llm_pipeline.prompts import build_map_prompt

from .response_model import KnowledgeMap


async def build_map(
    client: AsyncOpenAI, model: str, material: dict[str, str], language: str = 'ru'
) -> KnowledgeMap:
    """Build knowledge map from list of educational materials using OpenAI model"""
    prompt = build_map_prompt(material, language)

    response = await client.chat.completions.parse(
        model=model,  # 'gpt://<folder_ID>/yandexgpt',
        messages=prompt,
        response_format=KnowledgeMap,
    )

    content = response.choices[0].message
    assert content.parsed

    return content.parsed
