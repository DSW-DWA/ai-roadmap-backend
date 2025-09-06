import asyncio
import json

from openai import AsyncOpenAI

from .llm_pipeline import build_map
from .settings import settings


async def test():
    with open('distributed_systems.md') as file:
        material = {'Распределенные_системы.md': file.read()}

    client = AsyncOpenAI(
        api_key=settings.yandex_cloud_api_key, base_url=str(settings.openai_base_url)
    )

    knowledge_map = await build_map(client, settings.model_name, material)
    print(knowledge_map.model_dump_json(exclude_none=True))


asyncio.run(test())
