import asyncio

from openai import AsyncOpenAI

from app.llm_pipelines import BuildMapPipeline

from .settings import settings


async def test():
    with open('distributed_systems.md') as file:
        material = {'distributed_systems.md': file.read()}
    
    with open('k8s.md') as file:
        material |= {'k8s.md': file.read()}

    client = AsyncOpenAI(
        api_key=settings.yandex_cloud_api_key,
        base_url=str(settings.openai_base_url),
    )

    build_map_pipeline = BuildMapPipeline(
        client=client, model=settings.model_name
    )
    hierarchy, prerequisites = await build_map_pipeline.build(material)
    
    print(hierarchy.model_dump_json())
    print(prerequisites.model_dump_json())
    # print(knowledge_map.model_dump_json(exclude_none=True))


asyncio.run(test())
