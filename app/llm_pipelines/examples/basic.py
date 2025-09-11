import asyncio

from openai import AsyncOpenAI

from app.llm_pipelines import BuildMapPipeline
from app.llm_pipelines.edit_map.pipeline import EditMapPipeline
from app.settings import settings


async def basic_example():
    with open('example_data/distributed_systems.md') as file:
        material = {'distributed_systems.md': file.read()}

    with open('example_data/k8s.md') as file:
        material |= {'k8s.md': file.read()}

    client = AsyncOpenAI(
        api_key=settings.yandex_cloud_api_key,
        base_url=str(settings.openai_base_url),
    )

    build_map_pipeline = BuildMapPipeline(
        client=client, model=settings.model_name, model_lite=settings.model_name_lite
    )
    knowledge_map = await build_map_pipeline.build(material)
    print(knowledge_map.model_dump_json(exclude_none=True))

    edit_map_pipeline = EditMapPipeline(client=client, model=settings.model_name)
    new_map = await edit_map_pipeline.edit(
        material,
        knowledge_map,
        """
        Измени каждое описание так, чтобы оно содержало слова и словосочетания:
        пельмени, база, базовый, составляющая, штучка, синхронизация,
        соответственно, провалится, как бог на душу положит, отказоустойчивость,
        правая тройка векторов, дай бог, как душе угодно, литературно, выпадение из контекста.
        """,
    )
    print(new_map.model_dump_json())


asyncio.run(basic_example())
