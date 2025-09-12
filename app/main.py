import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from app.llm_pipelines import BuildMapPipeline
from app.llm_pipelines.edit_map.pipeline import EditMapPipeline
from app.llm_pipelines.models import KnowledgeMap
from app.settings import settings

from .utils import extract_text_blobs_to_dict, validate_files

client = AsyncOpenAI(
    api_key=settings.yandex_cloud_api_key,
    base_url=str(settings.openai_base_url),
    timeout=3600
)

build_map_pipeline = BuildMapPipeline(
    client=client, model=settings.model_name, model_lite=settings.model_name_lite
)

edit_map_pipeline = EditMapPipeline(client=client, model=settings.model_name)

app = FastAPI(
    title='SQL Roadmap API',
    description=(
        'Две ручки:\n'
        '1) POST /roadmap/from-files — до 5 файлов (≤5 МБ) ⇒ JSON роадмапа по SQL.\n'
        '2) POST /roadmap/rewrite — принимает JSON роадмапа + промпт ⇒ изменённый JSON.'
    ),
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/roadmap/from-files', response_model=KnowledgeMap)
async def roadmap_from_files(
    files: Annotated[list[UploadFile], File(..., description='До 5 файлов, ≤5 МБ каждый')],
):
    """Craete roadmap from raw materials"""
    await validate_files(files)
    material = await extract_text_blobs_to_dict(files)
    knowledge_map = await build_map_pipeline.build(material)
    return knowledge_map.model_dump()


@app.post(
    '/roadmap/rewrite', response_model=KnowledgeMap, summary='Переписать роадмап по промпту'
)
async def roadmap_rewrite(
    knowledge_map: Annotated[str, Form()],
    user_query: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File(..., description='До 5 файлов, ≤5 МБ каждый')],
):
    """Edit knowledge map by user prompt"""
    try:
        knowledge_map_model = KnowledgeMap(**json.loads(knowledge_map))
        await validate_files(files)
        material = await extract_text_blobs_to_dict(files)
        updated_knowledge_map = await edit_map_pipeline.edit(
            material, knowledge_map_model, user_query
        )
        return updated_knowledge_map
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f'Не удалось переписать роадмап: {e}'
        ) from e


@app.get('/', include_in_schema=False)
async def root():
    """Redirect to docs on root"""
    return {'ok': True, 'see': '/docs'}
