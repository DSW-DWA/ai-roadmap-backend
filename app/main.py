from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI

from app.llm_pipelines import BuildMapPipeline
from app.llm_pipelines.edit_map.pipeline import EditMapPipeline
from app.settings import settings

from .logic import rewrite_roadmap_with_prompt
from .models import RewriteRequest, Roadmap
from .utils import extract_text_blobs_to_dict, validate_files

client = AsyncOpenAI(
    api_key=settings.yandex_cloud_api_key,
    base_url=str(settings.openai_base_url),
)

build_map_pipeline = BuildMapPipeline(
    client=client, model=settings.model_name, model_lite=settings.model_name_lite
)

edit_map_pipeline = EditMapPipeline(
    client=client, model=settings.model_name
)

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


@app.post(
    '/roadmap/from-files', response_model=Roadmap, summary='Сгенерировать SQL-роадмап из файлов'
)
async def roadmap_from_files(
    files: List[UploadFile] = File(..., description='До 5 файлов, ≤5 МБ каждый'),
):
    await validate_files(files)
    material = await extract_text_blobs_to_dict(files)
    knowledge_map = await build_map_pipeline.build(material)
    return JSONResponse(content=knowledge_map.model_dump(), status_code=status.HTTP_200_OK)


@app.post('/roadmap/rewrite', response_model=Roadmap, summary='Переписать роадмап по промпту')
async def roadmap_rewrite(files: List[UploadFile] = File(..., description='До 5 файлов, ≤5 МБ каждый'), payload: RewriteRequest):
    try:
        await validate_files(files)
        material = await extract_text_blobs_to_dict(files)
        updated_knowledge_map = edit_map_pipeline.edit(material, payload.knowledge_map, payload.user_query)
        return JSONResponse(content=updated_knowledge_map.model_dump(), status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Не удалось переписать роадмап: {e}')


@app.get('/', include_in_schema=False)
async def root():
    return {'ok': True, 'see': '/docs'}
