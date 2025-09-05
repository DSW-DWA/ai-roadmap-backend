from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import Roadmap, RewriteRequest
from .logic import generate_sql_roadmap, rewrite_roadmap_with_prompt
from .utils import validate_files, extract_text_blobs

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
    blobs = await extract_text_blobs(files)
    roadmap = generate_sql_roadmap(blobs)
    return JSONResponse(content=roadmap.model_dump(), status_code=status.HTTP_200_OK)


@app.post('/roadmap/rewrite', response_model=Roadmap, summary='Переписать роадмап по промпту')
async def roadmap_rewrite(payload: RewriteRequest):
    try:
        updated = rewrite_roadmap_with_prompt(payload.roadmap, payload.prompt)
        return JSONResponse(content=updated.model_dump(), status_code=status.HTTP_200_OK)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'Не удалось переписать роадмап: {e}')


@app.get('/', include_in_schema=False)
async def root():
    return {'ok': True, 'see': '/docs'}
