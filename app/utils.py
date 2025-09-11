import io
from typing import Dict, List

from fastapi import HTTPException, UploadFile, status
from markitdown import MarkItDown

md = MarkItDown()

MAX_FILES = 5
MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB


async def validate_files(files: List[UploadFile]) -> None:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='Нужно загрузить хотя бы один файл.'
        )
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Можно не более {MAX_FILES} файлов.',
        )

    # Проверяем каждый файл по размеру
    for f in files:
        # читаем контент и проверяем размер
        content = await f.read()
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f'Файл {f.filename} превышает лимит 5 МБ.',
            )
        # вернём указатель в начало, если кто-то захочет перечитать
        await f.seek(0)


async def extract_text_blobs(files: List[UploadFile]) -> List[str]:
    """
    Простейшее извлечение текста:
    - .txt, .md читаем как UTF-8
    - остальное игнорируем (или кладём имя файла как подсказку)
    В реальном проекте можно добавить парсеры для PDF/Docx и т.п.
    """
    blobs = []
    for f in files:
        name = (f.filename or '').lower()

        content = await f.read()
        await f.seek(0)  # Reset for potential reuse

        if name.endswith(('.txt', '.md', '.csv', '.sql')):
            try:
                text = content.decode('utf-8', errors='ignore')
            except Exception:
                text = ''
            blobs.append(text[:200_000])
        # Use MarkItDown for other types (PDF, DOCX, PPTX, etc.)
        else:
            byte_stream = io.BytesIO(content)
            try:
                result = md.convert_stream(byte_stream, filename=f.filename)
                if result and result.text_content:
                    blobs.append(result.text_content[:200_000])
                else:
                    blobs.append(f"FILE:{f.filename} (no readable content)")
            except Exception as e:
                blobs.append(f"FILE:{f.filename} (conversion failed: {str(e)})")

    return blobs


async def extract_text_blobs_to_dict(files: List[UploadFile]) -> Dict[str, str]:
    """
    Extract text from files and return a dictionary where:
    - key: filename
    - value: extracted text content
    """
    material_dict = {}

    for f in files:
        name = (f.filename or '').lower()
        content = await f.read()
        await f.seek(0)

        if name.endswith(('.txt', '.md', '.csv', '.sql')):
            try:
                text = content.decode('utf-8', errors='ignore')
            except Exception:
                text = ''
            material_dict[f.filename] = text[:200_000]
        else:
            byte_stream = io.BytesIO(content)
            try:
                result = md.convert_stream(byte_stream, filename=f.filename)
                if result and result.text_content:
                    material_dict[f.filename] = result.text_content[:200_000]
                else:
                    material_dict[f.filename] = f"FILE:{f.filename} (no readable content)"
            except Exception as e:
                material_dict[f.filename] = f"FILE:{f.filename} (conversion failed: {str(e)})"

    return material_dict