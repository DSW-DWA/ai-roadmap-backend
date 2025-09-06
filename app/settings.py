from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration"""

    model_config = SettingsConfigDict(  # pyright: ignore[reportUnannotatedClassAttribute]
        env_file='.env', env_file_encoding='utf-8', extra='ignore'
    )

    yandex_cloud_folder: str
    yandex_cloud_api_key: str
    model_name: str
    openai_base_url: HttpUrl


settings = Settings()  # pyright: ignore[reportCallIssue]
