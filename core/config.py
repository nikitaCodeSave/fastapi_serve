"""Application settings loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    VLLM_MODEL: str = "/home/nikita/PROJECTS/vllm5090/project/vikhr/models/QVikhr-3-1.7B"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "info"
    API_PREFIX: str = "v1"
    MAX_TOKENS: int = 8192
    TEMPERATURE: float = 0.2
    STREAM: bool = False
    GPU_MEMORY_UTILIZATION: float = 0.5
    TENSOR_PARALLEL_SIZE: int = 1

    class Config:
        env_file = ".env"

settings = Settings()
