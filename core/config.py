import os
from dotenv import load_dotenv
from pydantic import BaseModel
# Загружаем переменные окружения из файла .env

load_dotenv()


class Settings(BaseModel):
    VLLM_MODEL: str = os.getenv("VLLM_MODEL", "/home/nikita/PROJECTS/vllm5090/project/vikhr/models/QVikhr-3-1.7B")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"  # Конвертируем строку в булево значение
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    API_PREFIX: str = os.getenv("API_PREFIX", "v1")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))  
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2")) 
    STREAM: bool = os.getenv("STREAM", "false").lower() == "true"  # Конвертируем строку в булево значение
    GPU_MEMORY_UTILIZATION: float = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.5"))  
    TENSOR_PARALLEL_SIZE: int = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))  

settings = Settings()