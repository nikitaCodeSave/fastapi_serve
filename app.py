from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from api.v1.routers import router as api_v1_router
from services.llm_service import get_llm_service


# Настройка логгирования (простая заглушка):
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI приложение с контекстным менеджером для управления жизненным циклом приложения
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting FastAPI application...")
    
    # Инициализация LLM сервиса при запуске
    try:
        llm_service = get_llm_service()
        llm_service.initialize()
        logger.info("LLM сервис успешно инициализирован")
    except Exception as e:
        logger.error(f"Ошибка инициализации LLM сервиса: {e}")
        raise
    
    yield
    
    # Завершение работы LLM сервиса при остановке
    try:
        llm_service = get_llm_service()
        llm_service.shutdown()
        logger.info("LLM сервис корректно завершил работу")
    except Exception as e:
        logger.error(f"Ошибка при завершении работы LLM сервиса: {e}")
    
    logger.info("Shutting down FastAPI application...")

app = FastAPI(lifespan=lifespan,
              title="vLLM Completion API",
             description="API for text completion using vLLM with OpenAI-compatible endpoints.",
             version="1.0.0",
)

app.include_router(api_v1_router, prefix="/v1")

