from fastapi import APIRouter
from services.llm_service import get_llm_service

health_router = APIRouter()

@health_router.get("/", tags=["health"])
async def health_check():
    return {"message": "Сервер работает!", "status": "healthy"}

@health_router.get("/model", tags=["health"])
async def model_status():
    """Проверить статус vLLM модели."""
    llm_service = get_llm_service()
    is_ready = llm_service.is_ready()
    
    return {
        "model_ready": is_ready,
        "status": "ready" if is_ready else "not_ready",
        "message": "Модель готова к работе" if is_ready else "Модель не инициализирована"
    }

# Для обратной совместимости
# router = health_router