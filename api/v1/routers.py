from fastapi import APIRouter

from api.v1.endpoints.health import health_router
from api.v1.endpoints.llm import llm_router

router = APIRouter()

router.include_router(health_router, prefix="/health", tags=["health"])
router.include_router(llm_router, tags=["llm"])