"""
Сервис для работы с vLLM моделью.
Обеспечивает инициализацию и управление жизненным циклом модели.
"""
from typing import Optional, List, Union
import logging
from vllm import LLM, SamplingParams
from core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Сервис для работы с vLLM моделью."""
    
    def __init__(self):
        self._llm_model: Optional[LLM] = None
        self._current_model_name: Optional[str] = None
        self._is_initialized = False
    
    def initialize(self, model_name: Optional[str] = None) -> None:
        """Инициализировать vLLM модель."""
        # Определяем имя модели
        target_model = model_name or settings.VLLM_MODEL
        
        # Если модель уже инициализирована с той же моделью, не переинициализируем
        if self._is_initialized and self._current_model_name == target_model:
            logger.info(f"LLM модель уже инициализирована с {target_model}")
            return
            
        # Если нужно сменить модель, сначала освобождаем ресурсы
        if self._is_initialized and self._current_model_name != target_model:
            logger.info(f"Смена модели с {self._current_model_name} на {target_model} Пока не работает, нужно будет реализовать очистку ресурсов")
            self.shutdown()
            
        try:
            logger.info(f"Инициализация vLLM модели: {target_model}")
            self._llm_model = LLM(
                model=target_model,
                trust_remote_code=True,
                gpu_memory_utilization=settings.GPU_MEMORY_UTILIZATION,
                tensor_parallel_size=settings.TENSOR_PARALLEL_SIZE,
                # Дополнительные параметры можно добавить здесь
            )
            self._current_model_name = target_model
            self._is_initialized = True
            logger.info("vLLM модель успешно инициализирована")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации vLLM модели: {e}")
            self._current_model_name = None
            self._is_initialized = False
            raise
    
    def get_model(self, model_name: Optional[str] = None) -> LLM:
        """Получить инициализированную vLLM модель."""
        target_model = model_name or settings.VLLM_MODEL
        
        if not self._is_initialized or self._llm_model is None or self._current_model_name != target_model:
            self.initialize(target_model)
            
        return self._llm_model
    
    def generate(
        self, 
        prompts: Union[str, List[str]], 
        sampling_params: SamplingParams,
        model_name: Optional[str] = None
    ):
        """Генерировать текст с использованием vLLM модели."""
        model = self.get_model(model_name)
        return model.generate(prompts, sampling_params)
    
    def chat_completions(
        self, 
        messages: List[dict], 
        sampling_params: SamplingParams, 
        model_name: Optional[str] = None
    ):
        """Выполнить чат с моделью."""
        model = self.get_model(model_name)
        return model.chat(messages, sampling_params)
    
    def get_current_model_name(self) -> Optional[str]:
        """Получить имя текущей загруженной модели."""
        return self._current_model_name or settings.VLLM_MODEL
    
    def is_ready(self) -> bool:
        """Проверить, готова ли модель к работе."""
        return self._is_initialized and self._llm_model is not None
    
    def shutdown(self) -> None:
        """Завершить работу с моделью."""
        if self._llm_model is not None:
            logger.info(f"Завершение работы с vLLM моделью: {self._current_model_name}")
            # vLLM не требует явного закрытия, но можно добавить логику очистки
            self._llm_model = None
            self._current_model_name = None
            self._is_initialized = False

# Глобальный экземпляр сервиса (синглтон)
llm_service = LLMService()


def get_llm_service() -> LLMService:
    """Получить экземпляр LLM сервиса."""
    return llm_service