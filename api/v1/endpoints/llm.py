from typing import Optional, List, Union, Any, Dict
import uuid
import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from vllm import SamplingParams
from core.config import settings
from services.llm_service import get_llm_service


# Pydantic-модели
class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request model."""

    prompt: Union[str, List[str]] = Field(
        ..., description="The prompt(s) to generate completions for"
    )

    model: Optional[str] = Field(default=None, description="ID of the model to use")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum number of tokens to generate")

    presence_penalty: Optional[float] = Field(
        default=None, ge=-2.0, le=2.0, description="Positive values penalize new tokens based on whether they appear in the text so far"
    )
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: Optional[int] = Field(default=None, ge=0, description="Top-k sampling parameter (0 means no top-k)")
    n: Optional[int] = Field(default=None, ge=1, le=128, description="Number of completions to generate")
    logprobs: Optional[int] = Field(default=None, le=5, description="Include the log probabilities on the logprobs most likely tokens")
    echo: Optional[bool] = Field(default=False, description="Echo back the prompt in addition to the completion")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Up to 4 sequences where the API will stop generating further tokens")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Positive values penalize new tokens based on their existing frequency in the text so far")
    logit_bias: Optional[Dict[int, float]] = Field(default=None, description="Modify the likelihood of specified tokens appearing in the completion")
    user: Optional[str] = Field(default=None, description="A unique identifier representing your end-user")
    seed: Optional[int] = Field(default=None, description="If specified, system will make a best effort to sample deterministically")


class CompletionRequestChat(CompletionRequest):
    """
    Запрос для Chat Completions API.
    Расширяет базовый CompletionRequest добавлением системного контекста.
    """
    content: Union[str, List[str]] = Field(
        ..., 
        description="Системное сообщение или контекст для модели. Может быть строкой или списком строк."
    )


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage


# Вспомогательные функции
def _build_sampling_params(request: Union[CompletionRequest, CompletionRequestChat]) -> SamplingParams:
    """
    Создает объект SamplingParams на основе настроек и параметров запроса.
    Поддерживает как обычные completion запросы, так и chat completion запросы.
    """
    # 1) Собираем полный словарь дефолтов из settings
    sampling_kwargs: Dict[str, Any] = {
        "temperature": settings.TEMPERATURE,
        "max_tokens": settings.MAX_TOKENS,
        "top_p": getattr(settings, "TOP_P", None),
        "top_k": getattr(settings, "TOP_K", None),
        "n": getattr(settings, "N", None),
        "presence_penalty": getattr(settings, "PRESENCE_PENALTY", None),
        "frequency_penalty": getattr(settings, "FREQUENCY_PENALTY", None),
        "stop": getattr(settings, "STOP", None),
        "logprobs": getattr(settings, "LOGPROBS", None),
        "logit_bias": getattr(settings, "LOGIT_BIAS", None),
        "seed": getattr(settings, "SEED", None),
    }

    # 2) Поверх дефолтов перезаписываем параметры из request
    if request.temperature is not None:
        sampling_kwargs["temperature"] = request.temperature
    if request.top_p is not None:
        sampling_kwargs["top_p"] = request.top_p
    if request.top_k is not None:
        sampling_kwargs["top_k"] = request.top_k
    if request.max_tokens is not None:
        sampling_kwargs["max_tokens"] = request.max_tokens
    if request.n is not None:
        sampling_kwargs["n"] = request.n
    if request.presence_penalty is not None:
        sampling_kwargs["presence_penalty"] = request.presence_penalty
    if request.frequency_penalty is not None:
        sampling_kwargs["frequency_penalty"] = request.frequency_penalty
    if request.seed is not None:
        sampling_kwargs["seed"] = request.seed
    if request.stop is not None:
        sampling_kwargs["stop"] = request.stop
    if request.logprobs is not None:
        sampling_kwargs["logprobs"] = request.logprobs
    if request.logit_bias is not None and request.logit_bias:
        sampling_kwargs["logit_bias"] = request.logit_bias

    # 3) Очищаем словарь от None значений
    clean_kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}

    # 4) Создаем SamplingParams
    sampling_params = SamplingParams(**clean_kwargs)
    print(f"Используемые параметры сэмплинга: {sampling_params}")
    
    return sampling_params


def _build_completion_response(outputs, request: Union[CompletionRequest, CompletionRequestChat], target_model: str) -> CompletionResponse:
    """
    Создает ответ в формате OpenAI из результатов vLLM.
    Поддерживает как обычные completion запросы, так и chat completion запросы.
    """
    choices: List[CompletionChoice] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for output in outputs:
        # output.outputs — список CompletionOutput (обычно length = n)
        for completion_output in output.outputs:
            choice = CompletionChoice(
                text=completion_output.text,
                index=len(choices),
                logprobs=completion_output.logprobs if request.logprobs else None,
                finish_reason=completion_output.finish_reason
            )
            choices.append(choice)

            # Подсчитываем completion_tokens, если у completion_output есть token_ids
            if hasattr(completion_output, "token_ids") and completion_output.token_ids:
                total_completion_tokens += len(completion_output.token_ids)

        # Подсчёт prompt_tokens: грубый, через разделение по пробелам
        if hasattr(output, "prompt") and output.prompt:
            total_prompt_tokens += len(output.prompt.split())

    usage = CompletionUsage(
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_prompt_tokens + total_completion_tokens
    )

    response = CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=target_model,
        choices=choices,
        usage=usage
    )
    return response


# Роутер
llm_router = APIRouter()


@llm_router.post("/completions", response_model=CompletionResponse, tags=["llm"])
async def create_completion(request: CompletionRequest):
    """
    Создаёт completion, совместимый с OpenAI Completions API, используя vLLM.
    """
    try:
        llm_service = get_llm_service()

        # Создаем параметры сэмплинга
        sampling_params = _build_sampling_params(request)

        # Обрабатываем параметр prompt — может быть строка или список строк
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

        # Выбираем модель: если пользователь не передал, берём из settings.VLLM_MODEL
        target_model = request.model or settings.VLLM_MODEL

        # Генерируем через llm_service
        outputs = llm_service.generate(prompts, sampling_params, target_model)

        # Собираем в OpenAI-совместимый формат ответа
        return _build_completion_response(outputs, request, target_model)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


@llm_router.post("/chat/completions", response_model=CompletionResponse, tags=["llm"])
async def create_chat_completion(request: CompletionRequestChat):
    """
    Создаёт chat completion, совместимый с OpenAI Chat Completions API, используя vLLM.
    """
    try:
        llm_service = get_llm_service()

        # Создаем параметры сэмплинга
        sampling_params = _build_sampling_params(request)

        # Обрабатываем параметры prompt и content
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        content = request.content if isinstance(request.content, list) else [request.content]
        
        # Формируем разговор для chat completions
        conversation = []
        
        # Добавляем системное сообщение(я)
        for system_content in content:
            if system_content:  # Только если content не пустой
                conversation.append({"role": "system", "content": system_content})
        
        # Добавляем сообщения пользователя
        for user_prompt in prompts:
            if user_prompt:  # Только если prompt не пустой
                conversation.append({"role": "user", "content": user_prompt})

        # Выбираем модель: если пользователь не передал, берём из settings.VLLM_MODEL
        target_model = request.model or settings.VLLM_MODEL

        # Генерируем через llm_service
        outputs = llm_service.chat_completions(conversation, sampling_params, target_model)

        # Собираем в OpenAI-совместимый формат ответа
        return _build_completion_response(outputs, request, target_model)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")
