from typing import Optional, List, Union, Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from vllm import SamplingParams
from core.config import settings
from services.llm_service import get_llm_service
import time
import uuid

# Pydantic-модели (ваши прежние, без изменений)
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
    # stream убираем из SamplingParams
    logprobs: Optional[int] = Field(default=None, le=5, description="Include the log probabilities on the logprobs most likely tokens")
    echo: Optional[bool] = Field(default=False, description="Echo back the prompt in addition to the completion")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="Up to 4 sequences where the API will stop generating further tokens")
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0, description="Positive values penalize new tokens based on their existing frequency in the text so far")
    best_of: Optional[int] = Field(default=None, ge=1, le=20, description="Generates best_of completions server-side and returns the best")
    logit_bias: Optional[Dict[int, float]] = Field(default=None, description="Modify the likelihood of specified tokens appearing in the completion")
    user: Optional[str] = Field(default=None, description="A unique identifier representing your end-user")
    seed: Optional[int] = Field(default=None, description="If specified, system will make a best effort to sample deterministically")

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

llm_router = APIRouter()

@llm_router.post("/completions", response_model=CompletionResponse, tags=["llm"])
async def create_completion(request: CompletionRequest):
    """
    Создаёт completion, совместимый с OpenAI Completions API, используя vLLM.
    """
    try:
        llm_service = get_llm_service()

        # 1) Собираем полный словарь дефолтов из settings (все ключи, которые хотим)
        sampling_kwargs: Dict[str, Any] = {
            "temperature": settings.TEMPERATURE,
            "max_tokens":  settings.MAX_TOKENS,
            "top_p":      getattr(settings, "TOP_P", None),  # TOP_P может быть не задан в settings
            "top_k":      getattr(settings, "TOP_K", None),  # TOP_K может быть не задан в settings
            # "stream":     settings.STREAM,  # Убираем из SamplingParams
            "n":          getattr(settings, "N", None),
            "presence_penalty":   getattr(settings, "PRESENCE_PENALTY", None),
            "frequency_penalty":  getattr(settings, "FREQUENCY_PENALTY", None),
            "best_of":            getattr(settings, "BEST_OF", None),
            "stop":               getattr(settings, "STOP", None),
            "logprobs":           getattr(settings, "LOGPROBS", None),
            "logit_bias":         getattr(settings, "LOGIT_BIAS", None),
            "seed":               getattr(settings, "SEED", None),
            # Параметры stream/echo убираем отсюда — vLLM не принимает их в SamplingParams
            # "stream": False,
            # "echo":   False,
        }

        # 2) Поверх дефолтов из settings перезаписываем те поля, что явно пришли в request != None
        if request.temperature is not None:
            sampling_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            sampling_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            sampling_kwargs["top_k"] = request.top_k 

        if request.max_tokens is not None:
            sampling_kwargs["max_tokens"] = request.max_tokens
        # if request.stream is not None:
        #     sampling_kwargs["stream"] = request.stream
        if request.n is not None:
            sampling_kwargs["n"] = request.n

        if request.presence_penalty is not None:
            sampling_kwargs["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            sampling_kwargs["frequency_penalty"] = request.frequency_penalty

        if request.best_of is not None:
            sampling_kwargs["best_of"] = request.best_of
        if request.seed is not None:
            sampling_kwargs["seed"] = request.seed
        if request.stop is not None:
            sampling_kwargs["stop"] = request.stop
        if request.logprobs is not None:
            sampling_kwargs["logprobs"] = request.logprobs
        if request.logit_bias is not None and request.logit_bias:
            sampling_kwargs["logit_bias"] = request.logit_bias

        # 3) “Очищаем” словарь от ключей со значением None, чтобы vLLM сам применил свои дефолты там, где нужно
        clean_kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}

        # 4) Создаём объект SamplingParams без stream/echo
        sampling_params = SamplingParams(**clean_kwargs)
        print(f"Используемые параметры сэмплинга: {sampling_params}")

        # 5) Обрабатываем параметр prompt — может быть строка или список строк
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]

        # 6) Выбираем модель: если пользователь не передал, берём из settings.VLLM_MODEL
        target_model = request.model or settings.VLLM_MODEL

        # 7) Генерируем через ваш llm_service
        outputs = llm_service.generate(prompts, sampling_params, target_model)

        # 8) Собираем в OpenAI-совместимый формат ответа
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

    except Exception as e:
        # Если vLLM вернёт какую-то ошибку (например, “unexpected keyword”),
        # она отловится здесь и вернётся как 500 с подходящим сообщением.
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")
    









class CompletionRequestChat(CompletionRequest):
    """Сообщение в формате OpenAI Chat Completions API."""
    content: Union[str, List[str]] = Field(
        ..., description="запрос к модели"
    )

@llm_router.post("/chat/completions", response_model=CompletionResponse, tags=["llm"])

async def create_chat_completion(request: CompletionRequestChat):
    """
    Создаёт chat completion, совместимый с OpenAI Chat Completions API, используя vLLM.
    """
    try:
        llm_service = get_llm_service()

        # 1) Собираем полный словарь дефолтов из settings (все ключи, которые хотим)
        sampling_kwargs: Dict[str, Any] = {
            "temperature": settings.TEMPERATURE,
            "max_tokens":  settings.MAX_TOKENS,
            "top_p":      getattr(settings, "TOP_P", None),  # TOP_P может быть не задан в settings
            "top_k":      getattr(settings, "TOP_K", None),  # TOP_K может быть не задан в settings
            # "stream":     settings.STREAM,  # Убираем из SamplingParams
            "n":          getattr(settings, "N", None),
            "presence_penalty":   getattr(settings, "PRESENCE_PENALTY", None),
            "frequency_penalty":  getattr(settings, "FREQUENCY_PENALTY", None),
            "best_of":            getattr(settings, "BEST_OF", None),
            "stop":               getattr(settings, "STOP", None),
            "logprobs":           getattr(settings, "LOGPROBS", None),
            "logit_bias":         getattr(settings, "LOGIT_BIAS", None),
            "seed":               getattr(settings, "SEED", None),
            # Параметры stream/echo убираем отсюда — vLLM не принимает их в SamplingParams
            # "stream": False,
            # "echo":   False,
        }

        # 2) Поверх дефолтов из settings перезаписываем те поля, что явно пришли в request != None
        if request.temperature is not None:
            sampling_kwargs["temperature"] = request.temperature
        if request.top_p is not None:
            sampling_kwargs["top_p"] = request.top_p
        if request.top_k is not None:
            sampling_kwargs["top_k"] = request.top_k 

        if request.max_tokens is not None:
            sampling_kwargs["max_tokens"] = request.max_tokens
        # if request.stream is not None:
        #     sampling_kwargs["stream"] = request.stream
        if request.n is not None:
            sampling_kwargs["n"] = request.n

        if request.presence_penalty is not None:
            sampling_kwargs["presence_penalty"] = request.presence_penalty
        if request.frequency_penalty is not None:
            sampling_kwargs["frequency_penalty"] = request.frequency_penalty

        if request.best_of is not None:
            sampling_kwargs["best_of"] = request.best_of
        if request.seed is not None:
            sampling_kwargs["seed"] = request.seed
        if request.stop is not None:
            sampling_kwargs["stop"] = request.stop
        if request.logprobs is not None:
            sampling_kwargs["logprobs"] = request.logprobs
        if request.logit_bias is not None and request.logit_bias:
            sampling_kwargs["logit_bias"] = request.logit_bias

        # 3) “Очищаем” словарь от ключей со значением None, чтобы vLLM сам применил свои дефолты там, где нужно
        clean_kwargs = {k: v for k, v in sampling_kwargs.items() if v is not None}

        # 4) Создаём объект SamplingParams без stream/echo
        sampling_params = SamplingParams(**clean_kwargs)
        print(f"Используемые параметры сэмплинга: {sampling_params}")
        # 5) Обрабатываем параметр prompt — может быть строка или список строк
        
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        content = request.content if isinstance(request.content, list) else [request.content]
        conversation = [
            {"role": "system", "content": content},
            {"role": "user", "content": prompts},
        ]

        # 6) Выбираем модель: если пользователь не передал, берём из settings.VLLM_MODEL
        target_model = request.model or settings.VLLM_MODEL

        # 7) Генерируем через ваш llm_service
        outputs = llm_service.chat_completions(conversation, sampling_params, target_model)

        # 8) Собираем в OpenAI-совместимый формат ответа
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

    except Exception as e:
        # Если vLLM вернёт какую-то ошибку (например, “unexpected keyword”),
        # она отловится здесь и вернётся как 500 с подходящим сообщением.
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")
