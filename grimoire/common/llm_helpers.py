import time
from dataclasses import dataclass
from urllib.parse import urljoin

import requests
from transformers import AutoTokenizer

from grimoire.common.loggers import general_logger
from grimoire.common.redis import redis_manager
from grimoire.common.utils import time_execution
from grimoire.core.settings import ApiSettings, settings

_tokenizer_cache: dict[str, AutoTokenizer] = {}

# Keys Grimoire injects for Kobold-style text completion that OpenAI-style chat completion
# endpoints don't understand and may reject.
_TEXT_ONLY_PARAM_KEYS = ("max_length", "truncation_length", "max_context_length", "stop_sequence")

DEFAULT_REASONING_START_TOKEN = "<think>"
DEFAULT_REASONING_END_TOKEN = "</think>"


@dataclass
class GenerationResult:
    text: str
    reasoning: str
    request_body: dict


def split_reasoning(
    text: str,
    strip_reasoning: bool = True,
    start_token: str = DEFAULT_REASONING_START_TOKEN,
    end_token: str = DEFAULT_REASONING_END_TOKEN,
) -> tuple[str, str]:
    """Split reasoning-model output into (content, reasoning).

    The delimiters are model-specific (``<think>``/``</think>`` for DeepSeek-R1 and Qwen,
    ``<reasoning>``, ``<|channel|>analysis``... for others), so they come from
    ``ApiSettings.reasoning_start_token`` / ``reasoning_end_token``. Everything up to the last
    end token is treated as reasoning, which handles two shapes:

    - a full ``start_token ... end_token`` block anywhere in the text
    - an orphaned end token with no opening one, which happens when the prompt prefills the
      start token itself (a common workaround for models that always think)

    When ``strip_reasoning`` is False, or no end token is configured, the text is returned
    unchanged with no reasoning captured.
    """
    if not strip_reasoning or not text or not end_token:
        return text, ""

    if end_token in text:
        reasoning, _, content = text.rpartition(end_token)
        if start_token:
            reasoning = reasoning.replace(start_token, "")
        return content.strip(), reasoning.strip()

    return text.strip(), ""


def _load_tokenizer(tokenizer_name: str) -> AutoTokenizer:
    if tokenizer_name in _tokenizer_cache:
        return _tokenizer_cache[tokenizer_name]
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=settings.HF_TOKEN or None)
    except OSError as s:
        general_logger.warning(f"Could not load {tokenizer_name} tokenizer, defaulting to llama-tokenizer")
        general_logger.warning(s)
        tokenizer = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")
    _tokenizer_cache[tokenizer_name] = tokenizer
    return tokenizer


def local_tokenization(texts: str | list[str], tokenizer_name: str) -> int | list[int]:
    general_logger.debug("using local tokenization")
    text = ""
    if texts is str:
        text = texts
    tokenizer = _load_tokenizer(tokenizer_name)
    if text:
        encoded = tokenizer(text)
        token_amount = len(encoded["input_ids"])
        return token_amount
    else:
        encoded_texts = tokenizer(texts)
        return [len(token_ids) for token_ids in encoded_texts["input_ids"]]


def cache_entries(keys: list, values: list) -> None:
    redis_client = redis_manager.get_client()
    for key, value in zip(keys, values, strict=False):
        redis_client.set(key, value, settings.redis.CACHE_EXPIRE_TIME)


def get_cached_tokens(keys: list[str]) -> list[int | None]:
    redis_client = redis_manager.get_client()
    cached_tokens = []
    for key in keys:
        cached_value: str | None = redis_client.get(key)
        cached_tokens.append(cached_value)

    cached_tokens = [int(number) if number is not None else None for number in cached_tokens]
    return cached_tokens


@time_execution
def token_count(
    batch: list[str], api_type: str, api_url: str, local_tokenizer, prefer_local=True, api_auth=None
) -> list[int]:
    unique_texts = list(set(batch))
    cache_keys = [f"llm_{api_type}_{api_url} {text}" for text in unique_texts]
    cached_tokens = get_cached_tokens(cache_keys)
    tokens_dict = {}
    to_tokenize = []

    for text, tokens in zip(unique_texts, cached_tokens, strict=False):
        tokens_dict[text] = tokens
        if tokens is None:
            to_tokenize.append(text)

    if to_tokenize:
        if not prefer_local and api_type.lower() in ("koboldai", "koboldcpp", "tabby", "aphrodite", "genericoai"):
            new_tokens = remote_tokenization(to_tokenize, api_url, api_auth, api_type)
        else:
            new_tokens = local_tokenization(to_tokenize, local_tokenizer)

        if new_tokens is None:
            new_tokens = local_tokenization(to_tokenize, local_tokenizer)

        for text, tokens in zip(to_tokenize, new_tokens, strict=False):
            tokens_dict[text] = tokens

        new_keys = [f"llm_{api_type}_{api_url} {text}" for text in to_tokenize]
        cache_entries(new_keys, new_tokens)
    tokens = [tokens_dict[text] for text in batch]

    return tokens


def remote_tokenization(batch: list[str], api_url: str, api_auth: str, api_type: str) -> list[int] | None:
    general_logger.debug("using remote tokenization")
    tokenization_endpoint = ""
    request_jsons = []
    header = {"Authorization": f"Bearer {api_auth}"}
    token_amounts = []
    responses = []

    if api_type.lower() in ("koboldai", "koboldcpp"):
        tokenization_endpoint = urljoin(api_url, "/api/extra/tokencount")
        for text in batch:
            request_jsons.append({"prompt": text})

    elif api_type.lower() == "tabby":
        tokenization_endpoint = urljoin(api_url, "/v1/token/encode")
        for text in batch:
            request_jsons.append({"text": text})

    elif api_type.lower() in ("genericoai", "aphrodite"):
        tokenization_endpoint = urljoin(api_url, "/v1/tokenize")
        for text in batch:
            request_jsons.append({"prompt": text})

    for request_json in request_jsons:
        response = requests.post(url=tokenization_endpoint, json=request_json, headers=header)
        if response and response.status_code == 200:
            responses.append(response.json())
        else:
            general_logger.error("Request to tokenize failed, using local tokenizer fallback")
            return None

    if api_type.lower() in ("koboldai", "koboldcpp"):
        for response in responses:
            token_amounts.append(int(response["value"]))

    elif api_type.lower() == "tabby":
        for response in responses:
            token_amounts.append(int(response["length"]))

    elif api_type.lower() in ("genericoai", "aphrodite"):
        for response in responses:
            token_amounts.append(int(response["value"]))

    return token_amounts


def get_context_length(api_url: str) -> int:
    length_endpoint = urljoin(api_url, "/v1/config/max_context_length")
    kobold_response = requests.get(length_endpoint)
    value = int(kobold_response.json()["value"])
    return value


_KOBOLD_BACKENDS = ("koboldai", "koboldcpp")


def is_chat_mode(api_settings: ApiSettings) -> bool:
    """Whether generate_text will use the chat-completions endpoint for these settings.

    Kobold backends have no chat-completions endpoint, so chat mode always falls back to text
    there regardless of api_mode.
    """
    return api_settings.api_mode == "chat" and api_settings.backend.lower() not in _KOBOLD_BACKENDS


def get_model_name(api_url: str, api_key, api_type):
    if api_type.lower() == "tabby":
        model_endpoint = urljoin(api_url, "/v1/model")
        response = requests.get(model_endpoint, headers={"Authorization": f"Bearer {api_key}"})
        model_name = response.json()["id"]
    else:
        model_endpoint = urljoin(api_url, "/v1/models")
        response = requests.get(model_endpoint, headers={"Authorization": f"Bearer {api_key}"})
        model_name = response.json()["data"][0]["id"]
    return model_name


def generate_text(
    prompt: str | list[dict],
    params: dict,
    api_settings: ApiSettings,
    max_retries: int = 50,
    retry_interval: int = 1,
) -> GenerationResult:
    api_type = api_settings.backend
    api_url = api_settings.url
    api_key = api_settings.auth_key
    chat_mode = is_chat_mode(api_settings)

    if api_settings.api_mode == "chat" and not chat_mode:
        general_logger.warning(f"{api_type} does not support chat completions, falling back to text completion")

    if api_type.lower() in _KOBOLD_BACKENDS:
        request_body = {"prompt": prompt}
        request_body.update(params)
        endpoint = urljoin(api_url, "/api/v1/generate")
    elif chat_mode:
        if api_settings.model:
            model_name = api_settings.model
        else:
            model_name = get_model_name(api_url, api_key, api_type)
        chat_params = {k: v for k, v in params.items() if k not in _TEXT_ONLY_PARAM_KEYS}
        if api_settings.thinking_budget > 0 and "max_tokens" in chat_params:
            chat_params["max_tokens"] = chat_params["max_tokens"] + api_settings.thinking_budget
        if api_settings.reasoning_effort:
            chat_params["reasoning_effort"] = api_settings.reasoning_effort
        if api_settings.chat_template_kwargs:
            chat_params["chat_template_kwargs"] = api_settings.chat_template_kwargs
        request_body = {"model": model_name, "messages": prompt}
        request_body.update(chat_params)
        endpoint = urljoin(api_url, "/v1/chat/completions")
    else:
        request_body = {"prompt": prompt}
        if api_settings.model:
            model_name = api_settings.model
        else:
            model_name = get_model_name(api_url, api_key, api_type)
        request_body["model"] = model_name
        request_body.update(params)
        endpoint = urljoin(api_url, "/v1/completions")

    response = None
    for retry_num in range(max_retries + 1):
        response = requests.post(endpoint, json=request_body, headers={"Authorization": f"Bearer {api_key}"})
        if response.status_code == 200:
            break
        else:
            general_logger.warning(
                f"API returned status code {response.status_code}, "
                f"retrying in {retry_interval}s ({retry_num}/{max_retries})"
            )
            time.sleep(retry_interval)

    if response is None:
        raise Exception("Could not generate text, request was not made to the api")
    if response.status_code != 200:
        raise Exception("Could not generate text, max retries exceeded")

    response_json = response.json()
    reasoning = ""
    if api_type.lower() in _KOBOLD_BACKENDS:
        generated_text = response_json["results"][0]["text"]
    elif chat_mode:
        message = response_json["choices"][0]["message"]
        generated_text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    else:
        generated_text = response_json["choices"][0]["text"]

    if not reasoning:
        generated_text, split_out_reasoning = split_reasoning(
            generated_text,
            api_settings.strip_reasoning,
            api_settings.reasoning_start_token,
            api_settings.reasoning_end_token,
        )
        reasoning = split_out_reasoning
    elif api_settings.strip_reasoning:
        generated_text = generated_text.strip()

    return GenerationResult(text=generated_text, reasoning=reasoning, request_body=request_body)
