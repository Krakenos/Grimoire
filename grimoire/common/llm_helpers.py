import asyncio
from urllib.parse import urljoin

import aiohttp
import redis
import requests
from transformers import AutoTokenizer

from grimoire.common.loggers import general_logger
from grimoire.common.utils import async_time_execution
from grimoire.core.settings import settings


# TODO old tokenization code, replace where it's used and remove
def count_context(text: str, api_type: str, api_url: str, api_auth=None) -> int:
    """
    Counts the token length of the text either through endpoint or with local tokenizer
    :param text:
    :param api_type:
    :param api_url:
    :param api_auth:
    :return:
    """
    if api_type.lower() in ("koboldai", "koboldcpp"):
        token_count_endpoint = urljoin(api_url, "/api/extra/tokencount")
        request_body = {"prompt": text}
        kobold_response = requests.post(token_count_endpoint, json=request_body)
        value = int(kobold_response.json()["value"])
        return value

    elif api_type.lower() == "tabby":
        tokenize_endpoint = urljoin(api_url, "/v1/token/encode")
        tokenize_json = {"text": text}
        tokenize_response = requests.post(
            tokenize_endpoint, json=tokenize_json, headers={"Authorization": f"Bearer {api_auth}"}
        )
        if tokenize_response.status_code == 200:
            tokenized = tokenize_response.json()
            return tokenized["length"]
    else:
        tokenize_endpoint = urljoin(api_url, "/v1/tokenize")
        tokenize_json = {"prompt": text}
        tokenize_response = requests.post(
            tokenize_endpoint, json=tokenize_json, headers={"Authorization": f"Bearer {api_auth}"}
        )
        if tokenize_response.status_code == 200:
            tokenized = tokenize_response.json()
            return tokenized["value"]
        general_logger.warning(f"Tokenize endpoint not found for {api_type}, proceeding to count based on tokenizer")
        model_name = get_model_name(api_url, api_auth, api_type)
        # Default to llama tokenizer if model tokenizer is not on huggingface
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError as s:
            general_logger.warning("Could not load model tokenizer, defaulting to llama-tokenizer")
            general_logger.warning(s)
            tokenizer = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")
        encoded = tokenizer(text)
        token_amount = len(encoded["input_ids"])
        return token_amount


def local_tokenization(texts: str | list[str], api_url: str, api_auth: str, api_type: str) -> int | list[int]:
    model_name = get_model_name(api_url, api_auth, api_type)
    text = ""
    if texts is str:
        text = texts
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError as s:
        general_logger.warning("Could not load model tokenizer, defaulting to llama-tokenizer")
        general_logger.warning(s)
        tokenizer = AutoTokenizer.from_pretrained("oobabooga/llama-tokenizer")
    if text:
        encoded = tokenizer(text)
        token_amount = len(encoded["input_ids"])
        return token_amount
    else:
        encoded_texts = [tokenizer(text) for text in texts]
        return [len(tokenized_text["input_ids"]) for tokenized_text in encoded_texts]


def cache_entries(keys: list, values: list) -> None:
    redis_client = redis.StrictRedis(host=settings["REDIS_HOST"], port=settings["REDIS_PORT"])
    for key, value in zip(keys, values, strict=False):
        redis_client.set(key, value, settings["CACHE_EXPIRE_TIME"])


def get_cached_tokens(keys: list[str]) -> list[int | None]:
    redis_client = redis.StrictRedis(host=settings["REDIS_HOST"], port=settings["REDIS_PORT"], decode_responses=True)
    cached_tokens = []
    for key in keys:
        cached_value: str | None = redis_client.get(key)
        cached_tokens.append(cached_value)

    cached_tokens = [int(number) if number is not None else None for number in cached_tokens]
    return cached_tokens


@async_time_execution
async def token_count(batch: list[str], api_type: str, api_url: str, api_auth=None) -> list[int]:
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
        if api_type.lower() in ("koboldai", "koboldcpp", "tabby", "aphrodite", "genericoai"):
            new_tokens = await remote_tokenization(to_tokenize, api_url, api_auth, api_type)
        else:
            new_tokens = local_tokenization(to_tokenize, api_url, api_auth, api_type)

        for text, tokens in zip(to_tokenize, new_tokens, strict=False):
            tokens_dict[text] = tokens

        new_keys = [f"llm_{api_type}_{api_url} {text}" for text in to_tokenize]
        cache_entries(new_keys, new_tokens)
    tokens = [tokens_dict[text] for text in batch]

    return tokens


async def remote_tokenization(batch: list[str], api_url: str, api_auth: str, api_type: str) -> list[int]:
    tasks = []
    tokenization_endpoint = ""
    request_jsons = []
    header = {"Authorization": f"Bearer {api_auth}"}
    token_amounts = []

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

    async with aiohttp.ClientSession() as session:
        for request_json in request_jsons:
            task = asyncio.ensure_future(post_json(request_json, header, tokenization_endpoint, session))
            tasks.append(task)
        responses = await asyncio.gather(*tasks)

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


async def post_json(json_content: dict, headers: dict, url: str, session: aiohttp.ClientSession) -> dict:
    async with session.post(url, json=json_content, headers=headers) as resp:
        return await resp.json()


def get_context_length(api_url: str) -> int:
    length_endpoint = urljoin(api_url, "/v1/config/max_context_length")
    kobold_response = requests.get(length_endpoint)
    value = int(kobold_response.json()["value"])
    return value


def get_model_name(api_url: str, api_key, api_type):
    if api_type == "tabby":
        model_endpoint = urljoin(api_url, "/v1/model")
        response = requests.get(model_endpoint, headers={"Authorization": f"Bearer {api_key}"})
        model_name = response.json()["id"]
    else:
        model_endpoint = urljoin(api_url, "/v1/models")
        response = requests.get(model_endpoint, headers={"Authorization": f"Bearer {api_key}"})
        model_name = response.json()["data"][0]["id"]
    return model_name


def generate_text(prompt: str, params: dict, api_type: str, api_url: str, api_key: str = None):
    if api_type.lower() in ("koboldai", "koboldcpp"):
        request_body = {"prompt": prompt}
        request_body.update(params)
        endpoint = urljoin(api_url, "/api/v1/generate")
        response = requests.post(endpoint, json=request_body)
        generated_text = response.json()["results"][0]["text"]
    else:
        request_body = {"prompt": prompt}
        model_name = get_model_name(api_url, api_key, api_type)
        request_body["model"] = model_name
        request_body.update(params)
        endpoint = urljoin(api_url, "/v1/completions")
        response = requests.post(endpoint, json=request_body, headers={"Authorization": f"Bearer {api_key}"})
        response_json = response.json()
        generated_text = response_json["choices"][0]["text"]
    return generated_text, request_body
