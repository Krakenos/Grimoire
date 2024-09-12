import json

import numpy as np
from sentence_transformers import SentenceTransformer
from torch import Tensor

from grimoire.common.redis import redis_manager
from grimoire.core.settings import settings

embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)


def vectorize_texts(texts: str | list[str]) -> Tensor | np.ndarray:
    embeddings = embedding_model.encode(texts, normalize_embeddings=True)
    return embeddings


def get_similarity(queries: list[str], texts: list[str]) -> np.ndarray:
    embed_1 = get_text_embeddings(queries)
    embed_2 = get_text_embeddings(texts)
    return embed_1 @ embed_2.T


def get_text_embeddings(texts: str | list) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    # get redis cached
    redis_client = redis_manager.get_client()
    redis_keys = [f"VECTOR_EMBEDDING_{text}" for text in texts]
    cached_entries = []
    embedding_dict = {}

    for key, text in zip(redis_keys, texts, strict=True):
        cached_value = redis_client.get(key)
        if cached_value is not None:
            cached_list = json.loads(cached_value)
            embedd = np.array(cached_list)
            cached_entries.append(embedd)
            embedding_dict[text] = embedd
        else:
            embedding_dict[text] = None
            cached_entries.append(None)

    to_vectorize = [text for text, embedding in zip(texts, cached_entries, strict=True) if cached_entries is None]
    new_embeddings = vectorize_texts(to_vectorize)

    # set new redis cache
    for text, embedding in zip(to_vectorize, new_embeddings, strict=True):
        key = f"VECTOR_EMBEDDING_{text}"
        redis_client.set(key, json.dumps(embedding), settings.redis.CACHE_EXPIRE_TIME)

    for text, embedding in zip(to_vectorize, new_embeddings, strict=True):
        embedding_dict[text] = embedding

    return np.array([embedding_dict[text] for text in texts])
