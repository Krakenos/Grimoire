import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

from grimoire.common.loggers import general_logger
from grimoire.common.redis import redis_manager
from grimoire.common.utils import time_execution
from grimoire.core.settings import settings


def reinit_rope_buffers(model: torch.nn.Module) -> int:
    """Re-materialize non-persistent rotary/position buffers after loading.

    `transformers` loads models on the meta device and only fills persistent
    (checkpoint) tensors. The gte `new_impl` model keeps `position_ids`,
    `inv_freq`, and the rotary `cos_cached`/`sin_cached` as non-persistent
    buffers built in `__init__`, so on the meta-init path they end up as
    uninitialized memory: garbage rotary frequencies (silently wrong
    embeddings) and out-of-range `position_ids` that crash indexing with a
    device-side assert / IndexError. These buffers are deterministic functions
    of the (preserved) plain attributes, so we just recompute them.

    Returns the number of modules that were fixed.
    """
    fixed = 0
    for module in model.modules():
        if torch.is_tensor(getattr(module, "position_ids", None)):
            n = module.position_ids.shape[0]
            module.position_ids.copy_(torch.arange(n, device=module.position_ids.device))
            fixed += 1
        if hasattr(module, "_set_cos_sin_cache") and torch.is_tensor(getattr(module, "inv_freq", None)):
            device = module.inv_freq.device
            dtype = module.cos_cached.dtype
            base_inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2).float().to(device) / module.dim))
            module.register_buffer("inv_freq", base_inv_freq, persistent=False)
            module._set_cos_sin_cache(int(module.max_seq_len_cached), device, dtype)
            fixed += 1
    return fixed


def verify_rope_buffers(model: torch.nn.Module) -> None:
    """Fail loudly at startup if any rotary/position buffer is still garbage.

    The meta-init bug (see `reinit_rope_buffers`) corrupts embeddings *silently*
    -- you cannot tell from the output vectors that they are wrong, you just get
    degraded retrieval and the occasional out-of-bounds crash. This turns that
    failure mode into a hard startup error, so a future `transformers` change or
    embedding-model swap that reintroduces it is caught before a single bad
    vector is written to the database, rather than discovered months later.
    """
    for name, module in model.named_modules():
        if torch.is_tensor(getattr(module, "position_ids", None)):
            pid = module.position_ids
            expected = torch.arange(pid.shape[0], device=pid.device)
            if not torch.equal(pid, expected):
                raise RuntimeError(
                    f"Embedding model buffer '{name}.position_ids' is not a valid arange "
                    f"(got {pid[:5].tolist()}...). The model loaded with uninitialized rotary "
                    f"buffers and would produce corrupt embeddings. Refusing to start."
                )
        if hasattr(module, "_set_cos_sin_cache") and torch.is_tensor(getattr(module, "cos_cached", None)):
            cos0 = module.cos_cached[0].float()
            sin0 = module.sin_cached[0].float()
            inv_freq = module.inv_freq.float()
            ok = (
                torch.allclose(cos0, torch.ones_like(cos0), atol=1e-2)
                and torch.allclose(sin0, torch.zeros_like(sin0), atol=1e-2)
                and bool(torch.isfinite(inv_freq).all())
                and float(inv_freq.abs().min()) > 0.0
            )
            if not ok:
                raise RuntimeError(
                    f"Embedding model rotary cache '{name}' is invalid "
                    f"(cos[0][:3]={cos0[:3].tolist()}, inv_freq[:3]={inv_freq[:3].tolist()}). "
                    f"Expected cos[0]~=1, sin[0]~=0, finite nonzero inv_freq. Refusing to start."
                )


# `code_revision` pins the trust_remote_code modeling.py, which lives in a separate repo
# (Alibaba-NLP/new-impl) that `revision` does not cover. See settings for details.
_model_kwargs = {"code_revision": settings.EMBEDDING_MODEL_CODE_REVISION}

if not settings.prefer_gpu:
    embedding_model = SentenceTransformer(
        settings.EMBEDDING_MODEL,
        revision=settings.EMBEDDING_MODEL_REVISION,
        trust_remote_code=True,
        device="cpu",
        model_kwargs=_model_kwargs,
    )
    general_logger.info(f"Running embedding model {settings.EMBEDDING_MODEL} on CPU")
else:
    if not torch.cuda.is_available():
        raise OSError("CUDA is not available")
    if torch.cuda.device_count() < 1:
        raise OSError("No GPU available")

    embedding_model = SentenceTransformer(
        settings.EMBEDDING_MODEL,
        revision=settings.EMBEDDING_MODEL_REVISION,
        trust_remote_code=True,
        model_kwargs=_model_kwargs,
    )
    general_logger.info(f"Running embedding model {settings.EMBEDDING_MODEL} on GPU")

fixed_buffers = reinit_rope_buffers(embedding_model)
general_logger.info(f"Re-initialized {fixed_buffers} rotary/position buffer group(s) after model load")
verify_rope_buffers(embedding_model)


@time_execution
def vectorize_texts(texts: str | list[str]) -> Tensor | np.ndarray:
    embeddings = embedding_model.encode(texts, show_progress_bar=False)
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

    to_vectorize = [text for text, embedding in zip(texts, cached_entries, strict=True) if embedding is None]
    new_embeddings = vectorize_texts(to_vectorize)

    # set new redis cache
    for text, embedding in zip(to_vectorize, new_embeddings, strict=True):
        key = f"VECTOR_EMBEDDING_{text}"
        redis_client.set(key, json.dumps(embedding.tolist()), settings.redis.CACHE_EXPIRE_TIME)

    for text, embedding in zip(to_vectorize, new_embeddings, strict=True):
        embedding_dict[text] = embedding

    return np.array([embedding_dict[text] for text in texts])
