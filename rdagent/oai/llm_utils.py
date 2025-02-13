from __future__ import annotations

import hashlib
from typing import Any, Type

import numpy as np

from rdagent.core.utils import import_class
from rdagent.oai.backend.base import APIBackend as BaseAPIBackend
from rdagent.oai.llm_conf import LLM_SETTINGS


def md5_hash(input_string: str) -> str:
    hash_md5 = hashlib.md5(usedforsecurity=False)
    input_bytes = input_string.encode("utf-8")
    hash_md5.update(input_bytes)
    return hash_md5.hexdigest()


def calculate_embedding_distance_between_str_list(
    source_str_list: list[str],
    target_str_list: list[str],
) -> list[list[float]]:
    if not source_str_list or not target_str_list:
        return [[]]

    embeddings = APIBackend().create_embedding(source_str_list + target_str_list)

    source_embeddings = embeddings[: len(source_str_list)]
    target_embeddings = embeddings[len(source_str_list) :]

    source_embeddings_np = np.array(source_embeddings)
    target_embeddings_np = np.array(target_embeddings)

    source_embeddings_np = source_embeddings_np / np.linalg.norm(source_embeddings_np, axis=1, keepdims=True)
    target_embeddings_np = target_embeddings_np / np.linalg.norm(target_embeddings_np, axis=1, keepdims=True)
    similarity_matrix = np.dot(source_embeddings_np, target_embeddings_np.T)

    return similarity_matrix.tolist()  # type: ignore[no-any-return]


def get_api_backend(*args: Any, **kwargs: Any) -> BaseAPIBackend:  # TODO: import it from base.py
    """
    get llm api backend based on settings dynamically.
    """
    api_backend_cls: Type[BaseAPIBackend] = import_class(LLM_SETTINGS.backend)
    return api_backend_cls(*args, **kwargs)


# Alias
APIBackend = get_api_backend
