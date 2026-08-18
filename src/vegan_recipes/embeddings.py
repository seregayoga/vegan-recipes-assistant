from functools import lru_cache

import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

from .config import get_settings


class LocalEmbedder:
    def __init__(self):
        settings = get_settings()
        self.tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model, revision=settings.embedding_revision)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            settings.embedding_model, revision=settings.embedding_revision, export=False
        )

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            outputs = self.model(**inputs).last_hidden_state.detach().numpy()
            mask = inputs["attention_mask"].numpy()[..., None]
            pooled = (outputs * mask).sum(1) / np.maximum(mask.sum(1), 1)
            batches.append(pooled / np.linalg.norm(pooled, axis=1, keepdims=True))
        return np.concatenate(batches)


@lru_cache
def get_embedder() -> LocalEmbedder:
    return LocalEmbedder()
