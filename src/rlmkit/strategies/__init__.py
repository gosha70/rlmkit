# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Strategy abstraction layer for multi-strategy evaluation."""

from .base import LLMStrategy, StrategyResult
from .direct import DirectStrategy
from .rlm_strategy import RLMStrategy
from .embeddings import EmbeddingProvider, OpenAIEmbedder
from .rag import RAGStrategy
from .evaluator import MultiStrategyEvaluator, EvaluationResult
from .indexed_rag import IndexedRAGStrategy

__all__ = [
    "LLMStrategy",
    "StrategyResult",
    "DirectStrategy",
    "RLMStrategy",
    "EmbeddingProvider",
    "OpenAIEmbedder",
    "RAGStrategy",
    "IndexedRAGStrategy",
    "MultiStrategyEvaluator",
    "EvaluationResult",
]
