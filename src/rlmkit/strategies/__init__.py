# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Strategy abstraction layer for multi-strategy evaluation."""

from .base import LLMStrategy, StrategyResult
from .direct import DirectStrategy
from .embeddings import EmbeddingProvider, OpenAIEmbedder
from .evaluator import EvaluationResult, MultiStrategyEvaluator
from .indexed_rag import IndexedRAGStrategy
from .rag import RAGStrategy
from .rlm_strategy import RLMStrategy
from .wiki_strategy import WikiRLMStrategy, WikiStrategy

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
    "WikiStrategy",
    "WikiRLMStrategy",
]
