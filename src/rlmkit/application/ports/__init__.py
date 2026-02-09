"""Port interfaces (Protocol classes) for dependency inversion.

Ports define the contracts that infrastructure adapters must satisfy.
They depend only on the domain layer and DTOs.
"""

from .llm_port import LLMPort
from .sandbox_port import SandboxPort
from .storage_port import StoragePort
from .embedding_port import EmbeddingPort

__all__ = [
    "LLMPort",
    "SandboxPort",
    "StoragePort",
    "EmbeddingPort",
]
