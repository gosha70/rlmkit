"""Shared fixtures for integration tests."""

import tempfile
from pathlib import Path

import pytest

from rlmkit import RLM, MockLLMClient, RLMConfig, BudgetLimits, BudgetTracker, CostTracker
from rlmkit.config import ExecutionConfig


# ---------------------------------------------------------------------------
# Mock LLM client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_llm_client():
    """MockLLMClient that returns a single FINAL answer."""
    return MockLLMClient(["FINAL: This is the mock answer"])


@pytest.fixture()
def mock_llm_multistep():
    """MockLLMClient that performs two code steps then returns FINAL."""
    return MockLLMClient([
        "```python\nresult = peek(0, 50)\nprint(result)\n```",
        "```python\nmatches = grep(r'important')\nprint(len(matches))\n```",
        "FINAL: Found relevant information in the document",
    ])


@pytest.fixture()
def mock_llm_never_final():
    """MockLLMClient that always returns code and never FINAL (for budget tests)."""
    return MockLLMClient(["```python\nx = peek(0, 10)\nprint(x)\n```"])


@pytest.fixture()
def mock_llm_failing():
    """LLM client that raises on every call."""

    class _FailingClient:
        def complete(self, messages):
            raise RuntimeError("LLM service unavailable")

    return _FailingClient()


# ---------------------------------------------------------------------------
# Content fixtures
# ---------------------------------------------------------------------------

_SAMPLE_DOCUMENT = (
    "The Recursive Language Model (RLM) paradigm represents a significant advancement "
    "in how large language models interact with extensive textual content. Unlike "
    "traditional approaches that attempt to fit entire documents into a context window, "
    "RLM treats the content as a variable in a programming environment and allows the "
    "LLM to write code that explores the content programmatically.\n\n"
    "This approach has several important advantages. First, it removes the context-window "
    "limitation entirely because the content is never sent to the model in full. Instead, "
    "the model issues targeted queries using tools like peek, grep, and chunk to inspect "
    "only the portions it needs. Second, the recursive nature allows sub-models to be "
    "spawned for focused sub-tasks, enabling divide-and-conquer strategies on complex "
    "documents.\n\n"
    "Budget management is a critical component of the RLM framework. Without proper "
    "budget enforcement, an RLM agent could enter infinite loops or consume excessive "
    "API resources. The budget tracker monitors steps taken, tokens consumed, cost "
    "incurred, wall-clock time elapsed, and recursion depth. When any limit is exceeded, "
    "execution halts gracefully with a BudgetExceeded exception.\n\n"
    "The system supports multiple LLM providers including OpenAI, Anthropic, Ollama, "
    "LM Studio, and vLLM. Each provider is accessed through a common interface that "
    "requires only a complete() method accepting a message list and returning a string. "
    "This makes it trivial to swap providers or use mock clients for testing.\n\n"
    "Multi-model cost optimization is another key feature. By using a powerful model "
    "(e.g., GPT-4) for root-level reasoning and a cheaper model (e.g., GPT-4o-mini) "
    "for sub-agent exploration tasks, users can reduce costs by 50-80% while maintaining "
    "quality on the final answer."
)


@pytest.fixture()
def sample_document():
    """Multi-paragraph text fixture (~1500 chars) covering RLM concepts."""
    return _SAMPLE_DOCUMENT


@pytest.fixture()
def large_document():
    """Longer text fixture (~10000 chars) for testing RLM on bigger content."""
    paragraphs = []
    topics = [
        (
            "Natural language processing",
            "has evolved dramatically over the past decade. Transformer architectures "
            "replaced recurrent networks and enabled models to attend to all positions "
            "in an input sequence simultaneously. This parallelism improved both training "
            "speed and model quality.",
        ),
        (
            "Tokenization strategies",
            "play a crucial role in model performance. Byte-pair encoding, WordPiece, "
            "and SentencePiece are popular subword tokenization methods that balance "
            "vocabulary size with the ability to represent rare words.",
        ),
        (
            "Retrieval-augmented generation",
            "combines the strengths of parametric knowledge stored in model weights with "
            "non-parametric knowledge retrieved from external documents. This approach "
            "reduces hallucination and improves factual accuracy.",
        ),
        (
            "Prompt engineering",
            "has become an important discipline. Techniques such as chain-of-thought "
            "prompting, few-shot examples, and system instructions allow practitioners "
            "to guide model behavior without retraining.",
        ),
        (
            "Code generation by LLMs",
            "enables novel interaction patterns. When an LLM can write and execute code, "
            "it gains access to precise computation, data manipulation, and tool use "
            "that pure text generation cannot achieve.",
        ),
    ]
    for i in range(20):
        title, body = topics[i % len(topics)]
        paragraphs.append(f"Section {i + 1}: {title}\n{body}\n")
    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def budget_config():
    """BudgetLimits with conservative defaults for testing."""
    return BudgetLimits(
        max_steps=5,
        max_tokens=5000,
        max_cost=1.0,
        max_time_seconds=30.0,
        max_recursion_depth=3,
    )


@pytest.fixture()
def rlm_config_strict():
    """RLMConfig with a low step limit for fast, bounded tests."""
    cfg = RLMConfig()
    cfg.execution.max_steps = 5
    return cfg


# ---------------------------------------------------------------------------
# RLM instance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rlm_instance(mock_llm_client):
    """Configured RLM instance with a simple mock LLM."""
    return RLM(client=mock_llm_client)


@pytest.fixture()
def rlm_multistep_instance(mock_llm_multistep, rlm_config_strict):
    """RLM instance wired to a multi-step mock LLM with strict config."""
    return RLM(client=mock_llm_multistep, config=rlm_config_strict)


# ---------------------------------------------------------------------------
# Storage / temp directory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def temp_db(tmp_path):
    """Temporary SQLite database path inside pytest's tmp_path."""
    return tmp_path / "test_rlmkit.db"
