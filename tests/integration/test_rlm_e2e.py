"""End-to-end RLM pipeline integration tests.

These tests exercise the full RLM loop (query -> LLM -> code -> execution -> answer)
using MockLLMClient so they run without external API keys.
"""

import pytest

from rlmkit import RLM, MockLLMClient, RLMConfig, BudgetExceeded


pytestmark = pytest.mark.integration


class TestRLMEndToEnd:
    """Full RLM pipeline tests with mock LLM."""

    def test_simple_query_returns_response(self, rlm_instance, sample_document):
        """RLM returns a non-empty answer for a straightforward query."""
        result = rlm_instance.run(prompt=sample_document, query="Summarize the document")

        assert result.success
        assert result.answer != ""
        assert result.steps >= 1

    def test_multi_step_exploration_with_mock(
        self, rlm_multistep_instance, sample_document
    ):
        """RLM executes multiple code steps before reaching FINAL."""
        result = rlm_multistep_instance.run(
            prompt=sample_document,
            query="What are the important concepts?",
        )

        assert result.success
        assert result.steps == 3  # 2 code steps + 1 FINAL step
        assert "Found relevant information" in result.answer

    def test_rlm_produces_execution_trace(
        self, rlm_multistep_instance, sample_document
    ):
        """Execution trace contains assistant and execution entries in order."""
        result = rlm_multistep_instance.run(
            prompt=sample_document,
            query="Find key details",
        )

        assert len(result.trace) >= 3

        # First entry: assistant response containing code
        assert result.trace[0]["role"] == "assistant"
        # Second entry: execution result from running that code
        assert result.trace[1]["role"] == "execution"
        assert "raw_result" in result.trace[1]

        # Trace entries carry step numbers
        for entry in result.trace:
            assert "step" in entry

    def test_rlm_respects_max_steps(self, mock_llm_never_final, sample_document):
        """RLM raises BudgetExceeded when max_steps is reached without FINAL."""
        config = RLMConfig()
        config.execution.max_steps = 3

        rlm = RLM(client=mock_llm_never_final, config=config)

        with pytest.raises(BudgetExceeded, match="Maximum steps"):
            rlm.run(prompt=sample_document, query="Analyze this")

    def test_rlm_handles_llm_error_gracefully(self, mock_llm_failing, sample_document):
        """RLM returns a failed RLMResult when the LLM raises an exception."""
        rlm = RLM(client=mock_llm_failing)
        result = rlm.run(prompt=sample_document, query="Any question")

        assert not result.success
        assert result.error is not None
        assert "LLM service unavailable" in result.error

    def test_rlm_with_large_document(self, mock_llm_multistep, large_document):
        """RLM handles a ~10K character document without issues."""
        config = RLMConfig()
        config.execution.max_steps = 10

        rlm = RLM(client=mock_llm_multistep, config=config)
        result = rlm.run(prompt=large_document, query="What topics are covered?")

        assert result.success
        assert result.steps >= 1

    def test_peek_returns_correct_substring(self, sample_document):
        """Code using peek() returns the correct slice of the document."""
        client = MockLLMClient([
            "```python\nfirst_50 = peek(0, 50)\nprint(first_50)\n```",
            "FINAL: Checked the beginning",
        ])
        rlm = RLM(client=client)
        result = rlm.run(prompt=sample_document, query="Read the start")

        assert result.success

        exec_entries = [t for t in result.trace if t["role"] == "execution"]
        assert len(exec_entries) == 1
        # peek(0, 50) returns the first 50 characters of sample_document
        assert sample_document[:50] in exec_entries[0]["content"]

    def test_grep_finds_pattern(self, sample_document):
        """Code using grep() locates patterns in the document."""
        client = MockLLMClient([
            "```python\nresults = grep(r'budget')\nprint(results)\n```",
            "FINAL: Found budget references",
        ])
        rlm = RLM(client=client)
        result = rlm.run(prompt=sample_document, query="Find budget info")

        assert result.success

        exec_entries = [t for t in result.trace if t["role"] == "execution"]
        assert len(exec_entries) == 1
        # The word "budget" appears in the sample_document (lowercase b in "budget")
        # grep should find it
        assert "budget" in exec_entries[0]["content"].lower() or "Budget" in exec_entries[0]["content"]

    def test_variables_persist_between_steps(self, sample_document):
        """Variables defined in one code step are available in the next."""
        client = MockLLMClient([
            "```python\nmy_var = peek(0, 10)\n```",
            "```python\nprint(my_var.upper())\n```",
            "FINAL: Variable persisted",
        ])
        rlm = RLM(client=client)
        result = rlm.run(prompt=sample_document, query="Test persistence")

        assert result.success
        assert result.steps == 3

        exec_entries = [t for t in result.trace if t["role"] == "execution"]
        assert len(exec_entries) == 2
        # Second execution should print the uppercased first 10 chars
        expected_upper = sample_document[:10].upper()
        assert expected_upper in exec_entries[1]["content"]

    def test_code_execution_error_does_not_crash(self, sample_document):
        """A division-by-zero in generated code is handled, then FINAL accepted."""
        client = MockLLMClient([
            "```python\nx = 1 / 0\n```",
            "FINAL: Handled the error",
        ])
        rlm = RLM(client=client)
        result = rlm.run(prompt=sample_document, query="Trigger error")

        assert result.success
        assert result.answer == "Handled the error"
