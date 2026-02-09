"""Tests for RecursiveController (deep recursion support)."""

import pytest

from rlmkit import BudgetExceeded, BudgetLimits, BudgetTracker, MockLLMClient, RLM, RLMConfig
from rlmkit.config import ExecutionConfig
from rlmkit.core.recursion import RecursiveController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_factory(client, max_steps=16):
    """Return a controller factory that builds an RLM for the given depth."""

    def factory(depth=0):
        config = RLMConfig(execution=ExecutionConfig(max_steps=max_steps))
        return RLM(client=client, config=config)

    return factory


# ---------------------------------------------------------------------------
# Basic depth tests
# ---------------------------------------------------------------------------

class TestRecursiveControllerBasics:
    """Core depth-management behaviour."""

    def test_depth_1_subcall_succeeds(self):
        """A single subcall at depth=1 succeeds and returns the child answer."""
        client = MockLLMClient(["FINAL: child answer"])
        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=5,
        )

        answer = rc.execute_subcall(
            content="some content",
            query="summarise",
            depth=0,
        )

        assert answer == "child answer"
        assert len(rc.recursion_trace) == 1
        assert rc.recursion_trace[0]["depth"] == 1
        assert rc.recursion_trace[0]["success"] is True

    def test_depth_2_nested_subcall(self):
        """Two levels of nesting (root -> child -> grandchild)."""
        # The MockLLMClient is shared.  Call sequence:
        #   child (depth=1): code that calls subcall -> triggers depth=2
        #   grandchild (depth=2): returns FINAL immediately
        #   child (depth=1): gets answer, returns FINAL
        client = MockLLMClient([
            "```python\nresult = subcall('inner', 'q2')\nprint(result)\n```",
            "FINAL: grandchild",
            "FINAL: child got grandchild",
        ])

        tracker = BudgetTracker(BudgetLimits(max_recursion_depth=5))

        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=5,
            budget_tracker=tracker,
        )

        answer = rc.execute_subcall(
            content="outer", query="q1", depth=0,
        )

        assert "grandchild" in answer
        # After completion, recursion depth should be back to 0
        assert tracker.recursion_depth == 0

    def test_depth_3_triple_nesting(self):
        """Three levels: root -> d1 -> d2 -> d3."""
        client = MockLLMClient([
            # d1: calls subcall to go to d2
            "```python\nr = subcall('d2_content', 'd2_query')\nprint(r)\n```",
            # d2: calls subcall to go to d3
            "```python\nr = subcall('d3_content', 'd3_query')\nprint(r)\n```",
            # d3: returns FINAL
            "FINAL: d3 answer",
            # d2: returns with d3 answer
            "FINAL: d2 got d3 answer",
            # d1: returns with d2 answer
            "FINAL: d1 got d2 got d3 answer",
        ])

        tracker = BudgetTracker(BudgetLimits(max_recursion_depth=5))
        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=5,
            budget_tracker=tracker,
        )

        answer = rc.execute_subcall(content="root", query="go deep", depth=0)

        assert "d3 answer" in answer
        assert tracker.recursion_depth == 0

    def test_max_depth_enforced(self):
        """BudgetExceeded raised when recursion exceeds max_depth."""
        client = MockLLMClient(["FINAL: should not reach"])
        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=2,
        )

        # depth=2 means child_depth=3 > max_depth=2
        with pytest.raises(BudgetExceeded, match="Maximum recursion depth"):
            rc.execute_subcall(content="x", query="q", depth=2)

    def test_max_depth_boundary(self):
        """Subcall at exactly max_depth - 1 succeeds; at max_depth fails."""
        client = MockLLMClient(["FINAL: ok"])

        # max_depth=2: depth 0->1 ok, 1->2 ok, 2->3 fail
        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=2,
        )

        # depth=1 -> child_depth=2 == max_depth: should succeed
        answer = rc.execute_subcall(content="x", query="q", depth=1)
        assert answer == "ok"

        # depth=2 -> child_depth=3 > max_depth: should fail
        with pytest.raises(BudgetExceeded):
            rc.execute_subcall(content="x", query="q", depth=2)


class TestRecursiveControllerBudgetSharing:
    """Verify the shared budget tracker across recursion levels."""

    def test_shared_tracker_accumulates_recursion(self):
        """The shared tracker correctly increments/decrements depth."""
        client = MockLLMClient(["FINAL: done"])
        tracker = BudgetTracker(BudgetLimits(max_recursion_depth=5))

        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=5,
            budget_tracker=tracker,
        )

        assert tracker.recursion_depth == 0

        rc.execute_subcall(content="c", query="q", depth=0)
        # After returning, depth is back to 0
        assert tracker.recursion_depth == 0

    def test_tracker_depth_returns_to_zero_after_error(self):
        """Even if the child fails, recursion depth is decremented."""

        class _FailClient:
            def complete(self, messages):
                raise RuntimeError("boom")

        tracker = BudgetTracker(BudgetLimits(max_recursion_depth=5))
        rc = RecursiveController(
            controller_factory=lambda depth=0: RLM(client=_FailClient()),
            max_depth=5,
            budget_tracker=tracker,
        )

        # Should not raise (error is caught inside RLM.run)
        answer = rc.execute_subcall(content="c", query="q", depth=0)
        assert "Error" in answer
        assert tracker.recursion_depth == 0


class TestRecursiveControllerTrace:
    """Verify that the recursion trace is populated."""

    def test_trace_records_subcall(self):
        """Each subcall appends an entry to recursion_trace."""
        client = MockLLMClient(["FINAL: a1"])
        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=5,
        )

        rc.execute_subcall(content="c1", query="q1", depth=0)
        rc.execute_subcall(content="c2", query="q2", depth=0)

        assert len(rc.recursion_trace) == 2
        assert rc.recursion_trace[0]["query"] == "q1"
        assert rc.recursion_trace[1]["query"] == "q2"

    def test_trace_includes_depth(self):
        """Trace entries include the child depth."""
        client = MockLLMClient(["FINAL: ok"])
        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=5,
        )

        rc.execute_subcall(content="c", query="q", depth=2)

        assert rc.recursion_trace[0]["depth"] == 3


class TestCreateSubcallFunction:
    """Verify the convenience subcall function factory."""

    def test_returns_callable(self):
        """create_subcall_function returns a callable."""
        client = MockLLMClient(["FINAL: ok"])
        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=5,
        )
        fn = rc.create_subcall_function(parent_depth=0)
        assert callable(fn)

    def test_callable_executes_subcall(self):
        """The returned callable delegates to execute_subcall."""
        client = MockLLMClient(["FINAL: via callable"])
        rc = RecursiveController(
            controller_factory=_make_factory(client),
            max_depth=5,
        )
        fn = rc.create_subcall_function(parent_depth=0)
        answer = fn(content="c", query="q")

        assert answer == "via callable"
        assert len(rc.recursion_trace) == 1
