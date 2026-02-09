"""Tests for the domain layer: entities, value objects, events, and exceptions.

All domain objects are plain Python dataclasses with no external dependencies.
These tests verify correctness, immutability, and serialization behaviour.
"""

import pytest

from rlmkit.domain.entities import (
    BudgetConfig,
    BudgetState,
    ExecutionTrace,
    Query,
    Response,
    TraceStep,
)
from rlmkit.domain.value_objects import (
    Cost,
    ModelId,
    ProviderId,
    RecursionDepth,
    TokenCount,
)
from rlmkit.domain.events import (
    BudgetExceeded,
    ExecutionCompleted,
    ExecutionStarted,
    StepCompleted,
)
from rlmkit.domain.exceptions import (
    BudgetExceededError,
    ConfigurationError,
    DomainError,
    ExecutionFailedError,
    ParseFailedError,
    SecurityViolationError,
)


# ---------------------------------------------------------------------------
# Entity tests: Query
# ---------------------------------------------------------------------------


class TestQuery:
    """Tests for Query entity."""

    def test_minimal_construction(self):
        q = Query(content="hello", question="what?")
        assert q.content == "hello"
        assert q.question == "what?"
        assert q.mode == "auto"
        assert q.metadata == {}

    def test_custom_mode(self):
        q = Query(content="c", question="q", mode="rlm")
        assert q.mode == "rlm"

    def test_metadata(self):
        q = Query(content="c", question="q", metadata={"source": "test"})
        assert q.metadata["source"] == "test"


# ---------------------------------------------------------------------------
# Entity tests: Response
# ---------------------------------------------------------------------------


class TestResponse:
    """Tests for Response entity."""

    def test_minimal_construction(self):
        r = Response(answer="42", mode_used="direct")
        assert r.answer == "42"
        assert r.mode_used == "direct"
        assert r.success is True
        assert r.error is None
        assert r.steps == 0

    def test_failure_response(self):
        r = Response(answer="", mode_used="rlm", success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"


# ---------------------------------------------------------------------------
# Entity tests: TraceStep
# ---------------------------------------------------------------------------


class TestTraceStep:
    """Tests for TraceStep entity."""

    def test_minimal_construction(self):
        step = TraceStep(index=0, action_type="inspect")
        assert step.index == 0
        assert step.action_type == "inspect"
        assert step.tokens_used == 0
        assert step.recursion_depth == 0
        assert step.code is None

    def test_full_construction(self):
        step = TraceStep(
            index=3,
            action_type="subcall",
            code="print(1)",
            output="1",
            tokens_used=150,
            timestamp=1000.0,
            recursion_depth=2,
            cost=0.005,
            duration=1.5,
            model="gpt-4o",
            raw_response="raw text",
            error=None,
        )
        assert step.tokens_used == 150
        assert step.recursion_depth == 2
        assert step.model == "gpt-4o"

    def test_to_dict(self):
        step = TraceStep(index=0, action_type="final", tokens_used=100, cost=0.01)
        d = step.to_dict()
        assert d["index"] == 0
        assert d["action_type"] == "final"
        assert d["tokens_used"] == 100
        assert d["cost"] == 0.01
        assert d["code"] is None


# ---------------------------------------------------------------------------
# Entity tests: ExecutionTrace
# ---------------------------------------------------------------------------


class TestExecutionTrace:
    """Tests for ExecutionTrace entity."""

    def test_empty_trace(self):
        trace = ExecutionTrace()
        assert trace.steps == []
        assert trace.total_tokens == 0
        assert trace.total_cost == 0.0
        assert trace.max_depth == 0

    def test_add_step(self):
        trace = ExecutionTrace()
        trace.add_step(TraceStep(index=0, action_type="inspect", tokens_used=10))
        trace.add_step(TraceStep(index=1, action_type="final", tokens_used=20))
        assert len(trace.steps) == 2
        assert trace.total_tokens == 30

    def test_total_cost(self):
        trace = ExecutionTrace()
        trace.add_step(TraceStep(index=0, action_type="inspect", cost=0.01))
        trace.add_step(TraceStep(index=1, action_type="final", cost=0.02))
        assert trace.total_cost == pytest.approx(0.03)

    def test_max_depth(self):
        trace = ExecutionTrace()
        trace.add_step(TraceStep(index=0, action_type="inspect", recursion_depth=0))
        trace.add_step(TraceStep(index=1, action_type="subcall", recursion_depth=2))
        trace.add_step(TraceStep(index=2, action_type="final", recursion_depth=1))
        assert trace.max_depth == 2

    def test_to_dict(self):
        trace = ExecutionTrace(start_time=1000.0, end_time=1005.0)
        trace.add_step(TraceStep(index=0, action_type="final", tokens_used=50, cost=0.01))
        d = trace.to_dict()
        assert d["start_time"] == 1000.0
        assert d["end_time"] == 1005.0
        assert d["total_tokens"] == 50
        assert d["total_cost"] == pytest.approx(0.01)
        assert d["max_depth"] == 0
        assert len(d["steps"]) == 1


# ---------------------------------------------------------------------------
# Entity tests: BudgetConfig & BudgetState
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    """Tests for BudgetConfig entity."""

    def test_defaults_are_none(self):
        cfg = BudgetConfig()
        assert cfg.max_steps is None
        assert cfg.max_tokens is None
        assert cfg.max_cost is None
        assert cfg.max_time_seconds is None
        assert cfg.max_recursion_depth is None

    def test_custom_limits(self):
        cfg = BudgetConfig(max_steps=10, max_cost=5.0, max_recursion_depth=3)
        assert cfg.max_steps == 10
        assert cfg.max_cost == 5.0
        assert cfg.max_recursion_depth == 3


class TestBudgetState:
    """Tests for BudgetState entity."""

    def test_initial_state(self):
        state = BudgetState()
        assert state.steps == 0
        assert state.total_tokens == 0
        assert state.cost == 0.0

    def test_total_tokens_property(self):
        state = BudgetState(input_tokens=100, output_tokens=50)
        assert state.total_tokens == 150

    def test_is_within_no_limits(self):
        state = BudgetState(steps=100, input_tokens=10000)
        config = BudgetConfig()  # all None => unlimited
        assert state.is_within(config) is True

    def test_is_within_steps_limit(self):
        config = BudgetConfig(max_steps=5)
        assert BudgetState(steps=4).is_within(config) is True
        assert BudgetState(steps=5).is_within(config) is False
        assert BudgetState(steps=6).is_within(config) is False

    def test_is_within_tokens_limit(self):
        config = BudgetConfig(max_tokens=1000)
        assert BudgetState(input_tokens=400, output_tokens=500).is_within(config) is True
        assert BudgetState(input_tokens=600, output_tokens=400).is_within(config) is False

    def test_is_within_cost_limit(self):
        config = BudgetConfig(max_cost=1.0)
        assert BudgetState(cost=0.99).is_within(config) is True
        assert BudgetState(cost=1.0).is_within(config) is False

    def test_is_within_time_limit(self):
        config = BudgetConfig(max_time_seconds=30.0)
        assert BudgetState(elapsed_seconds=29.0).is_within(config) is True
        assert BudgetState(elapsed_seconds=30.0).is_within(config) is False

    def test_is_within_recursion_depth_limit(self):
        config = BudgetConfig(max_recursion_depth=3)
        assert BudgetState(recursion_depth=2).is_within(config) is True
        assert BudgetState(recursion_depth=3).is_within(config) is False

    def test_is_within_multiple_limits(self):
        config = BudgetConfig(max_steps=10, max_cost=2.0)
        assert BudgetState(steps=5, cost=1.0).is_within(config) is True
        assert BudgetState(steps=10, cost=1.0).is_within(config) is False
        assert BudgetState(steps=5, cost=2.0).is_within(config) is False


# ---------------------------------------------------------------------------
# Value object tests: TokenCount
# ---------------------------------------------------------------------------


class TestTokenCount:
    """Tests for TokenCount value object (frozen dataclass)."""

    def test_construction(self):
        tc = TokenCount(input_tokens=100, output_tokens=50)
        assert tc.input_tokens == 100
        assert tc.output_tokens == 50

    def test_total_property(self):
        tc = TokenCount(input_tokens=100, output_tokens=50)
        assert tc.total == 150

    def test_default_zero(self):
        tc = TokenCount()
        assert tc.total == 0

    def test_add(self):
        a = TokenCount(input_tokens=10, output_tokens=5)
        b = TokenCount(input_tokens=20, output_tokens=15)
        c = a.add(b)
        assert c.input_tokens == 30
        assert c.output_tokens == 20
        assert c.total == 50

    def test_immutability(self):
        tc = TokenCount(input_tokens=10, output_tokens=5)
        with pytest.raises(AttributeError):
            tc.input_tokens = 99

    def test_to_dict(self):
        tc = TokenCount(input_tokens=10, output_tokens=5)
        d = tc.to_dict()
        assert d == {"input_tokens": 10, "output_tokens": 5, "total": 15}

    def test_equality(self):
        a = TokenCount(input_tokens=10, output_tokens=5)
        b = TokenCount(input_tokens=10, output_tokens=5)
        assert a == b

    def test_hash(self):
        a = TokenCount(input_tokens=10, output_tokens=5)
        b = TokenCount(input_tokens=10, output_tokens=5)
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Value object tests: Cost
# ---------------------------------------------------------------------------


class TestCost:
    """Tests for Cost value object."""

    def test_construction(self):
        c = Cost(input_cost=0.01, output_cost=0.03)
        assert c.input_cost == 0.01
        assert c.output_cost == 0.03

    def test_total(self):
        c = Cost(input_cost=0.01, output_cost=0.03)
        assert c.total == pytest.approx(0.04)

    def test_add(self):
        a = Cost(input_cost=0.01, output_cost=0.02)
        b = Cost(input_cost=0.005, output_cost=0.015)
        c = a.add(b)
        assert c.input_cost == pytest.approx(0.015)
        assert c.output_cost == pytest.approx(0.035)

    def test_to_dict(self):
        c = Cost(input_cost=1.0, output_cost=2.0)
        d = c.to_dict()
        assert d == {"input_cost": 1.0, "output_cost": 2.0, "total": 3.0}

    def test_immutability(self):
        c = Cost(input_cost=1.0, output_cost=2.0)
        with pytest.raises(AttributeError):
            c.input_cost = 5.0


# ---------------------------------------------------------------------------
# Value object tests: ModelId
# ---------------------------------------------------------------------------


class TestModelId:
    """Tests for ModelId value object."""

    def test_construction(self):
        m = ModelId(provider="openai", model_name="gpt-4o")
        assert m.provider == "openai"
        assert m.model_name == "gpt-4o"

    def test_display_name(self):
        m = ModelId(provider="anthropic", model_name="claude-3-opus")
        assert m.display_name == "anthropic/claude-3-opus"

    def test_str(self):
        m = ModelId(provider="openai", model_name="gpt-4o")
        assert str(m) == "openai/gpt-4o"

    def test_equality(self):
        a = ModelId(provider="openai", model_name="gpt-4o")
        b = ModelId(provider="openai", model_name="gpt-4o")
        assert a == b

    def test_inequality(self):
        a = ModelId(provider="openai", model_name="gpt-4o")
        b = ModelId(provider="openai", model_name="gpt-4o-mini")
        assert a != b


# ---------------------------------------------------------------------------
# Value object tests: ProviderId
# ---------------------------------------------------------------------------


class TestProviderId:
    """Tests for ProviderId value object."""

    def test_construction(self):
        p = ProviderId(name="openai")
        assert p.name == "openai"

    def test_str(self):
        p = ProviderId(name="anthropic")
        assert str(p) == "anthropic"

    def test_equality_with_provider_id(self):
        a = ProviderId(name="openai")
        b = ProviderId(name="openai")
        assert a == b

    def test_equality_with_string(self):
        p = ProviderId(name="openai")
        assert p == "openai"
        assert p != "anthropic"

    def test_hash(self):
        a = ProviderId(name="openai")
        b = ProviderId(name="openai")
        assert hash(a) == hash(b)
        # Can be used as dict key
        d = {a: "value"}
        assert d[b] == "value"


# ---------------------------------------------------------------------------
# Value object tests: RecursionDepth
# ---------------------------------------------------------------------------


class TestRecursionDepth:
    """Tests for RecursionDepth value object."""

    def test_defaults(self):
        rd = RecursionDepth()
        assert rd.current == 0
        assert rd.maximum == 5
        assert rd.is_exceeded is False

    def test_is_exceeded_at_boundary(self):
        rd = RecursionDepth(current=5, maximum=5)
        assert rd.is_exceeded is True

    def test_is_exceeded_below_max(self):
        rd = RecursionDepth(current=4, maximum=5)
        assert rd.is_exceeded is False

    def test_descend(self):
        rd = RecursionDepth(current=1, maximum=5)
        deeper = rd.descend()
        assert deeper.current == 2
        assert deeper.maximum == 5
        # Original unchanged
        assert rd.current == 1

    def test_descend_chain(self):
        rd = RecursionDepth(current=0, maximum=3)
        d1 = rd.descend()
        d2 = d1.descend()
        d3 = d2.descend()
        assert d3.current == 3
        assert d3.is_exceeded is True

    def test_to_dict(self):
        rd = RecursionDepth(current=2, maximum=5)
        d = rd.to_dict()
        assert d == {"current": 2, "maximum": 5, "is_exceeded": False}

    def test_immutability(self):
        rd = RecursionDepth(current=0, maximum=5)
        with pytest.raises(AttributeError):
            rd.current = 3


# ---------------------------------------------------------------------------
# Event tests
# ---------------------------------------------------------------------------


class TestStepCompleted:
    """Tests for StepCompleted event."""

    def test_construction(self):
        evt = StepCompleted(step_index=0, action_type="inspect")
        assert evt.step_index == 0
        assert evt.action_type == "inspect"
        assert evt.duration == 0.0
        assert evt.tokens_used == 0

    def test_full_construction(self):
        evt = StepCompleted(
            step_index=3,
            action_type="subcall",
            duration=1.5,
            tokens_used=200,
            recursion_depth=2,
            model="gpt-4o",
        )
        assert evt.recursion_depth == 2
        assert evt.model == "gpt-4o"

    def test_immutability(self):
        evt = StepCompleted(step_index=0, action_type="final")
        with pytest.raises(AttributeError):
            evt.step_index = 1


class TestBudgetExceededEvent:
    """Tests for BudgetExceeded event."""

    def test_construction(self):
        evt = BudgetExceeded(budget_type="steps", limit=10.0, current=11.0)
        assert evt.budget_type == "steps"
        assert evt.limit == 10.0
        assert evt.current == 11.0


class TestExecutionStarted:
    """Tests for ExecutionStarted event."""

    def test_construction(self):
        evt = ExecutionStarted(query="test", mode="rlm", content_length=5000)
        assert evt.query == "test"
        assert evt.mode == "rlm"
        assert evt.content_length == 5000


class TestExecutionCompleted:
    """Tests for ExecutionCompleted event."""

    def test_success(self):
        evt = ExecutionCompleted(
            success=True, total_steps=5, total_tokens=1000, total_cost=0.05, duration=3.2
        )
        assert evt.success is True
        assert evt.total_steps == 5
        assert evt.error is None

    def test_failure(self):
        evt = ExecutionCompleted(success=False, error="Timeout")
        assert evt.success is False
        assert evt.error == "Timeout"


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestDomainExceptions:
    """Tests for domain exception hierarchy."""

    def test_domain_error_is_exception(self):
        assert issubclass(DomainError, Exception)

    def test_budget_exceeded_inherits(self):
        assert issubclass(BudgetExceededError, DomainError)
        exc = BudgetExceededError("max steps")
        assert str(exc) == "max steps"
        assert isinstance(exc, DomainError)

    def test_execution_failed_inherits(self):
        assert issubclass(ExecutionFailedError, DomainError)

    def test_security_violation_inherits(self):
        assert issubclass(SecurityViolationError, DomainError)

    def test_parse_failed_inherits(self):
        assert issubclass(ParseFailedError, DomainError)

    def test_configuration_error_inherits(self):
        assert issubclass(ConfigurationError, DomainError)

    def test_can_catch_all_with_domain_error(self):
        """All domain exceptions are caught by a single except DomainError."""
        for exc_cls in [
            BudgetExceededError,
            ExecutionFailedError,
            SecurityViolationError,
            ParseFailedError,
            ConfigurationError,
        ]:
            with pytest.raises(DomainError):
                raise exc_cls("test")
