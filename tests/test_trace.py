"""
Tests for execution tracing module.

This tests the Bet 3 implementation - execution trajectory capture and export.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path

from rlmkit.core.trace import (
    TraceStep,
    ExecutionTrace,
    load_trace_from_jsonl,
    load_trace_from_json,
)


class TestTraceStep:
    """Test TraceStep dataclass."""
    
    def test_create_basic_step(self):
        """Test creating a basic trace step."""
        step = TraceStep(
            index=0,
            action_type="inspect",
            code="print(len(P))",
            output="1000",
            tokens_used=50
        )
        
        assert step.index == 0
        assert step.action_type == "inspect"
        assert step.code == "print(len(P))"
        assert step.output == "1000"
        assert step.tokens_used == 50
        assert step.recursion_depth == 0  # default
    
    def test_step_with_recursion(self):
        """Test step with recursion depth."""
        step = TraceStep(
            index=1,
            action_type="subcall",
            recursion_depth=2,
            tokens_used=100
        )
        
        assert step.recursion_depth == 2
    
    def test_step_with_error(self):
        """Test step with error."""
        step = TraceStep(
            index=2,
            action_type="error",
            error="Division by zero",
            tokens_used=20
        )
        
        assert step.error == "Division by zero"
        assert step.action_type == "error"
    
    def test_step_to_dict(self):
        """Test converting step to dictionary."""
        step = TraceStep(
            index=0,
            action_type="inspect",
            code="x = 1",
            output="None",
            tokens_used=10,
            cost=0.001,
            duration=0.5
        )
        
        step_dict = step.to_dict()
        
        assert step_dict["index"] == 0
        assert step_dict["action_type"] == "inspect"
        assert step_dict["code"] == "x = 1"
        assert step_dict["tokens_used"] == 10
        assert "timestamp" in step_dict
    
    def test_step_str_representation(self):
        """Test string representation of step."""
        step = TraceStep(
            index=0,
            action_type="inspect",
            code="print('hello')",
            output="hello",
            tokens_used=15,
            cost=0.002
        )
        
        str_repr = str(step)
        
        assert "[Step 0]" in str_repr
        assert "INSPECT" in str_repr
        assert "Code:" in str_repr
        assert "Output:" in str_repr
        assert "Tokens: 15" in str_repr


class TestExecutionTrace:
    """Test ExecutionTrace class."""
    
    def test_create_empty_trace(self):
        """Test creating an empty trace."""
        trace = ExecutionTrace()
        
        assert len(trace) == 0
        assert len(trace.steps) == 0
        assert trace.end_time is None
    
    def test_add_steps(self):
        """Test adding steps to trace."""
        trace = ExecutionTrace()
        
        step1 = TraceStep(0, "inspect", tokens_used=10)
        step2 = TraceStep(1, "subcall", tokens_used=20)
        
        trace.add_step(step1)
        trace.add_step(step2)
        
        assert len(trace) == 2
        assert trace.steps[0].index == 0
        assert trace.steps[1].index == 1
    
    def test_total_tokens(self):
        """Test calculating total tokens."""
        trace = ExecutionTrace()
        
        trace.add_step(TraceStep(0, "inspect", tokens_used=10))
        trace.add_step(TraceStep(1, "subcall", tokens_used=20))
        trace.add_step(TraceStep(2, "final", tokens_used=30))
        
        assert trace.total_tokens == 60
    
    def test_total_cost(self):
        """Test calculating total cost."""
        trace = ExecutionTrace()
        
        trace.add_step(TraceStep(0, "inspect", tokens_used=10, cost=0.001))
        trace.add_step(TraceStep(1, "subcall", tokens_used=20, cost=0.002))
        trace.add_step(TraceStep(2, "final", tokens_used=30, cost=0.003))
        
        assert trace.total_cost == 0.006
    
    def test_max_recursion_depth(self):
        """Test finding max recursion depth."""
        trace = ExecutionTrace()
        
        trace.add_step(TraceStep(0, "inspect", recursion_depth=0, tokens_used=10))
        trace.add_step(TraceStep(1, "subcall", recursion_depth=1, tokens_used=20))
        trace.add_step(TraceStep(2, "subcall", recursion_depth=2, tokens_used=15))
        trace.add_step(TraceStep(3, "final", recursion_depth=0, tokens_used=25))
        
        assert trace.max_recursion_depth == 2
    
    def test_filter_by_type(self):
        """Test filtering steps by action type."""
        trace = ExecutionTrace()
        
        trace.add_step(TraceStep(0, "inspect", tokens_used=10))
        trace.add_step(TraceStep(1, "subcall", tokens_used=20))
        trace.add_step(TraceStep(2, "inspect", tokens_used=15))
        trace.add_step(TraceStep(3, "final", tokens_used=25))
        
        inspects = trace.filter_by_type("inspect")
        subcalls = trace.filter_by_type("subcall")
        finals = trace.filter_by_type("final")
        
        assert len(inspects) == 2
        assert len(subcalls) == 1
        assert len(finals) == 1
    
    def test_filter_by_depth(self):
        """Test filtering steps by recursion depth."""
        trace = ExecutionTrace()
        
        trace.add_step(TraceStep(0, "inspect", recursion_depth=0, tokens_used=10))
        trace.add_step(TraceStep(1, "subcall", recursion_depth=1, tokens_used=20))
        trace.add_step(TraceStep(2, "subcall", recursion_depth=1, tokens_used=15))
        trace.add_step(TraceStep(3, "final", recursion_depth=0, tokens_used=25))
        
        depth_0 = trace.filter_by_depth(0)
        depth_1 = trace.filter_by_depth(1)
        
        assert len(depth_0) == 2
        assert len(depth_1) == 2
    
    def test_get_step(self):
        """Test getting step by index."""
        trace = ExecutionTrace()
        
        step1 = TraceStep(0, "inspect", tokens_used=10)
        step2 = TraceStep(1, "subcall", tokens_used=20)
        
        trace.add_step(step1)
        trace.add_step(step2)
        
        assert trace.get_step(0) == step1
        assert trace.get_step(1) == step2
        assert trace.get_step(5) is None
    
    def test_iteration(self):
        """Test iterating over steps."""
        trace = ExecutionTrace()
        
        trace.add_step(TraceStep(0, "inspect", tokens_used=10))
        trace.add_step(TraceStep(1, "subcall", tokens_used=20))
        
        count = 0
        for step in trace:
            count += 1
            assert isinstance(step, TraceStep)
        
        assert count == 2
    
    def test_indexing(self):
        """Test indexing into trace."""
        trace = ExecutionTrace()
        
        step1 = TraceStep(0, "inspect", tokens_used=10)
        step2 = TraceStep(1, "subcall", tokens_used=20)
        
        trace.add_step(step1)
        trace.add_step(step2)
        
        assert trace[0] == step1
        assert trace[1] == step2


class TestTraceExport:
    """Test trace export functionality."""
    
    def setup_method(self):
        """Create a sample trace for testing."""
        self.trace = ExecutionTrace()
        self.trace.add_step(TraceStep(
            0, "inspect",
            code="print(len(P))",
            output="1000",
            tokens_used=10,
            cost=0.001,
            duration=0.5
        ))
        self.trace.add_step(TraceStep(
            1, "subcall",
            code="subcall('analyze', P[:100])",
            output="Analysis complete",
            tokens_used=50,
            cost=0.005,
            duration=2.0,
            recursion_depth=1
        ))
        self.trace.add_step(TraceStep(
            2, "final",
            output="Final answer: The document is 1000 chars",
            tokens_used=30,
            cost=0.003,
            duration=1.0
        ))
        self.trace.finalize()
    
    def test_to_dict(self):
        """Test converting trace to dictionary."""
        trace_dict = self.trace.to_dict()
        
        assert "steps" in trace_dict
        assert "metadata" in trace_dict
        assert len(trace_dict["steps"]) == 3
        assert trace_dict["metadata"]["total_steps"] == 3
        assert trace_dict["metadata"]["total_tokens"] == 90
        assert trace_dict["metadata"]["max_recursion_depth"] == 1
    
    def test_export_to_json(self):
        """Test exporting to JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trace.json"
            
            self.trace.to_json(str(filepath))
            
            assert filepath.exists()
            
            # Verify it's valid JSON
            with open(filepath) as f:
                data = json.load(f)
            
            assert "steps" in data
            assert "metadata" in data
            assert len(data["steps"]) == 3
    
    def test_export_to_jsonl(self):
        """Test exporting to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trace.jsonl"
            
            self.trace.to_jsonl(str(filepath))
            
            assert filepath.exists()
            
            # Verify format (metadata + steps, one per line)
            with open(filepath) as f:
                lines = f.readlines()
            
            assert len(lines) == 4  # 1 metadata + 3 steps
            
            # Each line should be valid JSON
            for line in lines:
                json.loads(line)
    
    def test_export_to_markdown(self):
        """Test exporting to Markdown file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trace.md"
            
            self.trace.to_markdown(str(filepath))
            
            assert filepath.exists()
            
            # Verify it contains expected sections
            with open(filepath) as f:
                content = f.read()
            
            assert "# RLM Execution Trace" in content
            assert "## Summary" in content
            assert "## Execution Steps" in content
            assert "### Step 0: INSPECT" in content
            assert "**Total Steps:** 3" in content  # Markdown uses bold format


class TestTraceLoading:
    """Test loading traces from files."""
    
    def create_sample_trace(self) -> ExecutionTrace:
        """Create a sample trace for testing."""
        trace = ExecutionTrace()
        trace.add_step(TraceStep(0, "inspect", code="x=1", tokens_used=10))
        trace.add_step(TraceStep(1, "final", output="done", tokens_used=20))
        trace.finalize()
        return trace
    
    def test_load_from_json(self):
        """Test loading trace from JSON file."""
        trace = self.create_sample_trace()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trace.json"
            trace.to_json(str(filepath))
            
            # Load it back
            loaded = load_trace_from_json(str(filepath))
            
            assert len(loaded) == 2
            assert loaded.total_tokens == 30
            assert loaded[0].index == 0
            assert loaded[0].action_type == "inspect"
            assert loaded[1].index == 1
            assert loaded[1].action_type == "final"
    
    def test_load_from_jsonl(self):
        """Test loading trace from JSONL file."""
        trace = self.create_sample_trace()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trace.jsonl"
            trace.to_jsonl(str(filepath))
            
            # Load it back
            loaded = load_trace_from_jsonl(str(filepath))
            
            assert len(loaded) == 2
            assert loaded.total_tokens == 30
            assert loaded[0].code == "x=1"
            assert loaded[1].output == "done"
    
    def test_round_trip_json(self):
        """Test saving and loading preserves data."""
        trace = self.create_sample_trace()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "trace.json"
            
            # Save
            trace.to_json(str(filepath))
            
            # Load
            loaded = load_trace_from_json(str(filepath))
            
            # Compare
            assert len(loaded) == len(trace)
            assert loaded.total_tokens == trace.total_tokens
            assert loaded.total_cost == trace.total_cost
            
            for orig, load in zip(trace.steps, loaded.steps):
                assert orig.index == load.index
                assert orig.action_type == load.action_type
                assert orig.tokens_used == load.tokens_used


class TestTraceIntegration:
    """Integration tests for tracing."""
    
    def test_realistic_trace_scenario(self):
        """Test a realistic RLM execution trace."""
        trace = ExecutionTrace()
        trace.metadata = {
            "query": "Summarize this document",
            "model": "gpt-4o",
            "provider": "openai"
        }
        
        # Step 1: Inspect context
        trace.add_step(TraceStep(
            0, "inspect",
            code="print(f'Context length: {len(P)}')",
            output="Context length: 50000",
            tokens_used=25,
            cost=0.0001,
            duration=0.3,
            model="gpt-4o"
        ))
        
        # Step 2: Subcall for section 1
        trace.add_step(TraceStep(
            1, "subcall",
            code="subcall('Summarize section 1', P[:10000])",
            output="Section 1 summary: Introduction to topic",
            tokens_used=200,
            cost=0.002,
            duration=3.0,
            recursion_depth=1,
            model="gpt-4o"
        ))
        
        # Step 3: Subcall for section 2
        trace.add_step(TraceStep(
            2, "subcall",
            code="subcall('Summarize section 2', P[10000:20000])",
            output="Section 2 summary: Main findings",
            tokens_used=250,
            cost=0.0025,
            duration=3.5,
            recursion_depth=1,
            model="gpt-4o"
        ))
        
        # Step 4: Final answer
        trace.add_step(TraceStep(
            3, "final",
            output="FINAL: Document covers introduction and main findings...",
            tokens_used=100,
            cost=0.001,
            duration=1.0,
            model="gpt-4o"
        ))
        
        trace.finalize()
        
        # Verify trace properties
        assert len(trace) == 4
        assert trace.total_tokens == 575
        # 0.0001 + 0.002 + 0.0025 + 0.001 = 0.0056
        assert abs(trace.total_cost - 0.0056) < 0.0001
        assert trace.max_recursion_depth == 1
        
        # Verify can export
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "trace.json"
            md_path = Path(tmpdir) / "trace.md"
            
            trace.to_json(str(json_path))
            trace.to_markdown(str(md_path))
            
            assert json_path.exists()
            assert md_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
