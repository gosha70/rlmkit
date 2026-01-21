"""
Test suite for Phase 1 Infrastructure.

Run with: pytest tests/ui/test_infrastructure.py -v
"""

import pytest
from datetime import datetime
from src.rlmkit.ui.services import (
    ChatMessage,
    Response,
    ExecutionMetrics,
    ComparisonMetrics,
    LLMProviderConfig,
    SessionMetrics,
    ChatManager,
    MetricsCollector,
    MemoryMonitor,
    LLMConfigManager,
)


class TestDataModels:
    """Test data class definitions."""
    
    def test_response_creation(self):
        """Test Response dataclass."""
        response = Response(
            content="Hello, world!",
            stop_reason="stop",
        )
        assert response.content == "Hello, world!"
        assert response.stop_reason == "stop"
    
    def test_execution_metrics_creation(self):
        """Test ExecutionMetrics dataclass."""
        metrics = ExecutionMetrics(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost_usd=0.05,
            cost_breakdown={"input": 0.03, "output": 0.02},
            execution_time_seconds=2.5,
            steps_taken=3,
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
        )
        assert metrics.total_tokens == 150
        assert metrics.success is True
    
    def test_chat_message_creation(self):
        """Test ChatMessage dataclass."""
        message = ChatMessage(
            user_query="What is RLM?",
            mode="rlm_only",
        )
        assert message.user_query == "What is RLM?"
        assert message.mode == "rlm_only"
        assert message.id is not None
        assert message.timestamp is not None
    
    def test_chat_message_has_responses(self):
        """Test response checking methods."""
        message = ChatMessage(user_query="Test")
        assert not message.has_rlm_response()
        assert not message.has_direct_response()
        
        # Add RLM response
        message.rlm_response = Response(
            content="RLM answer",
            stop_reason="stop"
        )
        assert message.has_rlm_response()
        assert not message.has_direct_response()
    
    def test_llm_provider_config_creation(self):
        """Test LLMProviderConfig dataclass."""
        config = LLMProviderConfig(
            provider="openai",
            model="gpt-4",
            api_key_env_var="OPENAI_API_KEY",
            input_cost_per_1k_tokens=0.03,
            output_cost_per_1k_tokens=0.06,
        )
        assert config.provider == "openai"
        assert config.is_configured is False  # No API key yet
    
    def test_session_metrics_creation(self):
        """Test SessionMetrics dataclass."""
        metrics = SessionMetrics(
            total_messages=5,
            total_tokens=10000,
            total_cost=0.10,
        )
        assert metrics.total_messages == 5
        assert metrics.total_tokens == 10000


class TestChatManager:
    """Test ChatManager service."""
    
    def test_chat_manager_init(self):
        """Test ChatManager initialization."""
        session = {}
        manager = ChatManager(session)
        assert "messages" in session
        assert "conversation_id" in session
        assert len(manager.get_messages()) == 0
    
    def test_get_messages(self):
        """Test getting messages."""
        session = {}
        manager = ChatManager(session)
        messages = manager.get_messages()
        assert isinstance(messages, list)
        assert len(messages) == 0


class TestMemoryMonitor:
    """Test MemoryMonitor service."""
    
    def test_memory_monitor_init(self):
        """Test MemoryMonitor initialization."""
        monitor = MemoryMonitor()
        assert monitor.baseline_memory_mb >= 0
        assert monitor.peak_memory_mb >= 0
    
    def test_memory_capture(self):
        """Test memory capture."""
        monitor = MemoryMonitor()
        initial_current = monitor.current_mb()
        
        monitor.capture()
        stats = monitor.get_stats()
        
        assert "current" in stats
        assert "peak" in stats
        assert "average" in stats


class TestMetricsCollector:
    """Test MetricsCollector service."""
    
    def test_metrics_collector_init(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert "openai" in collector.provider_pricing
        assert "anthropic" in collector.provider_pricing
    
    def test_add_provider_pricing(self):
        """Test adding provider pricing."""
        collector = MetricsCollector()
        collector.add_provider_pricing(
            provider="custom",
            model="custom-model",
            input_cost_per_1k=0.001,
            output_cost_per_1k=0.002,
        )
        assert "custom" in collector.provider_pricing
        assert "custom-model" in collector.provider_pricing["custom"]
    
    def test_calculate_cost_openai_gpt4(self):
        """Test cost calculation for OpenAI GPT-4."""
        collector = MetricsCollector()
        # GPT-4: input $0.03/1k, output $0.06/1k
        # 1000 input tokens = $0.03
        # 500 output tokens = $0.03
        # Total = $0.06
        cost = collector._calculate_cost(1000, 500, "openai", "gpt-4")
        assert abs(cost - 0.06) < 0.0001
    
    def test_calculate_cost_anthropic_haiku(self):
        """Test cost calculation for Anthropic Claude 3 Haiku."""
        collector = MetricsCollector()
        # Claude 3 Haiku: input $0.00025/1k, output $0.00125/1k
        # 1000 input tokens = $0.00025
        # 1000 output tokens = $0.00125
        # Total = $0.0015
        cost = collector._calculate_cost(1000, 1000, "anthropic", "claude-3-haiku")
        assert abs(cost - 0.0015) < 0.0001
    
    def test_calculate_cost_unknown_provider(self):
        """Test cost calculation with unknown provider."""
        collector = MetricsCollector()
        cost = collector._calculate_cost(1000, 500, "unknown-provider", "model")
        assert cost == 0.0
    
    def test_calculate_cost_unknown_model(self):
        """Test cost calculation with unknown model."""
        collector = MetricsCollector()
        cost = collector._calculate_cost(1000, 500, "openai", "unknown-model")
        assert cost == 0.0
    
    def test_cost_breakdown_openai_gpt4(self):
        """Test cost breakdown for OpenAI GPT-4."""
        collector = MetricsCollector()
        # GPT-4: input $0.03/1k, output $0.06/1k
        # 1000 input tokens = $0.03
        # 500 output tokens = $0.03
        breakdown = collector._cost_breakdown(1000, 500, "openai", "gpt-4")
        assert abs(breakdown["input"] - 0.03) < 0.0001
        assert abs(breakdown["output"] - 0.03) < 0.0001
    
    def test_cost_breakdown_structure(self):
        """Test cost breakdown returns correct structure."""
        collector = MetricsCollector()
        breakdown = collector._cost_breakdown(1000, 500, "openai", "gpt-4")
        assert isinstance(breakdown, dict)
        assert "input" in breakdown
        assert "output" in breakdown
        assert isinstance(breakdown["input"], float)
        assert isinstance(breakdown["output"], float)
    
    def test_cost_breakdown_unknown_provider(self):
        """Test cost breakdown with unknown provider."""
        collector = MetricsCollector()
        breakdown = collector._cost_breakdown(1000, 500, "unknown", "model")
        assert breakdown["input"] == 0.0
        assert breakdown["output"] == 0.0
    
    def test_cost_calculation_consistency(self):
        """Test that _calculate_cost equals sum of breakdown."""
        collector = MetricsCollector()
        input_tokens = 1234
        output_tokens = 567
        provider = "openai"
        model = "gpt-4-turbo"
        
        total_cost = collector._calculate_cost(input_tokens, output_tokens, provider, model)
        breakdown = collector._cost_breakdown(input_tokens, output_tokens, provider, model)
        breakdown_sum = breakdown["input"] + breakdown["output"]
        
        assert abs(total_cost - breakdown_sum) < 0.0001


class TestLLMConfigManager:
    """Test LLMConfigManager service."""
    
    def test_llm_config_manager_init(self, tmp_path):
        """Test LLMConfigManager initialization."""
        manager = LLMConfigManager(config_dir=tmp_path)
        assert manager.config_dir.exists()
        assert len(manager.list_providers()) == 0
    
    def test_list_providers(self, tmp_path):
        """Test listing providers."""
        manager = LLMConfigManager(config_dir=tmp_path)
        providers = manager.list_providers()
        assert isinstance(providers, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
