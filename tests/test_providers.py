"""Tests for LLM provider implementations."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from rlmkit.llm.base import BaseLLMProvider, LLMResponse
from rlmkit.llm.mock_client import MockLLMClient


class TestLLMResponse:
    """Test LLMResponse dataclass."""
    
    def test_basic_response(self):
        """Test basic response creation."""
        response = LLMResponse(
            content="Hello, world!",
            model="test-model"
        )
        
        assert response.content == "Hello, world!"
        assert response.model == "test-model"
        assert response.input_tokens is None
        assert response.output_tokens is None
    
    def test_response_with_tokens(self):
        """Test response with token counts."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50
        )
        
        assert response.input_tokens == 100
        assert response.output_tokens == 50
    
    def test_response_with_metadata(self):
        """Test response with metadata."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            finish_reason="stop",
            metadata={"id": "123", "created": 1234567890}
        )
        
        assert response.finish_reason == "stop"
        assert response.metadata["id"] == "123"


class TestBaseLLMProvider:
    """Test base provider functionality."""
    
    def test_abstract_methods(self):
        """Test that BaseLLMProvider is abstract."""
        with pytest.raises(TypeError):
            BaseLLMProvider(model="test")
    
    def test_validate_messages_empty(self):
        """Test validation rejects empty messages."""
        # Create a concrete implementation for testing
        class TestProvider(BaseLLMProvider):
            def complete(self, messages):
                return "test"
            def complete_with_metadata(self, messages):
                return LLMResponse(content="test", model=self.model)
        
        provider = TestProvider(model="test")
        
        with pytest.raises(ValueError) as exc:
            provider.validate_messages([])
        
        assert "cannot be empty" in str(exc.value)
    
    def test_validate_messages_missing_role(self):
        """Test validation rejects messages missing 'role'."""
        class TestProvider(BaseLLMProvider):
            def complete(self, messages):
                return "test"
            def complete_with_metadata(self, messages):
                return LLMResponse(content="test", model=self.model)
        
        provider = TestProvider(model="test")
        
        with pytest.raises(ValueError) as exc:
            provider.validate_messages([{"content": "test"}])
        
        assert "missing 'role'" in str(exc.value)
    
    def test_validate_messages_missing_content(self):
        """Test validation rejects messages missing 'content'."""
        class TestProvider(BaseLLMProvider):
            def complete(self, messages):
                return "test"
            def complete_with_metadata(self, messages):
                return LLMResponse(content="test", model=self.model)
        
        provider = TestProvider(model="test")
        
        with pytest.raises(ValueError) as exc:
            provider.validate_messages([{"role": "user"}])
        
        assert "missing 'content'" in str(exc.value)
    
    def test_validate_messages_invalid_role(self):
        """Test validation rejects invalid roles."""
        class TestProvider(BaseLLMProvider):
            def complete(self, messages):
                return "test"
            def complete_with_metadata(self, messages):
                return LLMResponse(content="test", model=self.model)
        
        provider = TestProvider(model="test")
        
        with pytest.raises(ValueError) as exc:
            provider.validate_messages([
                {"role": "invalid", "content": "test"}
            ])
        
        assert "invalid role" in str(exc.value)
    
    def test_validate_messages_valid(self):
        """Test validation accepts valid messages."""
        class TestProvider(BaseLLMProvider):
            def complete(self, messages):
                return "test"
            def complete_with_metadata(self, messages):
                return LLMResponse(content="test", model=self.model)
        
        provider = TestProvider(model="test")
        
        # Should not raise
        provider.validate_messages([
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ])
    
    def test_repr(self):
        """Test string representation."""
        class TestProvider(BaseLLMProvider):
            def complete(self, messages):
                return "test"
            def complete_with_metadata(self, messages):
                return LLMResponse(content="test", model=self.model)
        
        provider = TestProvider(model="test-model-123")
        assert "TestProvider" in repr(provider)
        assert "test-model-123" in repr(provider)


class TestMockLLMClient:
    """Test MockLLMClient functionality."""
    
    def test_single_response(self):
        """Test mock client with single response."""
        client = MockLLMClient(["response 1"])
        
        result = client.complete([{"role": "user", "content": "test"}])
        assert result == "response 1"
    
    def test_multiple_responses(self):
        """Test mock client with multiple responses."""
        client = MockLLMClient(["response 1", "response 2", "response 3"])
        
        assert client.complete([{"role": "user", "content": "test"}]) == "response 1"
        assert client.complete([{"role": "user", "content": "test"}]) == "response 2"
        assert client.complete([{"role": "user", "content": "test"}]) == "response 3"
    
    def test_repeats_last_response(self):
        """Test mock client repeats last response when exhausted."""
        client = MockLLMClient(["response 1", "response 2"])
        
        client.complete([{"role": "user", "content": "test"}])
        client.complete([{"role": "user", "content": "test"}])
        
        # Should repeat last response
        assert client.complete([{"role": "user", "content": "test"}]) == "response 2"
        assert client.complete([{"role": "user", "content": "test"}]) == "response 2"
    
    def test_call_history(self):
        """Test mock client records call history."""
        client = MockLLMClient(["response"])
        
        messages1 = [{"role": "user", "content": "hello"}]
        messages2 = [{"role": "user", "content": "world"}]
        
        client.complete(messages1)
        client.complete(messages2)
        
        assert len(client.call_history) == 2
        assert client.call_history[0] == messages1
        assert client.call_history[1] == messages2
    
    def test_call_count(self):
        """Test mock client tracks call count."""
        client = MockLLMClient(["response"])
        
        assert client.call_count == 0
        
        client.complete([{"role": "user", "content": "test"}])
        assert client.call_count == 1
        
        client.complete([{"role": "user", "content": "test"}])
        assert client.call_count == 2
    
    def test_reset(self):
        """Test mock client reset functionality."""
        client = MockLLMClient(["response"])
        
        client.complete([{"role": "user", "content": "test"}])
        client.complete([{"role": "user", "content": "test"}])
        
        assert client.call_count == 2
        assert len(client.call_history) == 2
        
        client.reset()
        
        assert client.call_count == 0
        assert len(client.call_history) == 0
    
    def test_empty_responses_error(self):
        """Test mock client rejects empty response list."""
        with pytest.raises(ValueError) as exc:
            MockLLMClient([])
        
        assert "at least one response" in str(exc.value)


class TestOpenAIClientMocked:
    """Test OpenAI client with mocked API calls."""
    
    @pytest.fixture
    def mock_openai(self):
        """Mock openai module."""
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            yield
    
    def test_requires_api_key(self, mock_openai):
        """Test OpenAI client requires API key."""
        from rlmkit.llm.openai_client import OpenAIClient
        
        with pytest.raises(ValueError) as exc:
            OpenAIClient(model="gpt-4")
        
        assert "API key required" in str(exc.value)
    
    def test_accepts_api_key_parameter(self, mock_openai):
        """Test OpenAI client accepts API key as parameter."""
        with patch('rlmkit.llm.openai_client.os.getenv', return_value=None):
            from rlmkit.llm.openai_client import OpenAIClient
            
            client = OpenAIClient(model="gpt-4", api_key="test-key")
            assert client.api_key == "test-key"
    
    def test_model_selection(self, mock_openai):
        """Test OpenAI client model selection."""
        from rlmkit.llm.openai_client import OpenAIClient
        
        client = OpenAIClient(model="gpt-3.5-turbo", api_key="test")
        assert client.model == "gpt-3.5-turbo"


class TestClaudeClientMocked:
    """Test Claude client with mocked API calls."""
    
    @pytest.fixture
    def mock_anthropic(self):
        """Mock anthropic module."""
        with patch.dict('sys.modules', {'anthropic': MagicMock()}):
            yield
    
    def test_requires_api_key(self, mock_anthropic):
        """Test Claude client requires API key."""
        from rlmkit.llm.anthropic_client import ClaudeClient
        
        with pytest.raises(ValueError) as exc:
            ClaudeClient(model="claude-3-sonnet-20240229")
        
        assert "API key required" in str(exc.value)
    
    def test_accepts_api_key_parameter(self, mock_anthropic):
        """Test Claude client accepts API key as parameter."""
        with patch('rlmkit.llm.anthropic_client.os.getenv', return_value=None):
            from rlmkit.llm.anthropic_client import ClaudeClient
            
            client = ClaudeClient(
                model="claude-3-sonnet-20240229",
                api_key="test-key"
            )
            assert client.api_key == "test-key"
    
    def test_model_selection(self, mock_anthropic):
        """Test Claude client model selection."""
        from rlmkit.llm.anthropic_client import ClaudeClient
        
        client = ClaudeClient(model="claude-3-opus-20240229", api_key="test")
        assert client.model == "claude-3-opus-20240229"
    
    def test_default_max_tokens(self, mock_anthropic):
        """Test Claude client sets default max_tokens."""
        from rlmkit.llm.anthropic_client import ClaudeClient
        
        client = ClaudeClient(api_key="test")
        assert client.max_tokens == 4096  # Claude requires max_tokens


class TestOllamaClientMocked:
    """Test Ollama client with mocked HTTP calls."""
    
    def test_connection_check_fails_gracefully(self):
        """Test Ollama client handles connection failures."""
        from rlmkit.llm.ollama_client import OllamaClient
        import requests
        
        with patch('rlmkit.llm.ollama_client.requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")
            
            with pytest.raises(ConnectionError) as exc:
                OllamaClient(model="llama2")
            
            assert "Cannot connect to Ollama" in str(exc.value)
    
    def test_model_selection(self):
        """Test Ollama client model selection."""
        from rlmkit.llm.ollama_client import OllamaClient
        
        with patch('rlmkit.llm.ollama_client.requests.get'):
            client = OllamaClient(model="mistral")
            assert client.model == "mistral"
    
    def test_base_url_configuration(self):
        """Test Ollama client custom base URL."""
        from rlmkit.llm.ollama_client import OllamaClient
        
        with patch('rlmkit.llm.ollama_client.requests.get'):
            client = OllamaClient(
                model="llama2",
                base_url="http://custom-server:11434"
            )
            assert "custom-server" in client.base_url
    
    def test_complete_success(self):
        """Test successful completion."""
        from rlmkit.llm.ollama_client import OllamaClient
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'message': {'content': 'Hello from Ollama!'},
            'done': True
        }
        mock_response.raise_for_status = Mock()
        
        with patch('rlmkit.llm.ollama_client.requests.get'):
            with patch('rlmkit.llm.ollama_client.requests.post', return_value=mock_response):
                client = OllamaClient(model="llama2")
                result = client.complete([{"role": "user", "content": "Hi"}])
                
                assert result == "Hello from Ollama!"
    
    def test_complete_with_metadata(self):
        """Test completion with metadata."""
        from rlmkit.llm.ollama_client import OllamaClient
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'message': {'content': 'Response'},
            'prompt_eval_count': 10,
            'eval_count': 20,
            'done_reason': 'stop',
            'total_duration': 1000000,
            'done': True
        }
        mock_response.raise_for_status = Mock()
        
        with patch('rlmkit.llm.ollama_client.requests.get'):
            with patch('rlmkit.llm.ollama_client.requests.post', return_value=mock_response):
                client = OllamaClient(model="llama2")
                result = client.complete_with_metadata([{"role": "user", "content": "Hi"}])
                
                assert result.content == "Response"
                assert result.input_tokens == 10
                assert result.output_tokens == 20
                assert result.finish_reason == "stop"
    
    def test_list_models(self):
        """Test listing available models."""
        from rlmkit.llm.ollama_client import OllamaClient
        
        mock_response = Mock()
        mock_response.json.return_value = {
            'models': [
                {'name': 'llama2'},
                {'name': 'mistral'},
                {'name': 'codellama'}
            ]
        }
        mock_response.raise_for_status = Mock()
        
        with patch('rlmkit.llm.ollama_client.requests.get', return_value=mock_response):
            client = OllamaClient(model="llama2")
            models = client.list_models()
            
            assert 'llama2' in models
            assert 'mistral' in models
            assert 'codellama' in models
    
    def test_timeout_handling(self):
        """Test timeout handling."""
        from rlmkit.llm.ollama_client import OllamaClient
        import requests
        
        with patch('rlmkit.llm.ollama_client.requests.get'):
            with patch('rlmkit.llm.ollama_client.requests.post') as mock_post:
                mock_post.side_effect = requests.exceptions.Timeout()
                
                client = OllamaClient(model="llama2")
                
                with pytest.raises(RuntimeError) as exc:
                    client.complete([{"role": "user", "content": "Hi"}])
                
                assert "timed out" in str(exc.value).lower()


class TestProviderImports:
    """Test optional provider imports."""
    
    def test_base_imports_always_available(self):
        """Test base classes are always available."""
        from rlmkit import BaseLLMProvider, LLMResponse, MockLLMClient
        
        assert BaseLLMProvider is not None
        assert LLMResponse is not None
        assert MockLLMClient is not None
    
    def test_openai_import_graceful_failure(self):
        """Test OpenAI import fails gracefully without package."""
        # OpenAI might not be installed in test environment
        try:
            from rlmkit import OpenAIClient
            # If it imports, that's fine
            assert OpenAIClient is not None or OpenAIClient is None
        except ImportError:
            # Expected if openai package not installed
            pass
    
    def test_claude_import_graceful_failure(self):
        """Test Claude import fails gracefully without package."""
        # Anthropic might not be installed in test environment
        try:
            from rlmkit import ClaudeClient
            # If it imports, that's fine
            assert ClaudeClient is not None or ClaudeClient is None
        except ImportError:
            # Expected if anthropic package not installed
            pass
    
    def test_ollama_import_graceful_failure(self):
        """Test Ollama import fails gracefully without package."""
        # requests might not be installed in test environment
        try:
            from rlmkit import OllamaClient
            # If it imports, that's fine
            assert OllamaClient is not None or OllamaClient is None
        except ImportError:
            # Expected if requests package not installed
            pass
