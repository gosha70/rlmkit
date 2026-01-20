"""OpenAI LLM provider implementation."""

import os
from typing import List, Dict, Optional
from .base import BaseLLMProvider, LLMResponse


class OpenAIClient(BaseLLMProvider):
    """
    OpenAI API client for RLM.
    
    Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
    
    Example:
        >>> import os
        >>> os.environ['OPENAI_API_KEY'] = 'sk-...'
        >>> 
        >>> client = OpenAIClient(model="gpt-4")
        >>> response = client.complete([
        ...     {"role": "user", "content": "Hello!"}
        ... ])
        >>> print(response)
    """
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            organization: OpenAI organization ID (optional)
            base_url: Custom API base URL (for proxies/Azure)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters passed to OpenAI API
        """
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Provide via api_key parameter "
                "or OPENAI_API_KEY environment variable."
            )
        
        self.organization = organization or os.getenv('OPENAI_ORGANIZATION')
        self.base_url = base_url
        
        # Lazy import openai to avoid requiring it if not used
        try:
            import openai
            self._openai = openai
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )
        
        # Initialize OpenAI client
        self._client = self._openai.OpenAI(
            api_key=self.api_key,
            organization=self.organization,
            base_url=self.base_url,
        )
    
    def complete(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Generated text response
            
        Raises:
            ValueError: If messages are invalid
            openai.OpenAIError: If API call fails
        """
        self.validate_messages(messages)
        
        # Build API request parameters
        params = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
        }
        
        if self.max_tokens:
            params['max_tokens'] = self.max_tokens
        
        # Add extra parameters
        params.update(self.extra_params)
        
        # Make API call
        try:
            response = self._client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e
    
    def complete_with_metadata(
        self,
        messages: List[Dict[str, str]]
    ) -> LLMResponse:
        """
        Generate completion with full metadata.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            LLMResponse with content and metadata
            
        Raises:
            ValueError: If messages are invalid
            openai.OpenAIError: If API call fails
        """
        self.validate_messages(messages)
        
        # Build API request parameters
        params = {
            'model': self.model,
            'messages': messages,
            'temperature': self.temperature,
        }
        
        if self.max_tokens:
            params['max_tokens'] = self.max_tokens
        
        # Add extra parameters
        params.update(self.extra_params)
        
        # Make API call
        try:
            response = self._client.chat.completions.create(**params)
            choice = response.choices[0]
            usage = response.usage
            
            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
                finish_reason=choice.finish_reason,
                metadata={
                    'id': response.id,
                    'created': response.created,
                    'system_fingerprint': getattr(response, 'system_fingerprint', None),
                }
            )
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}") from e
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using tiktoken.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        try:
            import tiktoken
            
            # Get encoding for model
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to cl100k_base (used by gpt-4, gpt-3.5-turbo)
                encoding = tiktoken.get_encoding("cl100k_base")
            
            return len(encoding.encode(text))
        except ImportError:
            # Fallback to simple estimation if tiktoken not available
            return len(text) // 4
    
    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for token usage.
        
        Uses approximate pricing as of 2024. May need updates.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        # Pricing per 1M tokens (approximate, update as needed)
        pricing = {
            'gpt-4': {'input': 30.0, 'output': 60.0},
            'gpt-4-turbo': {'input': 10.0, 'output': 30.0},
            'gpt-4-turbo-preview': {'input': 10.0, 'output': 30.0},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50},
            'gpt-3.5-turbo-16k': {'input': 3.0, 'output': 4.0},
        }
        
        # Get pricing for model (or use gpt-4 as default)
        model_pricing = pricing.get(self.model, pricing['gpt-4'])
        
        input_cost = (input_tokens / 1_000_000) * model_pricing['input']
        output_cost = (output_tokens / 1_000_000) * model_pricing['output']
        
        return input_cost + output_cost
