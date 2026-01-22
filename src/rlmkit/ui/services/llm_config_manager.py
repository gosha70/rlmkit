# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
LLMConfigManager - Secure management of LLM provider configurations.
"""
from typing import Optional, Dict, List, Any
from pathlib import Path
import json
import os
from datetime import datetime

from .models import LLMProviderConfig


class LLMConfigManager:
    """
    Manage LLM provider configurations securely.
    
    Responsibilities:
    - Store and load provider configurations
    - Handle API key storage (memory-only or env vars)
    - Test provider connections
    - Validate configuration
    - Support multiple providers
    """    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize LLMConfigManager.
        
        Args:
            config_dir: Directory for storing configs (default: ~/.rlmkit)
        """
        if config_dir is None:
            config_dir = Path.home() / ".rlmkit"
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, mode=0o700)
        
        self._configs: Dict[str, LLMProviderConfig] = {}
        self._active_provider: Optional[str] = None
        self._load_all_configs()
    
    def add_provider(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
        input_cost_per_1k: float = 0.0,
        output_cost_per_1k: float = 0.0,
        **kwargs
    ) -> tuple:
        """
        Add or update a provider configuration.
        
        Args:
            provider: Provider name (openai, anthropic, ollama, lmstudio)
            model: Model name/ID
            api_key: API key (optional, stored in memory only)
            api_key_env_var: Environment variable name for API key
            input_cost_per_1k: Input cost per 1K tokens
            output_cost_per_1k: Output cost per 1K tokens
            **kwargs: Additional config (temperature, max_tokens, etc)
        
        Returns:
            Tuple of (success: bool, error_message: str)
            If successful: (True, "")
            If failed: (False, "error reason")
            
        Implementation notes:
        - Should validate at least one of api_key or api_key_env_var
        - Should test connection before saving
        - Should never save api_key to disk (only env_var)
        - Returns error message if connection test fails
        """
        if not api_key and not api_key_env_var:
            return False, "Must provide either api_key or api_key_env_var"
        
        config = LLMProviderConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            api_key_env_var=api_key_env_var,
            input_cost_per_1k_tokens=input_cost_per_1k,
            output_cost_per_1k_tokens=output_cost_per_1k,
            **kwargs
        )
        
        # Test connection before saving
        success, error_msg = self.test_connection(provider, model, api_key, api_key_env_var)
        if not success:
            print(f"DEBUG: test_connection returned False: {error_msg}")
            return False, error_msg
        
        # Mark test as successful
        config.test_successful = True
        print(f"DEBUG: Set test_successful=True for {provider}")
        
        # Save configuration
        self._save_config(config)
        self._configs[provider] = config
        print(f"DEBUG: Saved config. is_ready={config.is_ready}, test_successful={config.test_successful}")
        
        # Set as active provider (auto-select first provider added)
        if not self._active_provider:
            self._active_provider = provider
        
        return True, ""
    
    def get_provider_config(self, provider: str) -> Optional[LLMProviderConfig]:
        """
        Get configuration for a provider.
        
        Args:
            provider: Provider name
        
        Returns:
            LLMProviderConfig or None if not found
        """
        if provider not in self._configs:
            self._load_config(provider)
        return self._configs.get(provider)
    
    def list_providers(self) -> List[str]:
        """Get list of configured providers."""
        return list(self._configs.keys())
    
    def set_active_provider(self, provider: str) -> bool:
        """
        Set active provider.
        
        Args:
            provider: Provider name
        
        Returns:
            True if provider exists and is ready
        """
        config = self.get_provider_config(provider)
        if config and config.is_ready:
            self._active_provider = provider
            return True
        return False
    
    def get_active_provider(self) -> Optional[LLMProviderConfig]:
        """Get currently active provider configuration."""
        if self._active_provider:
            return self.get_provider_config(self._active_provider)
        return None
    
    def test_connection(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        api_key_env_var: Optional[str] = None,
    ) -> tuple:
        """
        Test connection to a provider.
        
        Args:
            provider: Provider name
            model: Model name
            api_key: API key (optional)
            api_key_env_var: Environment variable name
        
        Returns:
            Tuple of (success: bool, error_message: str)
            If successful: (True, "")
            If failed: (False, "detailed error message")
            
        Example:
            >>> manager = LLMConfigManager()
            >>> success, error = manager.test_connection(
            ...     provider="openai",
            ...     model="gpt-4",
            ...     api_key="sk-test-key..."
            ... )
            >>> if not success:
            ...     print(f"Error: {error}")
        """
        # Validate inputs
        if not provider or not model:
            return False, "Provider and model are required"
        
        # Get API key from environment if needed
        effective_api_key = api_key
        if not effective_api_key and api_key_env_var:
            effective_api_key = os.getenv(api_key_env_var)
        
        # If no API key, only allow for local providers (ollama, lmstudio)
        if not effective_api_key and provider not in ("ollama", "lmstudio"):
            env_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "API_KEY")
            return False, f"No API key provided. Set environment variable {env_var} or provide API key."
        
        supported_providers = {
            "openai": self._test_openai_connection,
            "anthropic": self._test_anthropic_connection,
            "ollama": self._test_ollama_connection,
            "lmstudio": self._test_lmstudio_connection,
        }
        
        if provider not in supported_providers:
            return False, f"Unsupported provider: {provider}"
        
        try:
            tester = supported_providers[provider]
            result = tester(model, effective_api_key or "")
            if isinstance(result, tuple):
                return result
            return (True, "") if result else (False, f"Connection to {provider} failed")
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def _test_openai_connection(self, model: str, api_key: str) -> tuple:
        """
        Test OpenAI connection with a dummy API request.
        
        Args:
            model: Model name
            api_key: API key
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        # Validate API key exists
        if not api_key or api_key.strip() == "":
            return False, "API key is required for OpenAI"
        
        # Try to get LLM client and make a test call
        try:
            print(f"DEBUG: Testing OpenAI connection with model {model}")
            from rlmkit.llm import get_llm_client
            client = get_llm_client(
                provider="openai",
                model=model,
                api_key=api_key
            )
            print(f"DEBUG: OpenAI client created successfully")
            # Make minimal test request
            response = client.complete([
                {"role": "user", "content": "Hello"}
            ])
            print(f"DEBUG: OpenAI API response received: {len(response) if response else 0} chars")
            if response:
                return True, ""
            else:
                return False, "OpenAI API returned empty response"
        except Exception as e:
            error_msg = str(e)
            print(f"DEBUG: OpenAI connection error: {error_msg}")
            # Extract the most relevant part of the error
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return False, "Invalid API key (401 Unauthorized). Check your OPENAI_API_KEY."
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                return False, "Access forbidden (403). Check your account permissions or billing."
            elif "rate_limit" in error_msg.lower():
                return False, "Rate limit exceeded. Wait a moment and try again."
            else:
                return False, f"OpenAI connection failed: {error_msg}"
    
    def _test_anthropic_connection(self, model: str, api_key: str) -> tuple:
        """
        Test Anthropic connection with a dummy API request.
        
        Args:
            model: Model name
            api_key: API key
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        # Validate API key exists
        if not api_key or api_key.strip() == "":
            return False, "API key is required for Anthropic"
        
        # Try to get LLM client and make a test call
        try:
            from rlmkit.llm import get_llm_client
            client = get_llm_client(
                provider="anthropic",
                model=model,
                api_key=api_key
            )
            # Make minimal test request
            response = client.complete([
                {"role": "user", "content": "Hello"}
            ])
            if response:
                return True, ""
            else:
                return False, "Anthropic API returned empty response"
        except Exception as e:
            error_msg = str(e)
            # Extract the most relevant part of the error
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                return False, "Invalid API key (401 Unauthorized). Check your ANTHROPIC_API_KEY."
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                return False, "Access forbidden (403). Check your account permissions or billing."
            elif "rate_limit" in error_msg.lower() or "overloaded" in error_msg.lower():
                return False, "Rate limit exceeded or API overloaded. Wait a moment and try again."
            else:
                return False, f"Anthropic connection failed: {error_msg}"
    
    def _test_ollama_connection(self, model: str, api_key: str) -> tuple:
        """
        Test Ollama connection with HTTP request.
        
        Args:
            model: Model name
            api_key: API key (unused for Ollama, but kept for consistency)
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        # Ollama doesn't need API key, but check if any model name provided
        if not model:
            return False, "Model name is required"
        
        # Test connection to Ollama server
        try:
            import requests
            response = requests.get(
                "http://localhost:11434/api/tags",
                timeout=2
            )
            if response.status_code != 200:
                return False, f"Ollama server returned status {response.status_code}"
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            if model in model_names:
                return True, ""
            else:
                available = ", ".join(model_names[:3])
                return False, f"Model '{model}' not found on Ollama. Available: {available}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama server. Is it running on localhost:11434?"
        except requests.exceptions.Timeout:
            return False, "Ollama server timed out. Is it responding?"
        except Exception as e:
            return False, f"Ollama connection failed: {str(e)}"
    
    def _test_lmstudio_connection(self, model: str, api_key: str) -> tuple:
        """
        Test LM Studio connection with HTTP request.
        
        Args:
            model: Model name
            api_key: API key (unused for LMStudio, but kept for consistency)
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        # LMStudio doesn't need API key, but check if any model name provided
        if not model:
            return False, "Model name is required"
        
        # Test connection to LM Studio server
        try:
            import requests
            response = requests.get(
                "http://localhost:1234/v1/models",
                timeout=2
            )
            if response.status_code != 200:
                return False, f"LM Studio server returned status {response.status_code}"
            
            # Check if model is available
            models = response.json().get("data", [])
            model_ids = [m.get("id", "") for m in models]
            if model in model_ids:
                return True, ""
            else:
                available = ", ".join(model_ids[:3])
                return False, f"Model '{model}' not found on LM Studio. Available: {available}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to LM Studio server. Is it running on localhost:1234?"
        except requests.exceptions.Timeout:
            return False, "LM Studio server timed out. Is it responding?"
        except Exception as e:
            return False, f"LM Studio connection failed: {str(e)}"
    
    def delete_provider(self, provider: str) -> bool:
        """
        Delete provider configuration.
        
        Args:
            provider: Provider name
        
        Returns:
            True if deleted, False if not found
        """
        config_file = self.config_dir / f"{provider}.json"
        if config_file.exists():
            config_file.unlink()
            if provider in self._configs:
                del self._configs[provider]
            return True
        return False
    
    def _save_config(self, config: LLMProviderConfig) -> None:
        """
        Save configuration to disk (without API key).
        
        Args:
            config: LLMProviderConfig to save
            
        Implementation notes:
        - Should never save api_key field (only api_key_env_var)
        - Should set file permissions to 0o600 (Unix only)
        - Should be JSON format
        """
        config_file = self.config_dir / f"{config.provider}.json"
        
        # Get dict without API key
        config_dict = config.to_dict(include_api_key=False)
        
        # Save to JSON
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        # Restrict file permissions (Unix only)
        try:
            config_file.chmod(0o600)
        except Exception:
            pass  # Windows doesn't support chmod
    
    def _load_config(self, provider: str) -> None:
        """
        Load configuration from disk.
        
        Args:
            provider: Provider name
            
        Implementation notes:
        - Should load from {config_dir}/{provider}.json
        - Should load API key from environment variable if specified
        - Should handle missing files gracefully
        """
        config_file = self.config_dir / f"{provider}.json"
        if not config_file.exists():
            return
        
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        
        # Load API key from environment variable if specified
        api_key_env_var = config_dict.get("api_key_env_var")
        if api_key_env_var:
            config_dict["api_key"] = os.getenv(api_key_env_var)
        
        config = LLMProviderConfig(**config_dict)
        self._configs[provider] = config
    
    def _load_all_configs(self) -> None:
        """Load all provider configurations from disk."""
        for config_file in self.config_dir.glob("*.json"):
            provider = config_file.stem
            if provider != "encryption":  # Skip encryption key file
                self._load_config(provider)
