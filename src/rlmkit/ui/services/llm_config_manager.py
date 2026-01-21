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
    ) -> bool:
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
            True if provider was added successfully, False otherwise
            
        Implementation notes:
        - Should validate at least one of api_key or api_key_env_var
        - Should test connection before saving
        - Should never save api_key to disk (only env_var)
        - Should return False if connection test fails
        """
        if not api_key and not api_key_env_var:
            raise ValueError("Must provide either api_key or api_key_env_var")
        
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
        if not self.test_connection(provider, model, api_key, api_key_env_var):
            return False
        
        # Save configuration
        self._save_config(config)
        self._configs[provider] = config
        
        return True
    
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
    ) -> bool:
        """
        Test connection to a provider.
        
        Args:
            provider: Provider name
            model: Model name
            api_key: API key (optional)
            api_key_env_var: Environment variable name
        
        Returns:
            True if connection successful, False otherwise
            
        Example:
            >>> manager = LLMConfigManager()
            >>> success = manager.test_connection(
            ...     provider="openai",
            ...     model="gpt-4",
            ...     api_key="sk-test-key..."
            ... )
            >>> print(success)
            True or False
        """
        # Validate inputs
        if not provider or not model:
            return False
        
        # Get API key from environment if needed
        effective_api_key = api_key
        if not effective_api_key and api_key_env_var:
            effective_api_key = os.getenv(api_key_env_var)
        
        # If no API key, only allow for local providers (ollama, lmstudio)
        if not effective_api_key and provider not in ("ollama", "lmstudio"):
            return False
        
        # LATER: Implement real API calls for each provider
        # For now, validate known providers with mock responses
        supported_providers = {
            "openai": self._test_openai_connection,
            "anthropic": self._test_anthropic_connection,
            "ollama": self._test_ollama_connection,
            "lmstudio": self._test_lmstudio_connection,
        }
        
        if provider not in supported_providers:
            return False
        
        try:
            tester = supported_providers[provider]
            return tester(model, effective_api_key or "")
        except Exception:
            return False
    
    def _test_openai_connection(self, model: str, api_key: str) -> bool:
        """
        Test OpenAI connection.
        
        Args:
            model: Model name
            api_key: API key
        
        Returns:
            True if connection successful
            
        Implementation notes:
        - LATER: Replace with actual OpenAI API call
        - For now: Validate key format and model
        """
        # Validate API key format (starts with "sk-")
        if not api_key.startswith("sk-"):
            return False
        
        # LATER: Make actual API call to test connection
        # For testing: just validate model is known
        valid_models = [
            "gpt-4", "gpt-4-32k", "gpt-4-turbo", "gpt-4-turbo-preview",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        ]
        
        if model not in valid_models:
            return False
        
        return True
    
    def _test_anthropic_connection(self, model: str, api_key: str) -> bool:
        """
        Test Anthropic connection.
        
        Args:
            model: Model name
            api_key: API key
        
        Returns:
            True if connection successful
            
        Implementation notes:
        - LATER: Replace with actual Anthropic API call
        - For now: Validate key format and model
        """
        # Validate API key format (starts with "sk-ant-")
        if not api_key.startswith("sk-ant-"):
            return False
        
        # LATER: Make actual API call to test connection
        # For testing: just validate model is known
        valid_models = [
            "claude-3-opus", "claude-3-sonnet", "claude-3-haiku",
            "claude-2.1", "claude-2"
        ]
        
        if model not in valid_models:
            return False
        
        return True
    
    def _test_ollama_connection(self, model: str, api_key: str) -> bool:
        """
        Test Ollama connection.
        
        Args:
            model: Model name
            api_key: API key (unused for Ollama, but kept for consistency)
        
        Returns:
            True if connection successful
            
        Implementation notes:
        - LATER: Replace with actual HTTP request to Ollama server
        - For now: Just validate model name exists
        - Ollama typically runs on localhost:11434
        """
        # Ollama doesn't need API key, but check if any model name provided
        if not model:
            return False
        
        # LATER: Make actual HTTP request to http://localhost:11434/api/tags
        # For testing: accept any model name
        return True
    
    def _test_lmstudio_connection(self, model: str, api_key: str) -> bool:
        """
        Test LMStudio connection.
        
        Args:
            model: Model name
            api_key: API key (unused for LMStudio, but kept for consistency)
        
        Returns:
            True if connection successful
            
        Implementation notes:
        - LATER: Replace with actual HTTP request to LMStudio server
        - For now: Just validate model name exists
        - LMStudio typically runs on localhost:1234
        """
        # LMStudio doesn't need API key, but check if any model name provided
        if not model:
            return False
        
        # LATER: Make actual HTTP request to http://localhost:1234/v1/models
        # For testing: accept any model name
        return True
    
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
