#!/usr/bin/env python3
"""
Test script to verify API key loading fix.
"""
from pathlib import Path
import os
import tempfile

def test_env_loading():
    """Test that .env file is properly loaded."""
    from src.rlmkit.ui.services.llm_config_manager import load_env_file, update_env_file
    
    # Create a temporary .env file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        env_path = Path(f.name)
        f.write("TEST_API_KEY=sk-test123456\n")
        f.write("OPENAI_API_KEY=sk-openai-test\n")
    
    try:
        # Clear any existing env vars
        if 'TEST_API_KEY' in os.environ:
            del os.environ['TEST_API_KEY']
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        # Load the env file
        load_env_file(env_path)
        
        # Verify the keys were loaded
        assert os.getenv('TEST_API_KEY') == 'sk-test123456', "TEST_API_KEY not loaded correctly"
        assert os.getenv('OPENAI_API_KEY') == 'sk-openai-test', "OPENAI_API_KEY not loaded correctly"
        
        print("[PASS] Environment variables loaded successfully from .env file")
        
        # Test updating env file
        update_env_file('TEST_API_KEY', 'sk-updated', env_path)
        
        # Reload and verify
        load_env_file(env_path)
        assert os.getenv('TEST_API_KEY') == 'sk-updated', "TEST_API_KEY not updated correctly"
        
        print("[PASS] Environment variable updated successfully in .env file")
        
    finally:
        # Clean up
        env_path.unlink(missing_ok=True)
        if 'TEST_API_KEY' in os.environ:
            del os.environ['TEST_API_KEY']
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

def test_config_loading():
    """Test that provider config properly loads API keys."""
    from src.rlmkit.ui.services.llm_config_manager import LLMConfigManager
    import tempfile
    import json
    
    # Create temporary config directory
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        
        # Create .env file in current directory
        env_path = Path(".env")
        original_exists = env_path.exists()
        original_content = None
        if original_exists:
            original_content = env_path.read_text()
        
        try:
            # Write test env vars
            with open(env_path, 'w') as f:
                f.write("OPENAI_API_KEY=sk-test-from-env\n")
            
            # Create a provider config file
            config_file = config_dir / "openai.json"
            config_data = {
                "provider": "openai",
                "model": "gpt-4",
                "api_key_env_var": "OPENAI_API_KEY",
                "input_cost_per_1k_tokens": 0.03,
                "output_cost_per_1k_tokens": 0.06,
                "test_successful": True,
                "enabled": True
            }
            with open(config_file, 'w') as f:
                json.dump(config_data, f)
            
            # Create config manager
            config_manager = LLMConfigManager(config_dir=config_dir)
            
            # Load the provider config
            provider_config = config_manager.get_provider_config("openai")
            
            # Verify API key was loaded from env
            assert provider_config is not None, "Provider config not loaded"
            assert provider_config.api_key == "sk-test-from-env", f"API key not loaded from .env: {provider_config.api_key}"
            
            print("[PASS] Provider configuration loaded API key from environment successfully")
            
        finally:
            # Restore original .env or remove test one
            if original_exists and original_content:
                env_path.write_text(original_content)
            elif env_path.exists():
                env_path.unlink()

if __name__ == "__main__":
    print("Testing API key loading fix...")
    print()
    
    try:
        test_env_loading()
        print()
        test_config_loading()
        print()
        print("=" * 60)
        print("[SUCCESS] All tests passed! API key loading is working correctly.")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"[FAILED] Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)
