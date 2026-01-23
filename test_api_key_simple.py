#!/usr/bin/env python3
"""
Simple test to verify API key loading fix without requiring package installation.
"""
from pathlib import Path
import os
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_load_env_file():
    """Test the load_env_file function."""
    # Import the module directly
    from rlmkit.ui.services.llm_config_manager import load_env_file
    
    # Create a temporary .env file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        env_path = Path(f.name)
        f.write("TEST_KEY_1=value1\n")
        f.write("TEST_KEY_2=value2\n")
        f.write("# This is a comment\n")
        f.write("TEST_KEY_3=value3\n")
    
    try:
        # Clear any existing test env vars
        for key in ['TEST_KEY_1', 'TEST_KEY_2', 'TEST_KEY_3']:
            if key in os.environ:
                del os.environ[key]
        
        # Load the env file
        load_env_file(env_path)
        
        # Verify the keys were loaded
        assert os.getenv('TEST_KEY_1') == 'value1', "TEST_KEY_1 not loaded"
        assert os.getenv('TEST_KEY_2') == 'value2', "TEST_KEY_2 not loaded"
        assert os.getenv('TEST_KEY_3') == 'value3', "TEST_KEY_3 not loaded"
        
        print("[PASS] load_env_file() correctly loads environment variables from .env file")
        return True
        
    finally:
        # Clean up
        env_path.unlink(missing_ok=True)
        for key in ['TEST_KEY_1', 'TEST_KEY_2', 'TEST_KEY_3']:
            if key in os.environ:
                del os.environ[key]

def test_update_env_file():
    """Test the update_env_file function."""
    from rlmkit.ui.services.llm_config_manager import update_env_file, load_env_file
    
    # Create a temporary .env file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        env_path = Path(f.name)
        f.write("EXISTING_KEY=old_value\n")
    
    try:
        # Update an existing key
        update_env_file('EXISTING_KEY', 'new_value', env_path)
        
        # Add a new key
        update_env_file('NEW_KEY', 'new_value', env_path)
        
        # Reload and verify
        load_env_file(env_path)
        
        assert os.getenv('EXISTING_KEY') == 'new_value', "EXISTING_KEY not updated"
        assert os.getenv('NEW_KEY') == 'new_value', "NEW_KEY not added"
        
        print("[PASS] update_env_file() correctly updates and adds keys to .env file")
        return True
        
    finally:
        # Clean up
        env_path.unlink(missing_ok=True)
        for key in ['EXISTING_KEY', 'NEW_KEY']:
            if key in os.environ:
                del os.environ[key]

def test_integration():
    """Test the integration scenario that was failing."""
    print("\nSimulating the original error scenario:")
    print("1. User configures OpenAI provider with API key")
    print("2. API key is saved to .env file")
    print("3. Config manager loads provider config")
    print("4. Chat manager tries to use the API key")
    print()
    
    from rlmkit.ui.services.llm_config_manager import update_env_file, load_env_file
    
    # Create a test .env file
    env_path = Path(".env.test")
    
    try:
        # Step 1: Simulate saving API key to .env
        update_env_file('OPENAI_API_KEY', 'sk-test-key-12345', env_path)
        print("[STEP 1] API key written to .env file")
        
        # Step 2: Simulate loading config (which should load .env)
        # Clear the env var first to simulate fresh start
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        # Load .env file
        load_env_file(env_path)
        print("[STEP 2] .env file loaded into environment")
        
        # Step 3: Verify API key is now available
        api_key = os.getenv('OPENAI_API_KEY')
        assert api_key == 'sk-test-key-12345', f"Expected 'sk-test-key-12345', got '{api_key}'"
        print("[STEP 3] API key successfully retrieved from environment")
        
        print("\n[PASS] Integration test: API key flow works correctly")
        print("       The original error 'Provider openai is configured but API key is missing'")
        print("       should now be FIXED!")
        return True
        
    finally:
        # Clean up
        env_path.unlink(missing_ok=True)
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']

if __name__ == "__main__":
    print("=" * 70)
    print("Testing API Key Loading Fix")
    print("=" * 70)
    print()
    
    try:
        # Run tests
        test_load_env_file()
        print()
        test_update_env_file()
        print()
        test_integration()
        
        print()
        print("=" * 70)
        print("[SUCCESS] All tests passed!")
        print("=" * 70)
        print()
        print("Summary of fix:")
        print("- Added load_env_file() function to load .env into environment")
        print("- Modified _load_config() to call load_env_file() before loading")
        print("- Modified chat_manager to call load_env_file() when needed")
        print("- API keys are now properly loaded from .env file")
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"[FAILED] Test failed: {e}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        sys.exit(1)
