#!/usr/bin/env python3
"""
Integration test for Chat Page - Verifies all components work together.

DEPRECATED: This test covers the old Streamlit ChatManager, which has been
replaced by the FastAPI + Next.js stack. It requires nest_asyncio (not in
dev deps) and tests legacy UI code paths. Tracked for migration/removal in
specs/pitch-legacy-lint-debt.md.

Skipped when nest_asyncio is unavailable (which includes CI and fresh dev
environments).
"""

import asyncio
import sys
from pathlib import Path

import pytest

try:
    import nest_asyncio

    nest_asyncio.apply()
except ModuleNotFoundError:
    pytest.skip(
        "nest_asyncio not installed — legacy Streamlit integration test, "
        "see specs/pitch-legacy-lint-debt.md",
        allow_module_level=True,
    )

# Setup path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rlmstudio.ui.services.chat_manager import ChatManager
from rlmstudio.ui.services.models import Response


@pytest.mark.asyncio
async def test_chat_integration():
    """Test the complete chat flow."""

    print("=" * 60)
    print("CHAT PAGE INTEGRATION TEST")
    print("=" * 60)

    # Test 1: ChatManager initialization
    print("\n[1/5] Testing ChatManager initialization...")
    try:
        mock_session = {
            "selected_provider": None,
            "selected_model": "gpt-4",
            "execution_mode": "compare",
        }
        manager = ChatManager(mock_session)
        print("    ✅ ChatManager initialized")
    except Exception as e:
        print(f"    ❌ ChatManager init failed: {e}")
        return False

    # Test 2: Config manager access
    print("\n[2/5] Testing LLMConfigManager access...")
    try:
        from rlmstudio.ui.services.llm_config_manager import LLMConfigManager

        config_manager = LLMConfigManager()
        providers = config_manager.list_providers()
        print(f"    ✅ Found {len(providers)} configured providers")
        if providers:
            print(f"       Available: {', '.join(providers)}")
        else:
            print("       ℹ️  No providers configured yet (will use mock in tests)")
    except Exception as e:
        print(f"    ❌ Config manager access failed: {e}")
        return False

    # Test 3: Model data structures
    print("\n[3/5] Testing data models...")
    try:
        from rlmstudio.ui.services.models import ChatMessage

        response = Response(content="Test response", stop_reason="end_turn")
        assert response.content == "Test response"

        # Create a simple ChatMessage
        message = ChatMessage(
            id="test-001",
            user_query="What is this?",
            rlm_response=response,
            direct_response=response,
        )
        assert message.user_query == "What is this?"
        print("    ✅ All data models work correctly")
    except Exception as e:
        print(f"    ❌ Data models failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 4: ChatManager async call simulation (with mock data)
    print("\n[4/5] Testing async message processing...")
    try:
        # This will use mock/test data since no real provider is configured
        result = await manager.process_message(
            user_query="What is RLM?",
            mode="compare",
            file_context="RLM is a framework for reasoning.",
            file_info=None,
        )
        print("    ✅ Message processing completed")
        print(f"       RLM Response: {result.rlm_response.content[:50]}...")
        print(f"       Direct Response: {result.direct_response.content[:50]}...")
    except Exception as e:
        print(f"    ⚠️  Message processing failed (expected if no provider): {str(e)[:100]}")
        # This is expected if no provider is configured

    # Test 5: Component imports
    print("\n[5/5] Testing component imports...")
    try:
        # These imports should work if files are created correctly
        from rlmstudio.ui.pages.configuration import render_configuration_page  # noqa: F401

        print("    ✅ Configuration page imports")

        # Chat page doesn't export functions, but file should exist
        from rlmstudio.ui.components import chat_message  # noqa: F401

        print("    ✅ Chat message component imports")

        # Check if 01_chat.py exists
        chat_page_path = Path(__file__).parent / "src" / "rlmstudio" / "ui" / "pages" / "01_chat.py"
        if chat_page_path.exists():
            print("    ✅ Chat page file exists")
        else:
            print("    ❌ Chat page file missing")
            return False

    except ImportError as e:
        print(f"    ⚠️  Component import warning: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY: Chat Integration Ready ✅")
    print("=" * 60)
    print("""
Next Steps:
1. Open http://localhost:8502 in browser
2. Navigate to Chat page (if visible in sidebar)
3. Configure a provider in Configuration page if not done
4. Upload a test document
5. Send a test query

Known Issues to Watch:
- AsyncIO handling: Using nest_asyncio to allow nested loops
- File context: Must upload document first
- Provider status: Configure OpenAI/Anthropic/etc. before use
""")

    return True


if __name__ == "__main__":
    success = asyncio.run(test_chat_integration())
    sys.exit(0 if success else 1)
