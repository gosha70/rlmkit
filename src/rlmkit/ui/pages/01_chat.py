# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
Chat Page - Main interactive chat interface
Page 1 of RLM Studio: ChatGPT-like interface for exploring documents
"""

import streamlit as st
import asyncio

from rlmkit.ui.services.chat_manager import ChatManager


def _run_async(coro):
    """
    Run an async coroutine in Streamlit context.
    Handles various event loop configurations including uvloop.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            # Loop is closed, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        else:
            # Loop exists and is running
            # For uvloop or other incompatible loops, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Run in separate thread to avoid event loop conflicts
                return executor.submit(asyncio.run, coro).result()
    except RuntimeError:
        # No event loop in current thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def render_chat_page():
    """Render the main chat interface page."""
    
    # Render sidebar with provider information
    render_sidebar()
    
    st.title("💬 RLM Studio Chat")
    st.markdown("Interactive exploration of your documents using RLM vs Direct LLM comparison")
    
    # Initialize session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None
    
    # Render document context
    render_document_context()
    
    st.divider()
    
    # Render message history
    render_message_history()
    
    st.divider()
    
    # Render input area
    render_chat_input()


def render_sidebar():
    """Display provider and configuration info in sidebar."""
    from rlmkit.ui.services import LLMConfigManager
    from pathlib import Path
    
    st.sidebar.header("⚙️ Current Setup")
    
    # Initialize ChatManager if not already done
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager()
    
    # Get provider information
    config_dir = Path.home() / ".rlmkit"
    manager = LLMConfigManager(config_dir=config_dir)
    
    # Get selected provider
    selected_provider = st.session_state.get('selected_provider')
    if not selected_provider:
        providers = manager.list_providers()
        selected_provider = providers[0] if providers else None
        if selected_provider:
            st.session_state.selected_provider = selected_provider
    
    if selected_provider:
        config = manager.get_provider_config(selected_provider)
        
        st.sidebar.subheader("🤖 Selected LLM")
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            st.sidebar.markdown(f"**Provider:**")
        with col2:
            st.sidebar.markdown(f"`{selected_provider.upper()}`")
        
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            st.sidebar.markdown(f"**Model:**")
        with col2:
            st.sidebar.markdown(f"`{config.model}`")
        
        col1, col2 = st.sidebar.columns([1, 1])
        with col1:
            st.sidebar.markdown(f"**Status:**")
        with col2:
            if config.is_ready:
                st.sidebar.success("✅ Ready")
            else:
                st.sidebar.warning("⚠️ Not Ready")
        
        st.sidebar.markdown("[🔧 Change settings](./configuration)")
    else:
        st.sidebar.warning("⚠️ No provider configured. Go to [Configuration](./configuration) to set up a provider.")


def render_document_context():
    """Show currently loaded document/context."""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.session_state.get('file_info'):
            file_info = st.session_state.file_info
            st.info(
                f"📄 **Document Context Loaded**\n\n"
                f"**File:** {file_info.filename}\n"
                f"**Size:** {file_info.size_bytes:,} bytes | "
                f"**Chars:** {file_info.char_count:,} | "
                f"**Tokens:** ~{file_info.estimated_tokens:,}"
            )
        else:
            st.warning("📄 No document context loaded. Upload or paste content to get started.")
    
    with col2:
        if st.button("🔄 Clear", key="clear_context"):
            st.session_state.file_content = None
            st.session_state.file_info = None
            st.success("Context cleared")
            st.rerun()


def render_message_history():
    """Display conversation history."""
    messages = st.session_state.get('chat_messages', [])
    
    if not messages:
        st.info("💭 No messages yet. Start by asking a question about your document.")
        return
    
    # Create container for scrollable messages
    message_container = st.container()
    
    with message_container:
        for i, message in enumerate(messages):
            if message['role'] == 'user':
                render_user_message(message, i)
            else:
                render_assistant_message(message, i)


def render_user_message(message: dict, index: int):
    """Render a user message."""
    with st.chat_message("user", avatar="👤"):
        st.markdown(message['content'])
        
        # Show metadata
        if message.get('timestamp'):
            st.caption(f"⏰ {message['timestamp']}")


def render_assistant_message(message: dict, index: int):
    """Render an assistant message with both RLM and Direct responses."""
    mode = message.get('mode', 'compare')
    
    # Create tabs for different response types
    if mode == 'compare' and message.get('rlm_response') and message.get('direct_response'):
        tab1, tab2, tab3 = st.tabs(["⚙️ RLM (Multi-step)", "📋 Direct (Single)", "📊 Comparison"])
        
        with tab1:
            render_rlm_response(message)
        
        with tab2:
            render_direct_response(message)
        
        with tab3:
            render_comparison(message)
    
    elif mode == 'rlm_only' and message.get('rlm_response'):
        render_rlm_response(message)
    
    elif mode == 'direct_only' and message.get('direct_response'):
        render_direct_response(message)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📋 Copy", key=f"copy_{index}"):
            st.success("Copied to clipboard!")
    with col2:
        if st.button("🔄 Regenerate", key=f"regen_{index}"):
            st.info("Regeneration not yet implemented")
    with col3:
        if st.button("⬇️ Export", key=f"export_{index}"):
            st.info("Export not yet implemented")


def render_rlm_response(message: dict):
    """Render RLM response with metrics."""
    with st.chat_message("assistant", avatar="⚙️"):
        response = message.get('rlm_response')
        metrics = message.get('rlm_metrics')
        trace = message.get('rlm_trace', [])

        if response:
            if response.content:
                # Show error differently if it's an error response
                if response.stop_reason == "error" or "failed:" in response.content.lower():
                    st.error(response.content)
                else:
                    st.markdown(response.content)
            else:
                st.warning("⚠️ No response content available")
        else:
            st.error("❌ No response object found")

        # Show metrics
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Steps", metrics.steps_taken)
            with col2:
                st.metric("Tokens", f"{metrics.total_tokens:,}")
            with col3:
                st.metric("Time", f"{metrics.execution_time_seconds:.2f}s")
            with col4:
                st.metric("Cost", f"${metrics.cost_usd:.4f}")

        # Show step indicator
        if trace:
            with st.expander(f"📈 Execution Trace ({len(trace)} steps)"):
                for i, step in enumerate(trace, 1):
                    # Generate description from role and content
                    role = step.get('role', 'unknown')
                    content = step.get('content', '')

                    if role == 'assistant':
                        # LLM response step - show preview
                        preview = content[:100] + "..." if len(content) > 100 else content
                        description = f"LLM response: {preview}"
                    elif role == 'execution':
                        # Code execution step - show result preview
                        preview = content[:80] + "..." if len(content) > 80 else content
                        description = f"Code execution: {preview}"
                    else:
                        # Fallback
                        description = step.get('description', 'N/A')

                    st.write(f"**Step {i}:** {description}")

                    # Show token/cost info if available
                    if step.get('input_tokens') or step.get('output_tokens'):
                        st.caption(
                            f"Tokens: {step.get('input_tokens', 0)} → {step.get('output_tokens', 0)} | "
                            f"Cost: ${step.get('cost', 0):.4f} | "
                            f"Time: {step.get('duration_ms', 0)}ms"
                        )


def render_direct_response(message: dict):
    """Render Direct LLM response with metrics."""
    with st.chat_message("assistant", avatar="📋"):
        response = message.get('direct_response')
        metrics = message.get('direct_metrics')

        if response:
            if response.content:
                # Show error differently if it's an error response
                if response.stop_reason == "error" or "failed:" in response.content.lower():
                    st.error(response.content)
                else:
                    st.markdown(response.content)
            else:
                st.warning("⚠️ No response content available")
        else:
            st.error("❌ No response object found")

        # Show metrics
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Steps", metrics.steps_taken)
            with col2:
                st.metric("Tokens", f"{metrics.total_tokens:,}")
            with col3:
                st.metric("Time", f"{metrics.execution_time_seconds:.2f}s")
            with col4:
                st.metric("Cost", f"${metrics.cost_usd:.4f}")


def render_comparison(message: dict):
    """Render comparison between RLM and Direct."""
    comparison = message.get('comparison_metrics')

    if not comparison:
        st.info("No comparison data available")
        return

    # Create comparison table
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚙️ RLM")
        st.metric("Tokens", f"{comparison.rlm_tokens:,}")
        st.metric("Cost", f"${comparison.rlm_cost_usd:.4f}")
        st.metric("Time", f"{comparison.rlm_time_seconds:.2f}s")
        st.metric("Steps", comparison.rlm_steps)

    with col2:
        st.subheader("📋 Direct")
        st.metric("Tokens", f"{comparison.direct_tokens:,}")
        st.metric("Cost", f"${comparison.direct_cost_usd:.4f}")
        st.metric("Time", f"{comparison.direct_time_seconds:.2f}s")
        st.metric("Steps", comparison.direct_steps)

    with col3:
        st.subheader("📊 Deltas")
        st.metric("Token Diff", f"{comparison.token_delta:,}")
        st.metric("Cost Diff", f"${comparison.cost_delta_usd:.4f} ({comparison.cost_delta_percent:.1f}%)")
        st.metric("Time Diff", f"{comparison.time_delta_seconds:.2f}s ({comparison.time_delta_percent:.1f}%)")
        # Note: quality_delta doesn't exist in ComparisonMetrics, removed

    # Only show recommendation when both responses actually succeeded
    # Check for real failures: missing content or error stop_reason
    if comparison.recommendation:
        # Check if responses actually have valid content
        rlm_response = message.get('rlm_response')
        direct_response = message.get('direct_response')

        rlm_failed = not rlm_response or not rlm_response.content or rlm_response.stop_reason == "error"
        direct_failed = not direct_response or not direct_response.content or direct_response.stop_reason == "error"

        if not rlm_failed and not direct_failed:
            # Both responses are valid - show recommendation
            st.success(f"💡 **Recommendation:** {comparison.recommendation}")
        else:
            # One or both responses actually failed
            st.warning("⚠️ Comparison data may be incomplete. One or both executions may have failed.")


def render_chat_input():
    """Render the chat input area."""
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_area(
            "Your question:",
            placeholder="Ask anything about the document...",
            height=100,
            key="chat_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")
        
        # Mode selector
        mode = st.radio(
            "Mode:",
            ["Compare", "RLM Only", "Direct Only"],
            horizontal=False,
            key="execution_mode_selector"
        )
        
        mode_map = {
            "Compare": "compare",
            "RLM Only": "rlm_only",
            "Direct Only": "direct_only"
        }
        
        send_button = st.button("📤 Send", use_container_width=True)
    
    # Handle message submission
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.chat_messages.append({
            'role': 'user',
            'content': user_input,
            'timestamp': st.session_state.get('current_time')
        })
        
        # Show loading state
        with st.spinner("Analyzing..."):
            try:
                # Get ChatManager from session state
                chat_manager = st.session_state.get('chat_manager')
                if not chat_manager:
                    st.error("Chat manager not initialized")
                    return
                
                # Get the API key from session state storage or config
                selected_provider = st.session_state.get('selected_provider')
                api_key = None
                if selected_provider:
                    # First try session state storage (from Configuration page)
                    if 'provider_api_keys' in st.session_state:
                        api_key = st.session_state.provider_api_keys.get(selected_provider)
                    
                    # Fallback to config (for env var based setups or reload)
                    if not api_key:
                        llm_manager = st.session_state.get('llm_manager')
                        if llm_manager:
                            provider_config = llm_manager.get_provider_config(selected_provider)
                            if provider_config:
                                api_key = provider_config.api_key

                
                # Execute query
                file_context = st.session_state.get('file_content')
                file_info = st.session_state.get('file_info')
                
                # Execute async message processing
                message = _run_async(
                    chat_manager.process_message(
                        user_query=user_input,
                        mode=mode_map[mode],
                        file_context=file_context,
                        file_info=file_info,
                        selected_provider=selected_provider,
                        api_key=api_key
                    )
                )
                
                # Add assistant message to history
                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': f"Response from {mode}",
                    'mode': mode_map[mode],
                    'rlm_response': message.rlm_response,
                    'rlm_metrics': message.rlm_metrics,
                    'rlm_trace': message.rlm_trace,
                    'direct_response': message.direct_response,
                    'direct_metrics': message.direct_metrics,
                    'comparison_metrics': message.comparison_metrics,
                    'timestamp': st.session_state.get('current_time')
                })
                
                # Don't try to clear input - Streamlit doesn't allow modifying widget state
                # Just rerun and the form will reset naturally
                st.success("✅ Response received")
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
                st.exception(e)


# Main entry point
if __name__ == "__main__":
    render_chat_page()
