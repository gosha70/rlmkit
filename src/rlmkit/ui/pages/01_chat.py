# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
Chat Page - Main interactive chat interface
Page 1 of RLM Studio: ChatGPT-like interface for exploring documents
"""

import streamlit as st
import asyncio

from rlmkit.ui.services.chat_manager import ChatManager


# -----------------------------------------------------------------------------
# RLM Chat Composer UI skin (Streamlit)
# -----------------------------------------------------------------------------

def _inject_rlmkit_desktop_css():
    """Global CSS RLM Chat Composer styling."""
    st.markdown(
        """
        <style>
        /* Page + padding */
        .stApp {
            background: #0b0b10;
            color: #e8e8ea;
        }

        /* Remove Streamlit's top chrome */
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        #MainMenu,
        footer {
            visibility: hidden;
            height: 0;
        }

        /* Tighten vertical rhythm */
        .block-container {
            padding-top: 1.0rem;
            padding-bottom: 4.5rem; /* room for bottom composer */
            max-width: 980px;
        }

        /* Chat bubbles */
        [data-testid="stChatMessage"] {
            border-radius: 18px;
        }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li {
            line-height: 1.5;
            font-size: 0.98rem;
        }

        /* Hide default "Message" labels etc */
        .stTextArea label, .stTextInput label {
            display: none !important;
        }

        button[data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
            display: none !important;
        }   

        /* Composer shell */
        .rlmkit-composer {
            border: 1px solid #2b2b33;
            background: #141419;
            border-radius: 18px;
            overflow: hidden;
        }
        .rlmkit-composer:focus-within {
            border-color: #3b3b46;
            box-shadow: 0 0 0 2px rgba(255,255,255,0.03);
        }

        /* Textarea styling */
        .rlmkit-composer textarea {
            background: transparent !important;
            border: none !important;
            padding: 14px 14px 10px 14px !important;
            color: #f2f2f4 !important;
            resize: none !important;
        }
        .rlmkit-composer textarea:focus {
            outline: none !important;
            box-shadow: none !important;
        }

        /* Bottom bar */
        .rlmkit-bottom-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            border-top: 1px solid #22222a;
            background: #111116;
        }

        /* Icon buttons */
        .rlmkit-icon button {
            background: transparent !important;
            border: 1px solid transparent !important;
            border-radius: 10px !important;
            padding: 0 !important;
            height: 36px !important;
            width: 36px !important;
            min-width: 36px !important;
            color: #c6c6cc !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 18px !important;
            line-height: 1 !important;
            text-align: center !important;
        }
        .rlmkit-icon button:hover {
            background: rgba(255,255,255,0.06) !important;
            border-color: rgba(255,255,255,0.06) !important;
            color: #ffffff !important;
        }
        
        /* Center all nested content inside icon buttons - target all levels */
        .rlmkit-icon button *,
        .rlmkit-icon button > div,
        .rlmkit-icon button span,
        .rlmkit-icon button [data-testid="stMarkdownContainer"],
        .rlmkit-icon button p {
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: 100% !important;
            height: 100% !important;
            margin: 0 !important;
            padding: 0 !important;
            text-align: center !important;
            line-height: 1 !important;
        }
        
        /* Hide any labels on popover buttons */
        .rlmkit-icon [data-testid="stPopover"] label {
            display: none !important;
        }
        .rlmkit-icon [data-testid="stPopover"] > div > div > span {
            display: none !important;
        }

        /* "Send" button: circular w/ accent */
        .rlmkit-send button {
            background: #b55a3c !important;
            border: none !important;
            border-radius: 12px !important;
            height: 36px !important;
            width: 44px !important;
            padding: 0 !important;
            color: #101014 !important;
            font-weight: 700 !important;
        }
        .rlmkit-send button:hover {
            filter: brightness(1.05);
        }

        /* Pills (mode chips / model selector button) */
        .rlmkit-pill button {
            background: rgba(255,255,255,0.04) !important;
            border: 1px solid rgba(255,255,255,0.06) !important;
            border-radius: 999px !important;
            padding: 6px 10px !important;
            height: 32px !important;
            color: #dedee3 !important;
            font-size: 0.85rem !important;
        }
        .rlmkit-pill button:hover {
            background: rgba(255,255,255,0.07) !important;
        }

        /* Popover content spacing */
        div[data-testid="stPopoverBody"] {
            padding: 10px 12px !important;
        }

        /* Subtle divider */
        hr {
            border-color: rgba(255,255,255,0.08) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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

    # Page config (safe if called multiple times)
    try:
        st.set_page_config(
            page_title="RLM Studio",
            page_icon="💬",
            layout="centered",
            initial_sidebar_state="collapsed",
        )
    except Exception:
        pass

    _inject_rlmkit_desktop_css()

    # Optional sidebar (collapsed by default). Keeps config discoverable without
    # dominating the UI like a traditional Streamlit app.
    render_sidebar()

    st.markdown(
        """<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;'>
        <div style='font-size:1.05rem;font-weight:650;'>RLM Studio</div>
        <div style='opacity:0.7;font-size:0.9rem;'>rlmkit-style composer • compare RLM vs direct</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Initialize session state
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = None

    # Show attached content (compact)
    render_document_context()

    # Render message history
    render_message_history()

    # Bottom composer
    st.divider()
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
    """Show currently loaded document/context (compact version)."""
    if st.session_state.get('file_info'):
        file_info = st.session_state.file_info
        col1, col2 = st.columns([4, 1])

        with col1:
            st.success(
                f"📎 **Attached:** {file_info.filename} | "
                f"{file_info.size_bytes / 1024:.1f} KB | "
                f"~{file_info.estimated_tokens:,} tokens"
            )

        with col2:
            if st.button("✕ Remove", key="clear_context", use_container_width=True):
                st.session_state.file_content = None
                st.session_state.file_info = None
                st.rerun()


def render_message_history():
    """Display conversation history."""
    messages = st.session_state.get('chat_messages', [])

    # Only show info message if no messages AND user hasn't typed anything
    if not messages and not st.session_state.get('chat_input', '').strip():
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
    """RLM Chat Composer Input Area.

    Goals vs the stock Streamlit UI:
    - Single rounded composer (textarea + bottom row)
    - Left "plus" opens a popover with attachment + toggles
    - Right side: model pill + send button
    """

    # ----------------------------
    # State
    # ----------------------------
    if "composer_mode" not in st.session_state:
        st.session_state.composer_mode = "compare"  # compare | rlm_only | direct_only
    if "research_enabled" not in st.session_state:
        st.session_state.research_enabled = False
    if "web_search_enabled" not in st.session_state:
        st.session_state.web_search_enabled = True
    if "style_preset" not in st.session_state:
        st.session_state.style_preset = "Compare"

    mode_map = {
        "Compare": "compare",
        "RLM": "rlm_only",
        "LLM": "direct_only",
    }

    # Provider/model text for the model pill
    provider_label = "Model"
    try:
        from rlmkit.ui.services import LLMConfigManager
        from pathlib import Path
        manager = LLMConfigManager(config_dir=Path.home() / ".rlmkit")
        selected_provider = st.session_state.get("selected_provider")
        if selected_provider:
            cfg = manager.get_provider_config(selected_provider)
            provider_label = f"{selected_provider.upper()} • {cfg.model}"
    except Exception:
        pass

    # ----------------------------
    # Composer shell
    # ----------------------------
    st.markdown('<div class="rlmkit-composer">', unsafe_allow_html=True)

    with st.container():
        user_input = st.text_area(
            "Message",
            placeholder="Ask anything about the document...",
            height=110,
            key="chat_input",
            label_visibility="collapsed",
        )

        st.markdown('<div class="rlmkit-bottom-bar">', unsafe_allow_html=True)

        # Left side: + popover and chips
        left, right = st.columns([7, 3], vertical_alignment="center")

        with left:
            c1, c2, c3, c4 = st.columns([1, 1, 1, 10], vertical_alignment="center")

            with c1:
                st.markdown('<div class="rlmkit-icon">', unsafe_allow_html=True)
                with st.popover('<p style="text-align:center;">➕</p>', use_container_width=True):
                    st.subheader("Add")
                    uploaded_file = st.file_uploader(
                        "Add files or photos",
                        type=[
                            "pdf",
                            "docx",
                            "txt",
                            "md",
                            "json",
                            "py",
                            "js",
                            "ts",
                            "java",
                            "cpp",
                            "c",
                            "h",
                            "png",
                            "jpg",
                            "jpeg",
                        ],
                        key="composer_file_upload",
                    )
                    if uploaded_file is not None:
                        from rlmkit.ui.file_processor import process_file

                        file_bytes = uploaded_file.read()
                        file_info = process_file(file_bytes=file_bytes, filename=uploaded_file.name)
                        if file_info.success:
                            st.session_state.file_content = file_info.content
                            st.session_state.file_info = file_info
                            st.success(f"Attached: {uploaded_file.name}")
                        else:
                            st.error(f"Error: {file_info.error}")

                    st.divider()
                    st.toggle("Research", key="research_enabled")
                    st.toggle("Web search", key="web_search_enabled")
                    st.selectbox(
                        "Use style",
                        ["Normal", "Learning", "Concise", "Professional", "Friendly", "Formal"],
                        key="style_preset",
                    )
                    st.caption(
                        "Styles allow you to customize how LLM communicates,\n"
                        "helping you achieve more while working in a way that feels natural to you.\n"                    )
                st.markdown("</div>", unsafe_allow_html=True)

        with right:
            m1, m2 = st.columns([3, 1], vertical_alignment="center")
            with m1:
                st.markdown('<div class="rlmkit-pill">', unsafe_allow_html=True)
                with st.popover(provider_label, use_container_width=True):
                    st.subheader("Execution")
                    st.radio(
                        "Mode",
                        ["Compare", "RLM only", "Direct only"],
                        index={
                            "compare": 0,
                            "rlm_only": 1,
                            "direct_only": 2,
                        }.get(st.session_state.composer_mode, 0),
                        key="composer_mode_radio",
                        label_visibility="collapsed",
                    )
                    st.caption("These map to the RLM Studio backend execution modes.")

                st.markdown("</div>", unsafe_allow_html=True)

            with m2:
                st.markdown('<div class="rlmkit-send">', unsafe_allow_html=True)
                send_button = st.button("↑", use_container_width=True, key="rlmkit_send")
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # rlmkit-bottom-bar

    st.markdown("</div>", unsafe_allow_html=True)  # rlmkit-composer

    # Keep composer_mode in sync
    mode_choice = st.session_state.get("composer_mode_radio")
    if mode_choice:
        st.session_state.composer_mode = {
            "Compare": "compare",
            "RLM only": "rlm_only",
            "Direct only": "direct_only",
        }[mode_choice]

    # Map style preset into execution mode (rlmkit-like chips)
    st.session_state.composer_mode = mode_map.get(
        st.session_state.get("style_preset", "Compare"),
        st.session_state.composer_mode,
    )

    # Handle message submission
    if send_button and user_input and user_input.strip():
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
                        mode=st.session_state.composer_mode,
                        file_context=file_context,
                        file_info=file_info,
                        selected_provider=selected_provider,
                        api_key=api_key
                    )
                )
                
                # Add assistant message to history
                st.session_state.chat_messages.append({
                    'role': 'assistant',
                    'content': f"Response ({st.session_state.composer_mode})",
                    'mode': st.session_state.composer_mode,
                    'rlm_response': message.rlm_response,
                    'rlm_metrics': message.rlm_metrics,
                    'rlm_trace': message.rlm_trace,
                    'direct_response': message.direct_response,
                    'direct_metrics': message.direct_metrics,
                    'comparison_metrics': message.comparison_metrics,
                    'timestamp': st.session_state.get('current_time')
                })

                # Don't modify chat_input - st.rerun() will reset it naturally
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
                st.exception(e)


# Main entry point
if __name__ == "__main__":
    render_chat_page()
