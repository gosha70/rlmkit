# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Streamlit web interface for RLMKit.

Interactive UI for:
- Uploading and processing large files
- Running RLM vs Direct mode comparisons
- Visualizing performance metrics
- Exporting results
"""

import streamlit as st
import json
import asyncio
from typing import Optional
from pathlib import Path

from rlmkit.core.rlm import RLM
from rlmkit.core.comparison import ComparisonResult
from rlmkit.config import RLMConfig, ExecutionConfig
from rlmkit.llm import get_llm_client
from rlmkit.ui.file_processor import process_file
from rlmkit.ui.charts import (
    create_token_comparison_chart,
    create_cost_comparison_chart,
    create_time_comparison_chart,
    create_metrics_radar_chart,
)

# Import Phase 2 service layer
from rlmkit.ui.services.chat_manager import ChatManager
from rlmkit.ui.services.memory_monitor import MemoryMonitor
from rlmkit.ui.services.models import ChatMessage


# Page configuration
st.set_page_config(
    page_title="RLMKit - Interactive Testing",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'file_content' not in st.session_state:
        st.session_state.file_content = None
    if 'file_info' not in st.session_state:
        st.session_state.file_info = None
    if 'comparison_result' not in st.session_state:
        st.session_state.comparison_result = None
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    
    # Initialize LLM manager first (needed by all pages)
    if 'llm_manager' not in st.session_state:
        from rlmkit.ui.services import LLMConfigManager
        from pathlib import Path
        config_dir = Path.home() / ".rlmkit"
        st.session_state.llm_manager = LLMConfigManager(config_dir=config_dir)
    
    # Phase 2 service layer initialization
    if 'chat_manager' not in st.session_state:
        st.session_state.chat_manager = ChatManager(st.session_state)
    if 'memory_monitor' not in st.session_state:
        st.session_state.memory_monitor = MemoryMonitor()
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    # Initialize provider API keys from persistent storage
    if 'provider_api_keys' not in st.session_state:
        from pathlib import Path
        import json
        
        st.session_state.provider_api_keys = {}
        
        # Try to load previously saved API keys
        keys_file = Path.home() / ".rlmkit" / "api_keys.json"
        if keys_file.exists():
            try:
                with open(keys_file, "r") as f:
                    st.session_state.provider_api_keys = json.load(f)
            except (json.JSONDecodeError, IOError):
                st.session_state.provider_api_keys = {}
    
    # Initialize selected_provider from .env or use first available
    if 'selected_provider' not in st.session_state:
        providers = st.session_state.llm_manager.list_providers()
        
        # Try to load previously selected provider from .env
        selected = None
        from pathlib import Path
        env_file = Path.home() / ".rlmkit" / ".env"
        if env_file.exists():
            with open(env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("SELECTED_PROVIDER="):
                        selected = line.split("=", 1)[1].strip()
                        break
        
        # Use saved provider if it exists, otherwise use first available
        if selected and selected in providers:
            st.session_state.selected_provider = selected
        else:
            st.session_state.selected_provider = providers[0] if providers else None


def render_header():
    """Render page header."""
    st.title("🔬 RLMKit Interactive Testing")
    st.markdown("""
    Test RLMKit's Recursive Language Model approach against traditional direct LLM queries.
    Upload large files, ask questions, and compare performance metrics.
    """)
    st.divider()


def render_sidebar():
    """Render simplified sidebar showing only selected provider and mode (read-only)."""
    st.sidebar.header("📊 Current Setup")
    
    # Use manager from session state (initialized by Configuration page)
    # This ensures we use the same instance with API keys in memory
    if 'llm_manager' not in st.session_state:
        from rlmkit.ui.services import LLMConfigManager
        from pathlib import Path
        config_dir = Path.home() / ".rlmkit"
        st.session_state.llm_manager = LLMConfigManager(config_dir=config_dir)
    
    manager = st.session_state.llm_manager
    
    # Get selected provider from session state
    if 'selected_provider' not in st.session_state:
        providers = manager.list_providers()
        st.session_state.selected_provider = providers[0] if providers else None
    
    if 'execution_mode' not in st.session_state:
        st.session_state.execution_mode = "Compare Both"
    
    if 'max_steps' not in st.session_state:
        st.session_state.max_steps = 16
    
    if 'timeout' not in st.session_state:
        st.session_state.timeout = 5
    
    # Display selected LLM (read-only)
    st.sidebar.subheader("🤖 Selected LLM")
    if st.session_state.selected_provider:
        config = manager.get_provider_config(st.session_state.selected_provider)
        if config:
            st.sidebar.write(f"**Provider:** {config.provider.upper()}")
            st.sidebar.write(f"**Model:** `{config.model}`")
            status = "✅ Ready" if config.is_ready else "⚠️ Not Ready"
            st.sidebar.write(f"**Status:** {status}")
        else:
            st.sidebar.warning("Provider config not found")
    else:
        st.sidebar.warning("No provider selected. Go to Configuration page to set up.")
    
    st.sidebar.divider()
    
    # Display execution mode (read-only)
    st.sidebar.subheader("⚡ Execution Mode")
    st.sidebar.write(f"**Mode:** {st.session_state.execution_mode}")
    st.sidebar.write(f"**Max Steps:** {st.session_state.max_steps}")
    st.sidebar.write(f"**Timeout:** {st.session_state.timeout}s")
    
    st.sidebar.info("⚙️ To change settings, go to the Configuration page")
    
    st.sidebar.divider()
    
    # Return configuration
    provider = st.session_state.selected_provider
    if provider:
        config = manager.get_provider_config(provider)
        model = config.model if config else None
        api_key = config.api_key if config else None
    else:
        config = None
        model = None
        api_key = None
    
    return {
        'mode': st.session_state.execution_mode,
        'max_steps': st.session_state.max_steps,
        'timeout': st.session_state.timeout,
        'provider': provider,
        'model': model,
        'api_key': api_key,
    }


def render_content_input():
    """Render content input section - file upload or direct text."""
    st.subheader("📄 Content")
    
    # Tabs for different input methods
    input_tab1, input_tab2, input_tab3 = st.tabs(["✍️ Enter Text", "📁 Upload File", "📚 Sample Content"])
    
    # Tab 1: Direct text input
    with input_tab1:
        direct_text = st.text_area(
            "Enter your content directly:",
            height=200,
            placeholder="Paste your document, article, or any text here...",
            help="Enter the content you want to analyze",
            key="direct_text_input"
        )
        
        if st.button("📝 Use This Text", key="use_direct_text"):
            if direct_text.strip():
                from rlmkit.ui.file_processor import FileInfo, FileProcessor
                st.session_state.file_content = direct_text
                st.session_state.file_info = FileInfo(
                    filename="direct_input.txt",
                    content=direct_text,
                    file_type=".txt",
                    size_bytes=len(direct_text.encode()),
                    char_count=len(direct_text),
                    estimated_tokens=FileProcessor.estimate_tokens(direct_text),
                    success=True
                )
                st.success(f"✓ Content loaded ({len(direct_text)} characters)")
                st.rerun()
            else:
                st.warning("Please enter some text first")
    
    # Tab 2: File upload
    with input_tab2:
        uploaded_file = st.file_uploader(
            "Upload a file (PDF, DOCX, TXT, MD, JSON, code files)",
            type=['pdf', 'docx', 'txt', 'md', 'json', 'py', 'js', 'ts', 'java', 'cpp', 'c', 'h'],
            help="Upload a large file to test RLMKit's ability to handle large prompts"
        )
        
        if uploaded_file is not None:
            # Process file
            file_bytes = uploaded_file.read()
            file_info = process_file(file_bytes=file_bytes, filename=uploaded_file.name)
            
            if file_info.success:
                st.session_state.file_content = file_info.content
                st.session_state.file_info = file_info
                
                st.success(f"✓ Loaded {uploaded_file.name}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("File Size", f"{file_info.size_bytes / 1024:.1f} KB")
                with col2:
                    st.metric("Est. Tokens", f"{file_info.estimated_tokens:,}")
            else:
                st.error(f"Error: {file_info.error}")
                st.session_state.file_content = None
                st.session_state.file_info = None
    
    # Tab 3: Sample content
    with input_tab3:
        sample_choice = st.radio(
            "Select sample:",
            ["Short Text (Demo)", "Medium Article", "Long Documentation"],
            key="sample_choice"
        )
        
        if st.button("📚 Load Sample", key="load_sample"):
            if sample_choice == "Short Text (Demo)":
                content = "This is a short sample text for testing RLMKit.\n" * 10
            elif sample_choice == "Medium Article":
                content = "Article content here.\n" * 100
            else:
                content = "Long documentation content.\n" * 500
            
            from rlmkit.ui.file_processor import FileInfo, FileProcessor
            st.session_state.file_content = content
            st.session_state.file_info = FileInfo(
                filename=f"{sample_choice}.txt",
                content=content,
                file_type=".txt",
                size_bytes=len(content.encode()),
                char_count=len(content),
                estimated_tokens=FileProcessor.estimate_tokens(content),
                success=True
            )
            st.success("✓ Sample content loaded")
            st.rerun()
    
    # Show current content status
    if st.session_state.file_content:
        st.info(f"✅ Content loaded: {st.session_state.file_info.filename if st.session_state.file_info else 'text'} "
                f"({len(st.session_state.file_content)} chars, "
                f"~{st.session_state.file_info.estimated_tokens if st.session_state.file_info else 'N/A'} tokens)")


def render_query_input():
    """Render query input section."""
    st.subheader("❓ Query")
    
    query = st.text_area(
        "Enter your question about the uploaded content:",
        value=st.session_state.last_query,
        height=100,
        placeholder="E.g., Summarize the main points of this document",
        help="Ask a question about the uploaded file"
    )
    
    return query


def render_run_button(config, query):
    """Render run button and execute comparison using Phase 2 service layer."""
    # Check if content and query are provided
    has_content = st.session_state.file_content is not None
    has_query = query and query.strip()
    
    # Show status
    col1, col2 = st.columns(2)
    with col1:
        if has_content:
            st.success("✓ Content ready")
        else:
            st.warning("⚠️ No content loaded")
    
    with col2:
        if has_query:
            st.success("✓ Query entered")
        else:
            st.warning("⚠️ No query entered")
    
    # Disable button if not ready
    can_run = has_content and has_query
    
    if st.button(
        "🚀 Run Analysis" if can_run else "⏸️ Enter Content & Query First",
        type="primary" if can_run else "secondary",
        use_container_width=True,
        disabled=not can_run,
        key="run_analysis_btn"
    ):
        st.session_state.last_query = query
        
        # Map UI mode to Phase 2 mode
        mode_mapping = {
            "Compare Both": "compare",
            "RLM Only": "rlm_only",
            "Direct Only": "direct_only"
        }
        phase2_mode = mode_mapping.get(config['mode'], "compare")
        
        # Use Phase 2 service layer via ChatManager
        with st.spinner(f"Running {config['mode']}..."):
            try:
                # Prepare file info for ChatManager
                file_info = None
                if st.session_state.file_info:
                    file_info = {
                        "filename": st.session_state.file_info.filename,
                        "size_bytes": st.session_state.file_info.size_bytes,
                        "estimated_tokens": st.session_state.file_info.estimated_tokens,
                    }
                
                # Call Phase 2 ChatManager.process_message()
                chat_message = asyncio.run(
                    st.session_state.chat_manager.process_message(
                        user_query=query,
                        mode=phase2_mode,
                        file_context=st.session_state.file_content,
                        file_info=file_info
                    )
                )
                
                # Store the chat message
                st.session_state.chat_messages.append(chat_message)
                
                # Also store in legacy comparison_result for backwards compatibility
                st.session_state.current_chat_message = chat_message
                
                st.success("✓ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during execution: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_results():
    """Render results section using Phase 2 ChatMessage with improved chat-like interface."""
    if 'current_chat_message' not in st.session_state:
        return
    
    message = st.session_state.current_chat_message
    
    st.divider()
    st.subheader("📊 Results")
    
    # Chat-like display with better UX
    with st.container():
        # Display user query
        st.markdown(f"**📌 Your Query:** {message.user_query}")
        st.divider()
        
        # RLM Response if available
        if message.rlm_response:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("#### 🤖 RLM Response (Multi-step Analysis)")
                    st.info(message.rlm_response.content)
                with col2:
                    if message.rlm_metrics:
                        st.metric("Steps", message.rlm_metrics.steps_taken, "exploration steps")
                        st.metric("Time", f"{message.rlm_metrics.execution_time_seconds:.2f}s")
                        st.metric("Cost", f"${message.rlm_metrics.cost_usd:.4f}")
        
        # Direct Response if available  
        if message.direct_response:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("#### 📝 Direct Response (Single Call)")
                    st.info(message.direct_response.content)
                with col2:
                    if message.direct_metrics:
                        st.metric("Type", "Direct")
                        st.metric("Time", f"{message.direct_metrics.execution_time_seconds:.2f}s")
                        st.metric("Cost", f"${message.direct_metrics.cost_usd:.4f}")
        
        # Comparison if available
        if message.comparison_metrics and message.rlm_response and message.direct_response:
            st.divider()
            with st.container():
                st.markdown("#### 💡 AI Recommendation")
                st.success(message.comparison_metrics.recommendation)
        
        # Error if present
        if message.error:
            st.divider()
            st.error(f"⚠️ Error: {message.error}")
        
        # More details in expander
        st.divider()
        with st.expander("📋 Detailed Metrics & Export Options"):
            tab_names = ["Metrics Breakdown", "Detailed Comparison", "Export Results"]
            tabs = st.tabs(tab_names)
            
            with tabs[0]:
                render_metrics_comparison_tab(message)
            with tabs[1]:
                render_comparison_analysis_tab(message)
            with tabs[2]:
                render_export_phase2_tab(message)


def render_responses_tab(message: ChatMessage):
    """Render side-by-side responses from Phase 2."""
    has_rlm = message.rlm_response is not None
    has_direct = message.direct_response is not None
    
    if has_rlm and has_direct:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🤖 RLM Response")
            st.info(message.rlm_response.content if message.rlm_response else "N/A")
            if message.rlm_metrics:
                st.caption(f"Steps: {message.rlm_metrics.steps_taken} | Time: {message.rlm_metrics.execution_time_seconds:.2f}s")
        
        with col2:
            st.markdown("### 📝 Direct Response")
            st.info(message.direct_response.content if message.direct_response else "N/A")
            if message.direct_metrics:
                st.caption(f"Direct call (0 steps) | Time: {message.direct_metrics.execution_time_seconds:.2f}s")
    elif has_rlm:
        st.markdown("### 🤖 RLM Response")
        st.info(message.rlm_response.content if message.rlm_response else "N/A")
        if message.rlm_metrics:
            st.caption(f"Steps: {message.rlm_metrics.steps_taken} | Time: {message.rlm_metrics.execution_time_seconds:.2f}s")
    elif has_direct:
        st.markdown("### 📝 Direct Response")
        st.info(message.direct_response.content if message.direct_response else "N/A")
        if message.direct_metrics:
            st.caption(f"Direct call (0 steps) | Time: {message.direct_metrics.execution_time_seconds:.2f}s")


def render_metrics_comparison_tab(message: ChatMessage):
    """Render metrics comparison from Phase 2."""
    if message.rlm_metrics and message.direct_metrics:
        # Side-by-side metrics comparison
        col1, col2, col3 = st.columns(3)
        
        # Cost comparison
        with col1:
            st.metric(
                "💰 Total Cost",
                f"${message.rlm_metrics.cost_usd:.6f} vs ${message.direct_metrics.cost_usd:.6f}",
                f"RLM +${message.rlm_metrics.cost_usd - message.direct_metrics.cost_usd:.6f}"
            )
        
        # Token comparison
        with col2:
            st.metric(
                "🔤 Total Tokens",
                f"{message.rlm_metrics.total_tokens} vs {message.direct_metrics.total_tokens}",
                f"+{message.rlm_metrics.total_tokens - message.direct_metrics.total_tokens} (RLM)"
            )
        
        # Time comparison
        with col3:
            st.metric(
                "⏱️ Execution Time",
                f"{message.rlm_metrics.execution_time_seconds:.2f}s vs {message.direct_metrics.execution_time_seconds:.2f}s",
                f"+{message.rlm_metrics.execution_time_seconds - message.direct_metrics.execution_time_seconds:.2f}s (RLM)"
            )
        
        # Detailed breakdown
        st.markdown("### Detailed Breakdown")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### RLM Metrics")
            rlm_dict = {
                "Input Tokens": message.rlm_metrics.input_tokens,
                "Output Tokens": message.rlm_metrics.output_tokens,
                "Total Tokens": message.rlm_metrics.total_tokens,
                "Cost (USD)": f"${message.rlm_metrics.cost_usd:.6f}",
                "Cost Breakdown": message.rlm_metrics.cost_breakdown,
                "Execution Time (s)": f"{message.rlm_metrics.execution_time_seconds:.2f}",
                "Steps": message.rlm_metrics.steps_taken,
                "Memory (MB)": f"{message.rlm_metrics.memory_used_mb:.1f} (peak: {message.rlm_metrics.memory_peak_mb:.1f})",
            }
            st.json(rlm_dict)
        
        with col2:
            st.markdown("#### Direct Metrics")
            direct_dict = {
                "Input Tokens": message.direct_metrics.input_tokens,
                "Output Tokens": message.direct_metrics.output_tokens,
                "Total Tokens": message.direct_metrics.total_tokens,
                "Cost (USD)": f"${message.direct_metrics.cost_usd:.6f}",
                "Cost Breakdown": message.direct_metrics.cost_breakdown,
                "Execution Time (s)": f"{message.direct_metrics.execution_time_seconds:.2f}",
                "Steps": message.direct_metrics.steps_taken,
                "Memory (MB)": f"{message.direct_metrics.memory_used_mb:.1f} (peak: {message.direct_metrics.memory_peak_mb:.1f})",
            }
            st.json(direct_dict)
    
    elif message.rlm_metrics:
        st.markdown("### RLM Metrics")
        rlm_dict = {
            "Input Tokens": message.rlm_metrics.input_tokens,
            "Output Tokens": message.rlm_metrics.output_tokens,
            "Total Tokens": message.rlm_metrics.total_tokens,
            "Cost (USD)": f"${message.rlm_metrics.cost_usd:.6f}",
            "Execution Time (s)": f"{message.rlm_metrics.execution_time_seconds:.2f}",
            "Steps": message.rlm_metrics.steps_taken,
        }
        st.json(rlm_dict)
    
    elif message.direct_metrics:
        st.markdown("### Direct Metrics")
        direct_dict = {
            "Input Tokens": message.direct_metrics.input_tokens,
            "Output Tokens": message.direct_metrics.output_tokens,
            "Total Tokens": message.direct_metrics.total_tokens,
            "Cost (USD)": f"${message.direct_metrics.cost_usd:.6f}",
            "Execution Time (s)": f"{message.direct_metrics.execution_time_seconds:.2f}",
            "Steps": message.direct_metrics.steps_taken,
        }
        st.json(direct_dict)


def render_comparison_analysis_tab(message: ChatMessage):
    """Render comparison analysis from Phase 2."""
    if message.comparison_metrics:
        comp = message.comparison_metrics
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Cost Delta",
                f"${comp.cost_delta_usd:.6f}",
                f"{comp.cost_delta_percent:.0f}%"
            )
        
        with col2:
            st.metric(
                "Token Delta",
                f"{comp.token_delta:+d}",
                "RLM more"
            )
        
        with col3:
            st.metric(
                "Time Delta",
                f"{comp.time_delta_seconds:.2f}s",
                f"{comp.time_delta_percent:+.0f}%"
            )
        
        with col4:
            st.metric(
                "RLM Steps",
                f"{comp.rlm_steps}",
                f"vs {comp.direct_steps} direct"
            )
        
        # Recommendation
        st.markdown("### 💡 Analysis & Recommendation")
        st.info(comp.recommendation)
        
        # Trade-off visualization
        st.markdown("### Trade-off Analysis")
        st.markdown(f"""
        **Cost**: RLM is ${abs(comp.cost_delta_usd):.6f} **{'more expensive' if comp.cost_delta_usd > 0 else 'cheaper'}** ({abs(comp.cost_delta_percent):.0f}%)
        
        **Speed**: RLM is {abs(comp.time_delta_seconds):.2f}s **{'slower' if comp.time_delta_seconds > 0 else 'faster'}** ({abs(comp.time_delta_percent):.0f}%)
        
        **Tokens**: RLM uses {comp.token_delta} **more tokens** for {comp.rlm_steps}-step exploration vs {comp.direct_steps}-step direct call
        
        **Insight**: {comp.recommendation}
        """)
    else:
        st.info("Run comparison mode to see detailed analysis")


def render_export_phase2_tab(message: ChatMessage):
    """Render export options for Phase 2 ChatMessage."""
    st.markdown("### 💾 Export Results")
    
    # Get ChatManager to use export_conversation method
    manager = st.session_state.chat_manager
    
    col1, col2, col3 = st.columns(3)
    
    # JSON export
    with col1:
        st.markdown("#### JSON Export")
        json_export = manager.export_conversation("json")
        st.download_button(
            label="📥 Download JSON",
            data=json_export,
            file_name="rlmkit_analysis.json",
            mime="application/json"
        )
    
    # Markdown export
    with col2:
        st.markdown("#### Markdown Export")
        md_export = manager.export_conversation("markdown")
        st.download_button(
            label="📥 Download Markdown",
            data=md_export,
            file_name="rlmkit_analysis.md",
            mime="text/markdown"
        )
    
    # CSV export
    with col3:
        st.markdown("#### CSV Export")
        csv_export = manager.export_conversation("csv")
        st.download_button(
            label="📥 Download CSV",
            data=csv_export,
            file_name="rlmkit_metrics.csv",
            mime="text/csv"
        )


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Main content
    render_content_input()
    query = render_query_input()
    render_run_button(config, query)
    render_results()
    
    # Footer
    st.divider()
    st.caption("RLMKit - Recursive Language Model Toolkit | Week #6: Polish & Release")


if __name__ == "__main__":
    main()
