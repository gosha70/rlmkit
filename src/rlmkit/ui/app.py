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
    
    # Load from session state (set by Configuration page)
    from rlmkit.ui.services import LLMConfigManager
    from pathlib import Path
    
    config_dir = Path.home() / ".rlmkit"
    manager = LLMConfigManager(config_dir=config_dir)
    
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
    """Render run button and execute comparison."""
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
        
        # Create RLM configuration
        exec_config = ExecutionConfig(
            max_steps=config['max_steps'],
            default_timeout=config['timeout'],
            enable_rlm=True,
        )
        rlm_config = RLMConfig(execution=exec_config)
        
        # Get LLM client
        try:
            if config['provider'] == "Mock (Testing)":
                from rlmkit.llm.mock_client import MockLLMClient
                client = MockLLMClient([
                    "```python\nx = len(P)\nprint(f'Content length: {x}')\n```",
                    "FINAL: This is a mock response for testing purposes."
                ])
            else:
                client = get_llm_client(
                    provider=config['provider'].lower(),
                    model=config['model'],
                    api_key=config.get('api_key')
                )
        except Exception as e:
            st.error(f"Error creating LLM client: {str(e)}")
            return
        
        # Create RLM instance
        rlm = RLM(client=client, config=rlm_config)
        
        # Run based on selected mode
        with st.spinner(f"Running {config['mode']}..."):
            try:
                if config['mode'] == "Compare Both":
                    result = rlm.run_comparison(
                        prompt=st.session_state.file_content,
                        query=query
                    )
                    st.session_state.comparison_result = result
                elif config['mode'] == "RLM Only":
                    from rlmkit.core.comparison import ComparisonResult, ExecutionMetrics
                    from rlmkit.core.budget import TokenUsage
                    import time
                    
                    start = time.time()
                    rlm_result = rlm.run(st.session_state.file_content, query)
                    elapsed = time.time() - start
                    
                    # Create comparison result with only RLM metrics
                    result = ComparisonResult()
                    result.rlm_metrics = ExecutionMetrics(
                        mode="rlm",
                        answer=rlm_result.answer,
                        steps=rlm_result.steps,
                        tokens=TokenUsage(),  # Simplified for now
                        elapsed_time=elapsed,
                        success=rlm_result.success,
                        error=rlm_result.error,
                        trace=rlm_result.trace
                    )
                    st.session_state.comparison_result = result
                else:  # Direct Only
                    from rlmkit.core.comparison import ComparisonResult, ExecutionMetrics
                    from rlmkit.core.budget import TokenUsage
                    
                    direct_result = rlm.run_direct(st.session_state.file_content, query)
                    
                    result = ComparisonResult()
                    result.direct_metrics = ExecutionMetrics(
                        mode="direct",
                        answer=direct_result.answer,
                        steps=0,
                        tokens=TokenUsage(),
                        elapsed_time=0,
                        success=direct_result.success,
                        error=direct_result.error,
                        trace=direct_result.trace
                    )
                    st.session_state.comparison_result = result
                
                st.success("✓ Analysis complete!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error during execution: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


def render_results():
    """Render results section."""
    if st.session_state.comparison_result is None:
        return
    
    result = st.session_state.comparison_result
    
    st.divider()
    st.subheader("📊 Results")
    
    # Create tabs for different views
    tabs = st.tabs(["Answers", "Metrics", "Charts", "Trace", "Export"])
    
    # Tab 1: Answers
    with tabs[0]:
        render_answers_tab(result)
    
    # Tab 2: Metrics
    with tabs[1]:
        render_metrics_tab(result)
    
    # Tab 3: Charts
    with tabs[2]:
        render_charts_tab(result)
    
    # Tab 4: Trace
    with tabs[3]:
        render_trace_tab(result)
    
    # Tab 5: Export
    with tabs[4]:
        render_export_tab(result)


def render_answers_tab(result: ComparisonResult):
    """Render answers comparison."""
    if result.rlm_metrics:
        st.markdown("### 🤖 RLM Mode Answer")
        if result.rlm_metrics.success:
            st.info(result.rlm_metrics.answer)
            st.caption(f"Steps: {result.rlm_metrics.steps} | Time: {result.rlm_metrics.elapsed_time:.2f}s")
        else:
            st.error(f"Error: {result.rlm_metrics.error}")
    
    if result.direct_metrics:
        st.markdown("### 📝 Direct Mode Answer")
        if result.direct_metrics.success:
            st.info(result.direct_metrics.answer)
            st.caption(f"Time: {result.direct_metrics.elapsed_time:.2f}s")
        else:
            st.error(f"Error: {result.direct_metrics.error}")


def render_metrics_tab(result: ComparisonResult):
    """Render metrics comparison."""
    summary = result.get_summary()
    
    if summary.get('can_compare'):
        col1, col2, col3 = st.columns(3)
        
        # Token comparison
        token_savings = summary.get('token_savings')
        if token_savings:
            with col1:
                st.metric(
                    "Token Difference",
                    f"{token_savings['savings_tokens']:,}",
                    f"{token_savings['savings_percent']:.1f}%",
                    delta_color="normal" if token_savings['rlm_is_better'] else "inverse"
                )
        
        # Time comparison
        time_comp = summary.get('time_comparison')
        if time_comp:
            with col2:
                st.metric(
                    "Time Difference",
                    f"{abs(time_comp['difference']):.2f}s",
                    "Direct faster" if time_comp['direct_is_faster'] else "RLM faster"
                )
        
        # Recommendation
        with col3:
            if summary.get('recommendation'):
                rec = summary['recommendation']
                reason = summary.get('recommendation_reason', '')
                
                if rec == 'rlm':
                    st.success(f"✓ RLM Recommended\n\n{reason}")
                elif rec == 'direct':
                    st.info(f"→ Direct Recommended\n\n{reason}")
                else:
                    st.warning(f"≈ Similar Performance\n\n{reason}")
    
    # Detailed metrics
    st.markdown("### Detailed Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if result.rlm_metrics:
            st.markdown("#### RLM Mode")
            metrics_dict = result.rlm_metrics.to_dict()
            st.json(metrics_dict)
    
    with col2:
        if result.direct_metrics:
            st.markdown("#### Direct Mode")
            metrics_dict = result.direct_metrics.to_dict()
            st.json(metrics_dict)


def render_charts_tab(result: ComparisonResult):
    """Render visualization charts."""
    if not result.rlm_metrics and not result.direct_metrics:
        st.info("No data available for charts")
        return
    
    # Token comparison chart
    if result.rlm_metrics and result.direct_metrics:
        st.markdown("### Token Usage Comparison")
        token_chart = create_token_comparison_chart(result)
        if token_chart:
            st.plotly_chart(token_chart, use_container_width=True)
        
        # Time comparison chart
        st.markdown("### Execution Time Comparison")
        time_chart = create_time_comparison_chart(result)
        if time_chart:
            st.plotly_chart(time_chart, use_container_width=True)
        
        # Metrics radar chart
        st.markdown("### Performance Radar")
        radar_chart = create_metrics_radar_chart(result)
        if radar_chart:
            st.plotly_chart(radar_chart, use_container_width=True)


def render_trace_tab(result: ComparisonResult):
    """Render execution trace."""
    if result.rlm_metrics and result.rlm_metrics.trace:
        st.markdown("### 🔍 RLM Execution Trace")
        for item in result.rlm_metrics.trace:
            with st.expander(f"Step {item.get('step', 0)} - {item.get('role', 'unknown')}"):
                st.code(item.get('content', ''), language='text')
    
    if result.direct_metrics and result.direct_metrics.trace:
        st.markdown("### 📝 Direct Mode Trace")
        for item in result.direct_metrics.trace:
            st.code(item.get('content', ''), language='text')


def render_export_tab(result: ComparisonResult):
    """Render export options."""
    st.markdown("### 💾 Export Results")
    
    # JSON export
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### JSON Export")
        json_data = json.dumps(result.to_dict(), indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="rlmkit_results.json",
            mime="application/json"
        )
    
    with col2:
        st.markdown("#### Summary Report")
        summary = result.get_summary()
        summary_text = f"""# RLMKit Analysis Report

## File Information
- File: {st.session_state.file_info.filename if st.session_state.file_info else 'N/A'}
- Size: {st.session_state.file_info.size_bytes if st.session_state.file_info else 0} bytes
- Estimated Tokens: {st.session_state.file_info.estimated_tokens if st.session_state.file_info else 0}

## Query
{st.session_state.last_query}

## Results Summary
{json.dumps(summary, indent=2)}
"""
        st.download_button(
            label="Download Report",
            data=summary_text,
            file_name="rlmkit_report.md",
            mime="text/markdown"
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
