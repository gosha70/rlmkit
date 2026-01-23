# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
Chat Message Component - Reusable message rendering
"""

import streamlit as st
from typing import Optional, Dict, Any


def render_chat_message(
    content: str,
    role: str = "assistant",
    metrics: Optional[Dict[str, Any]] = None,
    trace: Optional[list] = None,
    comparison: Optional[Dict[str, Any]] = None,
    show_actions: bool = True,
    key_suffix: str = ""
) -> None:
    """
    Render a chat message with optional metrics, trace, and actions.
    
    Args:
        content: Message content
        role: "user" or "assistant"
        metrics: ExecutionMetrics dict
        trace: List of trace steps
        comparison: ComparisonMetrics dict
        show_actions: Whether to show copy/regenerate/export buttons
        key_suffix: Suffix for widget keys to avoid collisions
    """
    
    # Render message container
    with st.chat_message(role, avatar=_get_avatar(role)):
        st.markdown(content)
    
    # Render metrics if provided
    if metrics:
        render_metrics(metrics, key_suffix)
    
    # Render trace if provided
    if trace:
        render_trace(trace, key_suffix)
    
    # Render comparison if provided
    if comparison:
        render_comparison(comparison, key_suffix)
    
    # Render action buttons if requested
    if show_actions:
        render_message_actions(key_suffix)


def _get_avatar(role: str) -> str:
    """Get avatar for message role."""
    avatars = {
        "user": "👤",
        "assistant": "🤖",
        "rlm": "⚙️",
        "direct": "📋",
        "system": "⚙️",
    }
    return avatars.get(role, "💬")


def render_metrics(metrics: Dict[str, Any], key_suffix: str = "") -> None:
    """Render execution metrics in a compact format."""
    
    st.markdown("---")
    st.subheader("📊 Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        steps = metrics.get('steps_taken', 0)
        st.metric("Steps", steps)
    
    with col2:
        tokens = metrics.get('total_tokens', 0)
        st.metric("Tokens", f"{tokens:,}")
    
    with col3:
        time = metrics.get('execution_time_seconds', 0)
        st.metric("Time", f"{time:.2f}s")
    
    with col4:
        cost = metrics.get('cost_usd', 0)
        st.metric("Cost", f"${cost:.4f}")
    
    # Show cost breakdown if available
    if metrics.get('cost_breakdown'):
        breakdown = metrics['cost_breakdown']
        with st.expander("💰 Cost Breakdown"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Input Cost", f"${breakdown.get('input', 0):.4f}")
            with col2:
                st.metric("Output Cost", f"${breakdown.get('output', 0):.4f}")


def render_trace(trace: list, key_suffix: str = "") -> None:
    """Render execution trace as timeline."""
    
    if not trace:
        return
    
    st.markdown("---")
    
    with st.expander(f"📈 Execution Trace ({len(trace)} steps)", expanded=False):
        for i, step in enumerate(trace, 1):
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.write(f"**Step {i}**")
            
            with col2:
                description = step.get('description', step.get('type', 'Unknown'))
                st.write(f"**{description}**")
                
                # Show step metrics
                metrics_text = []
                if step.get('input_tokens') or step.get('output_tokens'):
                    metrics_text.append(
                        f"Tokens: {step.get('input_tokens', 0)} → {step.get('output_tokens', 0)}"
                    )
                if step.get('duration_ms'):
                    metrics_text.append(f"Time: {step.get('duration_ms')}ms")
                if step.get('cost'):
                    metrics_text.append(f"Cost: ${step.get('cost'):.4f}")
                
                if metrics_text:
                    st.caption(" | ".join(metrics_text))
                
                # Show step content if available
                if step.get('content'):
                    with st.expander(f"View step output", key=f"trace_{i}{key_suffix}"):
                        st.code(step['content'], language="text")


def render_comparison(comparison: Dict[str, Any], key_suffix: str = "") -> None:
    """Render side-by-side comparison between RLM and Direct."""
    
    st.markdown("---")
    st.subheader("📊 RLM vs Direct Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**⚙️ RLM (Multi-step)**")
        st.metric("Tokens", f"{comparison.get('rlm_tokens', 0):,}")
        st.metric("Cost", f"${comparison.get('rlm_cost_usd', 0):.4f}")
        st.metric("Time", f"{comparison.get('rlm_time_seconds', 0):.2f}s")
        st.metric("Steps", comparison.get('rlm_steps', 0))
    
    with col2:
        st.write("**📋 Direct (Single)**")
        st.metric("Tokens", f"{comparison.get('direct_tokens', 0):,}")
        st.metric("Cost", f"${comparison.get('direct_cost_usd', 0):.4f}")
        st.metric("Time", f"{comparison.get('direct_time_seconds', 0):.2f}s")
        st.metric("Steps", comparison.get('direct_steps', 0))
    
    with col3:
        st.write("**📈 Deltas**")
        
        token_delta = comparison.get('token_delta', 0)
        token_pct = comparison.get('token_delta_percent', 0)
        st.metric("Tokens", f"{token_delta:+,}", delta=f"{token_pct:+.1f}%")
        
        cost_delta = comparison.get('cost_delta_usd', 0)
        cost_pct = comparison.get('cost_delta_percent', 0)
        st.metric("Cost", f"${cost_delta:+.4f}", delta=f"{cost_pct:+.1f}%")
        
        time_delta = comparison.get('time_delta_seconds', 0)
        time_pct = comparison.get('time_delta_percent', 0)
        st.metric("Time", f"{time_delta:+.2f}s", delta=f"{time_pct:+.1f}%")
        
        quality_delta = comparison.get('quality_delta', 0)
        st.metric("Quality", f"{quality_delta:+.2f}", delta="points")
    
    # Show recommendation
    if comparison.get('recommendation'):
        st.success(f"💡 **Recommendation:** {comparison['recommendation']}")


def render_message_actions(key_suffix: str = "") -> None:
    """Render message action buttons (copy, regenerate, export)."""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy", key=f"copy_{key_suffix}"):
            st.success("Copied to clipboard!")
    
    with col2:
        if st.button("🔄 Regenerate", key=f"regen_{key_suffix}"):
            st.info("Regeneration feature coming soon")
    
    with col3:
        if st.button("⬇️ Export", key=f"export_{key_suffix}"):
            st.info("Export feature coming soon")
