# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
Analysis Page - Metrics dashboard comparing RLM vs Direct LLM
Page 2 of RLM Studio: Performance analysis and recommendations
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


def render_analysis_page():
    """Render the analysis dashboard page."""
    
    st.title("📊 Analysis Dashboard")
    st.markdown("Compare RLM vs Direct LLM performance metrics")
    
    # Check if we have any chat history - use 'chat_messages' key where message dicts are stored
    conversations = st.session_state.get('chat_messages', [])
    if not conversations:
        st.info("📝 No conversations yet. Visit **Chat** page to start analyzing documents.")
        return
    
    # Render page sections
    render_summary_cards()
    st.divider()
    
    render_metrics_comparison()
    st.divider()
    
    render_cost_analysis()
    st.divider()
    
    render_quality_metrics()


def render_summary_cards():
    """Render summary metric cards at top of page."""
    
    latest_message = get_latest_message()
    if not latest_message:
        st.warning("No metrics available yet")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get metrics - ChatMessage is a dataclass, access attributes directly
    rlm_metrics = latest_message.rlm_metrics
    direct_metrics = latest_message.direct_metrics
    
    # Tokens metric
    rlm_tokens = rlm_metrics.total_tokens if rlm_metrics else 0
    direct_tokens = direct_metrics.total_tokens if direct_metrics else 0
    token_diff = abs(rlm_tokens - direct_tokens)
    token_winner = "Direct" if direct_tokens < rlm_tokens else "RLM"
    
    with col1:
        st.metric(
            "Tokens",
            f"{rlm_tokens:,} vs {direct_tokens:,}",
            f"Δ {token_diff:,} ({token_winner} wins)" if token_diff > 0 else "Equal"
        )
    
    # Cost metric
    rlm_cost = rlm_metrics.cost_usd if rlm_metrics else 0
    direct_cost = direct_metrics.cost_usd if direct_metrics else 0
    cost_diff = abs(rlm_cost - direct_cost)
    cost_pct = (cost_diff / max(direct_cost, rlm_cost) * 100) if max(direct_cost, rlm_cost) > 0 else 0
    cost_winner = "Direct" if direct_cost < rlm_cost else "RLM"
    
    with col2:
        st.metric(
            "Cost",
            f"${rlm_cost:.4f} vs ${direct_cost:.4f}",
            f"Δ ${cost_diff:.4f} ({cost_pct:.0f}%) ({cost_winner} wins)" if cost_diff > 0 else "Equal"
        )
    
    # Time metric
    rlm_time = rlm_metrics.execution_time_seconds if rlm_metrics else 0
    direct_time = direct_metrics.execution_time_seconds if direct_metrics else 0
    time_diff = abs(rlm_time - direct_time)
    time_pct = (time_diff / max(direct_time, rlm_time) * 100) if max(direct_time, rlm_time) > 0 else 0
    time_winner = "Direct" if direct_time < rlm_time else "RLM"
    
    with col3:
        st.metric(
            "Time",
            f"{rlm_time:.2f}s vs {direct_time:.2f}s",
            f"Δ {time_diff:.2f}s ({time_pct:.0f}%) ({time_winner} wins)" if time_diff > 0 else "Equal"
        )
    
    # Recommendation metric
    comparison = latest_message.comparison_metrics
    recommendation = comparison.recommendation if comparison else 'No recommendation'
    
    with col4:
        if "faster" in recommendation.lower() or "direct" in recommendation.lower():
            st.metric("Recommendation", "✅ Direct", "More efficient")
        elif "rlm" in recommendation.lower():
            st.metric("Recommendation", "✅ RLM", "More thorough")
        else:
            st.metric("Recommendation", "⚠️ Neutral", "Trade-offs")


def render_metrics_comparison():
    """Render detailed metrics comparison table."""
    
    st.subheader("📋 Detailed Metrics Comparison")
    
    latest_message = get_latest_message()
    if not latest_message:
        st.warning("No metrics available")
        return
    
    # Access dataclass attributes directly
    rlm_metrics = latest_message.rlm_metrics
    direct_metrics = latest_message.direct_metrics
    comparison = latest_message.comparison_metrics
    
    # Build comparison dataframe
    data = {
        'Metric': [
            'Steps',
            'Total Tokens',
            'Input Tokens',
            'Output Tokens',
            'Execution Time (s)',
            'Cost (USD)',
            'Memory Used (MB)',
            'Success'
        ],
        'RLM': [
            rlm_metrics.steps_taken if rlm_metrics else 0,
            f"{rlm_metrics.total_tokens if rlm_metrics else 0:,}",
            f"{rlm_metrics.input_tokens if rlm_metrics else 0:,}",
            f"{rlm_metrics.output_tokens if rlm_metrics else 0:,}",
            f"{rlm_metrics.execution_time_seconds if rlm_metrics else 0:.3f}",
            f"${rlm_metrics.cost_usd if rlm_metrics else 0:.6f}",
            f"{rlm_metrics.memory_used_mb if rlm_metrics else 0:.1f}",
            "✅" if (rlm_metrics and rlm_metrics.success) else "❌"
        ],
        'Direct': [
            direct_metrics.steps_taken if direct_metrics else 0,
            f"{direct_metrics.total_tokens if direct_metrics else 0:,}",
            f"{direct_metrics.input_tokens if direct_metrics else 0:,}",
            f"{direct_metrics.output_tokens if direct_metrics else 0:,}",
            f"{direct_metrics.execution_time_seconds if direct_metrics else 0:.3f}",
            f"${direct_metrics.cost_usd if direct_metrics else 0:.6f}",
            f"{direct_metrics.memory_used_mb if direct_metrics else 0:.1f}",
            "✅" if (direct_metrics and direct_metrics.success) else "❌"
        ],
        'Delta': [
            (rlm_metrics.steps_taken if rlm_metrics else 0) - (direct_metrics.steps_taken if direct_metrics else 0),
            f"{comparison.token_delta if comparison else 0:,}",
            f"{(rlm_metrics.input_tokens if rlm_metrics else 0) - (direct_metrics.input_tokens if direct_metrics else 0):,}",
            f"{(rlm_metrics.output_tokens if rlm_metrics else 0) - (direct_metrics.output_tokens if direct_metrics else 0):,}",
            f"{comparison.time_delta_seconds if comparison else 0:.3f}",
            f"${comparison.cost_delta_usd if comparison else 0:+.6f}",
            f"{(rlm_metrics.memory_used_mb if rlm_metrics else 0) - (direct_metrics.memory_used_mb if direct_metrics else 0):+.1f}",
            "-"
        ]
    }
    
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_cost_analysis():
    """Render cost breakdown and comparison charts."""
    
    st.subheader("💰 Cost Analysis")
    
    latest_message = get_latest_message()
    if not latest_message:
        st.warning("No metrics available")
        return
    
    # Access dataclass attributes directly
    rlm_metrics = latest_message.rlm_metrics
    direct_metrics = latest_message.direct_metrics
    
    col1, col2 = st.columns(2)
    
    # Left: Cost breakdown table
    with col1:
        st.write("**RLM Cost Breakdown**")
        
        rlm_breakdown = rlm_metrics.cost_breakdown if rlm_metrics else {}
        rlm_input_cost = rlm_breakdown.get('input', 0)
        rlm_output_cost = rlm_breakdown.get('output', 0)
        rlm_total = rlm_metrics.cost_usd if rlm_metrics else 0
        
        rlm_data = {
            'Category': ['Input', 'Output', 'Total'],
            'Cost (USD)': [
                f"${rlm_input_cost:.6f}",
                f"${rlm_output_cost:.6f}",
                f"${rlm_total:.6f}"
            ],
            'Share': [
                f"{(rlm_input_cost/max(rlm_total, 0.0001)*100):.1f}%" if rlm_total > 0 else "0%",
                f"{(rlm_output_cost/max(rlm_total, 0.0001)*100):.1f}%" if rlm_total > 0 else "0%",
                "100%"
            ]
        }
        
        st.dataframe(rlm_data, use_container_width=True, hide_index=True)
    
    # Right: Direct cost breakdown
    with col2:
        st.write("**Direct Cost Breakdown**")
        
        direct_breakdown = direct_metrics.cost_breakdown if direct_metrics else {}
        direct_input_cost = direct_breakdown.get('input', 0)
        direct_output_cost = direct_breakdown.get('output', 0)
        direct_total = direct_metrics.cost_usd if direct_metrics else 0
        
        direct_data = {
            'Category': ['Input', 'Output', 'Total'],
            'Cost (USD)': [
                f"${direct_input_cost:.6f}",
                f"${direct_output_cost:.6f}",
                f"${direct_total:.6f}"
            ],
            'Share': [
                f"{(direct_input_cost/max(direct_total, 0.0001)*100):.1f}%" if direct_total > 0 else "0%",
                f"{(direct_output_cost/max(direct_total, 0.0001)*100):.1f}%" if direct_total > 0 else "0%",
                "100%"
            ]
        }
        
        st.dataframe(direct_data, use_container_width=True, hide_index=True)
    
    # Cost comparison chart
    st.write("**Cost Comparison Chart**")
    
    fig = go.Figure(data=[
        go.Bar(name='RLM', x=['Input', 'Output'], y=[rlm_input_cost, rlm_output_cost]),
        go.Bar(name='Direct', x=['Input', 'Output'], y=[direct_input_cost, direct_output_cost])
    ])
    
    fig.update_layout(
        title="Cost Comparison by Category",
        xaxis_title="Cost Category",
        yaxis_title="Cost (USD)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_quality_metrics():
    """Render quality metrics and recommendations."""
    
    st.subheader("⭐ Quality Metrics & Recommendations")
    
    latest_message = get_latest_message()
    if not latest_message:
        st.warning("No metrics available")
        return
    
    # Access dataclass attributes directly
    comparison = latest_message.comparison_metrics
    rlm_metrics = latest_message.rlm_metrics
    direct_metrics = latest_message.direct_metrics
    
    col1, col2 = st.columns(2)
    
    # Left: Performance deltas
    with col1:
        st.write("**Performance Deltas**")
        
        token_delta = comparison.token_delta if comparison else 0
        cost_delta = comparison.cost_delta_usd if comparison else 0
        time_delta = comparison.time_delta_seconds if comparison else 0
        
        deltas_data = {
            'Metric': ['Tokens', 'Cost', 'Time'],
            'Delta': [
                f"{token_delta:+,}",
                f"${cost_delta:+.6f}",
                f"{time_delta:+.3f}s"
            ],
            'Percentage': [
                f"{token_delta:+}",
                f"{comparison.cost_delta_percent if comparison else 0:+.1f}%",
                f"{comparison.time_delta_percent if comparison else 0:+.1f}%"
            ],
            'Winner': [
                "Direct" if token_delta > 0 else "RLM" if token_delta < 0 else "Tie",
                "Direct" if cost_delta > 0 else "RLM" if cost_delta < 0 else "Tie",
                "Direct" if time_delta > 0 else "RLM" if time_delta < 0 else "Tie"
            ]
        }
        
        st.dataframe(deltas_data, use_container_width=True, hide_index=True)
    
    # Right: Recommendation
    with col2:
        st.write("**Recommendation**")
        
        recommendation = comparison.recommendation if comparison else 'No recommendation available'
        reasoning = comparison.reasoning if (comparison and hasattr(comparison, 'reasoning')) else 'Based on trade-off analysis'
        
        # Style recommendation
        if "faster" in recommendation.lower() or "direct" in recommendation.lower():
            st.success(f"✅ {recommendation}")
        elif "rlm" in recommendation.lower():
            st.info(f"ℹ️ {recommendation}")
        else:
            st.warning(f"⚠️ {recommendation}")
        
        st.caption(f"_Reasoning:_ {reasoning}")
    
    # Token efficiency chart
    st.write("**Token Efficiency (Cost per Token)**")
    
    rlm_cost = rlm_metrics.cost_usd if rlm_metrics else 0
    rlm_tokens = rlm_metrics.total_tokens if rlm_metrics else 1
    direct_cost = direct_metrics.cost_usd if direct_metrics else 0
    direct_tokens = direct_metrics.total_tokens if direct_metrics else 1
    
    rlm_efficiency = (rlm_cost / max(rlm_tokens, 1)) * 1000000
    direct_efficiency = (direct_cost / max(direct_tokens, 1)) * 1000000
    
    fig = go.Figure(data=[
        go.Bar(x=['RLM', 'Direct'], y=[rlm_efficiency, direct_efficiency])
    ])
    
    fig.update_layout(
        title="Cost per Token (micro-USD)",
        yaxis_title="Cost per Token (μ$)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


class MessageWrapper:
    """Wrapper to access dict message data with attribute syntax."""
    def __init__(self, data):
        self._data = data

    def __getattr__(self, name):
        value = self._data.get(name)
        # If value is a dict, wrap it too
        if isinstance(value, dict):
            return MessageWrapper(value)
        return value


def get_latest_message():
    """Get the latest message with metrics from chat history."""

    messages = st.session_state.get('chat_messages', [])

    # Find latest message with complete metrics (stored as dict)
    for message in reversed(messages):
        # Messages are dicts, not ChatMessage objects - access via dict keys
        if (isinstance(message, dict) and
            message.get('comparison_metrics') is not None):
            return MessageWrapper(message)

    return None


# Main entry point
if __name__ == "__main__":
    render_analysis_page()
