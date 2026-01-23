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
    
    # Check if we have any chat history
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
    
    # Get metrics
    rlm_metrics = latest_message.get('rlm_metrics', {})
    direct_metrics = latest_message.get('direct_metrics', {})
    
    # Tokens metric
    rlm_tokens = rlm_metrics.get('total_tokens', 0)
    direct_tokens = direct_metrics.get('total_tokens', 0)
    token_diff = abs(rlm_tokens - direct_tokens)
    token_winner = "Direct" if direct_tokens < rlm_tokens else "RLM"
    
    with col1:
        st.metric(
            "Tokens",
            f"{rlm_tokens:,} vs {direct_tokens:,}",
            f"Δ {token_diff:,} ({token_winner} wins)" if token_diff > 0 else "Equal"
        )
    
    # Cost metric
    rlm_cost = rlm_metrics.get('cost_usd', 0)
    direct_cost = direct_metrics.get('cost_usd', 0)
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
    rlm_time = rlm_metrics.get('execution_time_seconds', 0)
    direct_time = direct_metrics.get('execution_time_seconds', 0)
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
    comparison = latest_message.get('comparison_metrics', {})
    recommendation = comparison.get('recommendation', 'No recommendation')
    
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
    
    rlm_metrics = latest_message.get('rlm_metrics', {})
    direct_metrics = latest_message.get('direct_metrics', {})
    comparison = latest_message.get('comparison_metrics', {})
    
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
            rlm_metrics.get('steps_taken', 0),
            f"{rlm_metrics.get('total_tokens', 0):,}",
            f"{rlm_metrics.get('input_tokens', 0):,}",
            f"{rlm_metrics.get('output_tokens', 0):,}",
            f"{rlm_metrics.get('execution_time_seconds', 0):.3f}",
            f"${rlm_metrics.get('cost_usd', 0):.6f}",
            f"{rlm_metrics.get('memory_used_mb', 0):.1f}",
            "✅" if rlm_metrics.get('success', False) else "❌"
        ],
        'Direct': [
            direct_metrics.get('steps_taken', 0),
            f"{direct_metrics.get('total_tokens', 0):,}",
            f"{direct_metrics.get('input_tokens', 0):,}",
            f"{direct_metrics.get('output_tokens', 0):,}",
            f"{direct_metrics.get('execution_time_seconds', 0):.3f}",
            f"${direct_metrics.get('cost_usd', 0):.6f}",
            f"{direct_metrics.get('memory_used_mb', 0):.1f}",
            "✅" if direct_metrics.get('success', False) else "❌"
        ],
        'Delta': [
            rlm_metrics.get('steps_taken', 0) - direct_metrics.get('steps_taken', 0),
            f"{comparison.get('token_delta', 0):,}",
            f"{rlm_metrics.get('input_tokens', 0) - direct_metrics.get('input_tokens', 0):,}",
            f"{rlm_metrics.get('output_tokens', 0) - direct_metrics.get('output_tokens', 0):,}",
            f"{comparison.get('time_delta_seconds', 0):.3f}",
            f"${comparison.get('cost_delta_usd', 0):+.6f}",
            f"{rlm_metrics.get('memory_used_mb', 0) - direct_metrics.get('memory_used_mb', 0):+.1f}",
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
    
    rlm_metrics = latest_message.get('rlm_metrics', {})
    direct_metrics = latest_message.get('direct_metrics', {})
    
    col1, col2 = st.columns(2)
    
    # Left: Cost breakdown table
    with col1:
        st.write("**RLM Cost Breakdown**")
        
        rlm_breakdown = rlm_metrics.get('cost_breakdown', {})
        rlm_input_cost = rlm_breakdown.get('input', 0)
        rlm_output_cost = rlm_breakdown.get('output', 0)
        rlm_total = rlm_metrics.get('cost_usd', 0)
        
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
        
        direct_breakdown = direct_metrics.get('cost_breakdown', {})
        direct_input_cost = direct_breakdown.get('input', 0)
        direct_output_cost = direct_breakdown.get('output', 0)
        direct_total = direct_metrics.get('cost_usd', 0)
        
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
    
    comparison = latest_message.get('comparison_metrics', {})
    rlm_metrics = latest_message.get('rlm_metrics', {})
    direct_metrics = latest_message.get('direct_metrics', {})
    
    col1, col2 = st.columns(2)
    
    # Left: Performance deltas
    with col1:
        st.write("**Performance Deltas**")
        
        token_delta = comparison.get('token_delta', 0)
        cost_delta = comparison.get('cost_delta_usd', 0)
        time_delta = comparison.get('time_delta_seconds', 0)
        
        deltas_data = {
            'Metric': ['Tokens', 'Cost', 'Time'],
            'Delta': [
                f"{token_delta:+,}",
                f"${cost_delta:+.6f}",
                f"{time_delta:+.3f}s"
            ],
            'Percentage': [
                f"{comparison.get('token_delta', 0):+}",
                f"{comparison.get('cost_delta_percent', 0):+.1f}%",
                f"{comparison.get('time_delta_percent', 0):+.1f}%"
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
        
        recommendation = comparison.get('recommendation', 'No recommendation available')
        reasoning = comparison.get('reasoning', 'Based on trade-off analysis')
        
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
    
    rlm_efficiency = (rlm_metrics.get('cost_usd', 0) / max(rlm_metrics.get('total_tokens', 1), 1)) * 1000000
    direct_efficiency = (direct_metrics.get('cost_usd', 0) / max(direct_metrics.get('total_tokens', 1), 1)) * 1000000
    
    fig = go.Figure(data=[
        go.Bar(x=['RLM', 'Direct'], y=[rlm_efficiency, direct_efficiency])
    ])
    
    fig.update_layout(
        title="Cost per Token (micro-USD)",
        yaxis_title="Cost per Token (μ$)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def get_latest_message():
    """Get the latest assistant message with metrics from chat history."""
    
    messages = st.session_state.get('chat_messages', [])
    
    # Find latest assistant message with complete metrics
    for message in reversed(messages):
        if (message.get('role') == 'assistant' and 
            message.get('rlm_metrics') and 
            message.get('direct_metrics')):
            return message
    
    return None


# Main entry point
if __name__ == "__main__":
    render_analysis_page()
