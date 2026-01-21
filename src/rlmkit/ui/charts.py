# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Chart and visualization utilities for RLMKit UI.

Creates interactive charts using Plotly to visualize:
- Token usage comparisons
- Cost comparisons
- Time comparisons
- Performance metrics
"""

from typing import Optional
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from rlmkit.core.comparison import ComparisonResult


def create_token_comparison_chart(result: ComparisonResult) -> Optional[go.Figure]:
    """
    Create bar chart comparing token usage between RLM and Direct modes.
    
    Args:
        result: ComparisonResult with metrics
        
    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not result.rlm_metrics or not result.direct_metrics:
        return None
    
    # Prepare data
    modes = ['RLM Mode', 'Direct Mode']
    input_tokens = [
        result.rlm_metrics.tokens.input_tokens,
        result.direct_metrics.tokens.input_tokens
    ]
    output_tokens = [
        result.rlm_metrics.tokens.output_tokens,
        result.direct_metrics.tokens.output_tokens
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add input tokens
    fig.add_trace(go.Bar(
        name='Input Tokens',
        x=modes,
        y=input_tokens,
        marker_color='lightblue',
        text=input_tokens,
        textposition='auto',
    ))
    
    # Add output tokens
    fig.add_trace(go.Bar(
        name='Output Tokens',
        x=modes,
        y=output_tokens,
        marker_color='lightcoral',
        text=output_tokens,
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title='Token Usage Comparison',
        xaxis_title='Mode',
        yaxis_title='Token Count',
        barmode='stack',
        hovermode='x unified',
        height=400,
    )
    
    return fig


def create_cost_comparison_chart(result: ComparisonResult) -> Optional[go.Figure]:
    """
    Create bar chart comparing costs between modes.
    
    Args:
        result: ComparisonResult with metrics
        
    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not result.rlm_metrics or not result.direct_metrics:
        return None
    
    # Prepare data
    modes = ['RLM Mode', 'Direct Mode']
    costs = [
        result.rlm_metrics.cost,
        result.direct_metrics.cost
    ]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=modes,
        y=costs,
        marker_color=['#3498db', '#e74c3c'],
        text=[f'${c:.4f}' for c in costs],
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title='Cost Comparison',
        xaxis_title='Mode',
        yaxis_title='Cost (USD)',
        hovermode='x',
        height=400,
    )
    
    return fig


def create_time_comparison_chart(result: ComparisonResult) -> Optional[go.Figure]:
    """
    Create bar chart comparing execution times.
    
    Args:
        result: ComparisonResult with metrics
        
    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not result.rlm_metrics or not result.direct_metrics:
        return None
    
    # Prepare data
    modes = ['RLM Mode', 'Direct Mode']
    times = [
        result.rlm_metrics.elapsed_time,
        result.direct_metrics.elapsed_time
    ]
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=modes,
        y=times,
        marker_color=['#9b59b6', '#2ecc71'],
        text=[f'{t:.2f}s' for t in times],
        textposition='auto',
    ))
    
    # Update layout
    fig.update_layout(
        title='Execution Time Comparison',
        xaxis_title='Mode',
        yaxis_title='Time (seconds)',
        hovermode='x',
        height=400,
    )
    
    return fig


def create_metrics_radar_chart(result: ComparisonResult) -> Optional[go.Figure]:
    """
    Create radar chart comparing multiple metrics.
    
    Args:
        result: ComparisonResult with metrics
        
    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not result.rlm_metrics or not result.direct_metrics:
        return None
    
    # Normalize metrics to 0-1 scale for radar chart
    def normalize(value, max_val):
        return value / max_val if max_val > 0 else 0
    
    # Get max values for normalization
    max_tokens = max(
        result.rlm_metrics.tokens.total_tokens,
        result.direct_metrics.tokens.total_tokens
    )
    max_time = max(
        result.rlm_metrics.elapsed_time,
        result.direct_metrics.elapsed_time
    )
    max_steps = max(result.rlm_metrics.steps, 1)  # Direct is always 0
    
    # Categories
    categories = ['Token Efficiency', 'Speed', 'Simplicity']
    
    # RLM metrics (inverted so lower is better on radar)
    rlm_values = [
        1 - normalize(result.rlm_metrics.tokens.total_tokens, max_tokens),  # Lower tokens is better
        1 - normalize(result.rlm_metrics.elapsed_time, max_time),  # Lower time is better
        1 - normalize(result.rlm_metrics.steps, max_steps),  # Fewer steps is simpler
    ]
    
    # Direct metrics
    direct_values = [
        1 - normalize(result.direct_metrics.tokens.total_tokens, max_tokens),
        1 - normalize(result.direct_metrics.elapsed_time, max_time),
        1.0,  # Direct mode is always simplest (0 steps)
    ]
    
    # Create figure
    fig = go.Figure()
    
    # Add RLM trace
    fig.add_trace(go.Scatterpolar(
        r=rlm_values + [rlm_values[0]],  # Close the polygon
        theta=categories + [categories[0]],
        fill='toself',
        name='RLM Mode',
        line_color='#3498db',
    ))
    
    # Add Direct trace
    fig.add_trace(go.Scatterpolar(
        r=direct_values + [direct_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='Direct Mode',
        line_color='#e74c3c',
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title='Performance Comparison Radar',
        height=500,
    )
    
    return fig


def create_step_timeline_chart(result: ComparisonResult) -> Optional[go.Figure]:
    """
    Create timeline chart showing RLM execution steps.
    
    Args:
        result: ComparisonResult with RLM metrics
        
    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not result.rlm_metrics or not result.rlm_metrics.trace:
        return None
    
    # Extract step information
    steps = []
    roles = []
    colors = []
    
    color_map = {
        'assistant': '#3498db',
        'execution': '#2ecc71',
        'user': '#95a5a6',
    }
    
    for item in result.rlm_metrics.trace:
        step = item.get('step', 0)
        role = item.get('role', 'unknown')
        
        steps.append(step)
        roles.append(role.capitalize())
        colors.append(color_map.get(role, '#95a5a6'))
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=steps,
        y=[1] * len(steps),
        marker_color=colors,
        text=roles,
        textposition='auto',
        hovertext=[f"Step {s}: {r}" for s, r in zip(steps, roles)],
        hoverinfo='text',
    ))
    
    # Update layout
    fig.update_layout(
        title='RLM Execution Timeline',
        xaxis_title='Step',
        yaxis_title='',
        showlegend=False,
        height=300,
        yaxis_visible=False,
    )
    
    return fig


def create_token_breakdown_chart(result: ComparisonResult) -> Optional[go.Figure]:
    """
    Create pie charts showing token distribution.
    
    Args:
        result: ComparisonResult with metrics
        
    Returns:
        Plotly figure or None if data unavailable
    """
    if not PLOTLY_AVAILABLE:
        return None
    
    if not result.rlm_metrics and not result.direct_metrics:
        return None
    
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RLM Mode', 'Direct Mode'),
        specs=[[{'type': 'pie'}, {'type': 'pie'}]]
    )
    
    # RLM pie chart
    if result.rlm_metrics:
        fig.add_trace(
            go.Pie(
                labels=['Input', 'Output'],
                values=[
                    result.rlm_metrics.tokens.input_tokens,
                    result.rlm_metrics.tokens.output_tokens
                ],
                marker_colors=['lightblue', 'lightcoral'],
            ),
            row=1, col=1
        )
    
    # Direct pie chart
    if result.direct_metrics:
        fig.add_trace(
            go.Pie(
                labels=['Input', 'Output'],
                values=[
                    result.direct_metrics.tokens.input_tokens,
                    result.direct_metrics.tokens.output_tokens
                ],
                marker_colors=['lightblue', 'lightcoral'],
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        title_text='Token Distribution',
        height=400,
    )
    
    return fig
