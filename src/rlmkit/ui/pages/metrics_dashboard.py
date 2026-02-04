# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
Metrics Dashboard — Session-wide monitoring and telemetry.

Page 3 of RLM Studio: aggregated metrics, interactive charts,
query-level insights, advanced analytics, and CSV export.
"""
from __future__ import annotations

import io
from typing import TYPE_CHECKING

import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # type: ignore

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None  # type: ignore

from rlmkit.ui.services.analytics_engine import AnalyticsEngine, SessionAnalytics
from rlmkit.ui.components.navigation import render_custom_navigation
from rlmkit.ui.components.session_summary import render_session_summary
from rlmkit.ui.utils import format_memory

if TYPE_CHECKING:
    import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color palette (matches existing charts.py)
# ---------------------------------------------------------------------------
COLOR_RLM = "#3498db"
COLOR_DIRECT = "#e74c3c"
COLOR_RAG = "#2ecc71"
COLOR_INPUT = "lightblue"
COLOR_OUTPUT = "lightcoral"


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render_metrics_dashboard() -> None:
    """Render the full Metrics Dashboard page."""

    st.session_state.current_nav_page = "metrics"

    try:
        st.set_page_config(
            page_title="Metrics - RLM Studio",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="auto",
        )
    except Exception:
        pass

    # Hide default Streamlit nav
    st.markdown(
        """<style>
        [data-testid="stSidebarNav"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    # Try to load global CSS
    try:
        from rlmkit.ui.app import _inject_rlmkit_desktop_css
        _inject_rlmkit_desktop_css()
    except Exception:
        pass

    # Sidebar: navigation + session summary
    render_custom_navigation()
    with st.sidebar:
        st.divider()
        render_session_summary()

    # Page content
    st.title("📈 Metrics Dashboard")
    st.markdown("Session-wide monitoring and telemetry")

    messages = st.session_state.get("chat_messages", [])
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

    if not assistant_msgs:
        st.info("No data yet. Visit **Chat** to start a conversation.")
        return

    analytics = AnalyticsEngine(messages).compute()

    _render_overview_cards(analytics)
    st.divider()
    _render_interactive_charts(analytics)
    st.divider()
    _render_query_insights(analytics)
    st.divider()
    _render_advanced_analytics(analytics)
    st.divider()
    _render_export_section(analytics)


# ---------------------------------------------------------------------------
# Section A: Overview Cards
# ---------------------------------------------------------------------------

def _render_overview_cards(a: SessionAnalytics) -> None:
    """Render 6 overview metric cards."""
    st.subheader("Overview")

    row1_c1, row1_c2, row1_c3 = st.columns(3)

    with row1_c1:
        st.metric(
            "Total Tokens",
            f"{a.total_tokens:,}",
            help="Sum of all tokens across all modes and messages",
        )
    with row1_c2:
        st.metric(
            "Total Cost",
            f"${a.total_cost_usd:.4f}",
            help="Sum of all costs across all modes and messages",
        )
    with row1_c3:
        st.metric(
            "Total Requests",
            a.total_requests,
            help="Number of assistant responses (across all modes)",
        )

    row2_c1, row2_c2, row2_c3 = st.columns(3)

    with row2_c1:
        st.metric(
            "Avg Speed",
            f"{a.avg_response_time_seconds:.2f}s",
            help="Average execution time per request",
        )
    with row2_c2:
        st.metric(
            "Peak Memory",
            format_memory(a.peak_memory_mb),
            help="Highest memory peak across all queries",
        )
    with row2_c3:
        score_label = f"{a.efficiency_score:.0f}/100"
        if a.efficiency_score >= 80:
            st.metric("Efficiency", score_label, delta="Good")
        elif a.efficiency_score >= 50:
            st.metric("Efficiency", score_label, delta="OK", delta_color="off")
        else:
            st.metric("Efficiency", score_label, delta="Low", delta_color="inverse")

    # Mode breakdown
    if a.rlm_count + a.direct_count + a.rag_count > 0:
        parts = []
        if a.rlm_count:
            parts.append(f"RLM: {a.rlm_count}")
        if a.direct_count:
            parts.append(f"Direct: {a.direct_count}")
        if a.rag_count:
            parts.append(f"RAG: {a.rag_count}")
        st.caption(" · ".join(parts))

    if a.rlm_savings_tokens > 0 or a.rlm_savings_cost > 0:
        st.success(
            f"RLM saved **{a.rlm_savings_tokens:,} tokens** "
            f"(**${a.rlm_savings_cost:.4f}**) compared to Direct in compare-mode queries"
        )


# ---------------------------------------------------------------------------
# Section B: Interactive Charts
# ---------------------------------------------------------------------------

def _render_interactive_charts(a: SessionAnalytics) -> None:
    """Render 5 interactive Plotly charts."""
    if not PLOTLY_AVAILABLE:
        st.warning("Plotly is not installed. Charts are unavailable.")
        return

    st.subheader("Charts")

    # Layout: 2 charts per row
    col_left, col_right = st.columns(2)

    with col_left:
        _chart_token_trends(a)
    with col_right:
        _chart_cost_efficiency(a)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        _chart_memory_timeline(a)
    with col_right2:
        _chart_request_breakdown(a)

    # Radar chart full-width
    _chart_performance_radar(a)


def _chart_token_trends(a: SessionAnalytics) -> None:
    """Line chart: tokens per query over time."""
    if not a.token_trends:
        return

    indices = [t["index"] for t in a.token_trends]
    fig = go.Figure()

    rlm_vals = [t["rlm_tokens"] for t in a.token_trends]
    direct_vals = [t["direct_tokens"] for t in a.token_trends]
    rag_vals = [t["rag_tokens"] for t in a.token_trends]

    if any(v > 0 for v in rlm_vals):
        fig.add_trace(go.Scatter(
            x=indices, y=rlm_vals, mode="lines+markers",
            name="RLM", line=dict(color=COLOR_RLM),
        ))
    if any(v > 0 for v in direct_vals):
        fig.add_trace(go.Scatter(
            x=indices, y=direct_vals, mode="lines+markers",
            name="Direct", line=dict(color=COLOR_DIRECT),
        ))
    if any(v > 0 for v in rag_vals):
        fig.add_trace(go.Scatter(
            x=indices, y=rag_vals, mode="lines+markers",
            name="RAG", line=dict(color=COLOR_RAG),
        ))

    fig.update_layout(
        title="Token Trends Over Time",
        xaxis_title="Query #",
        yaxis_title="Tokens",
        hovermode="x unified",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_cost_efficiency(a: SessionAnalytics) -> None:
    """Bar chart: total cost by mode (input vs output)."""
    if a.total_cost_usd == 0:
        st.caption("No cost data available")
        return

    fig = go.Figure()

    modes = []
    input_costs = []
    output_costs = []

    if a.rlm_total_cost > 0:
        modes.append("RLM")
        # Approximate split: 40% input, 60% output (consistent with pricing)
        input_costs.append(a.rlm_total_cost * 0.4)
        output_costs.append(a.rlm_total_cost * 0.6)

    if a.direct_total_cost > 0:
        modes.append("Direct")
        input_costs.append(a.direct_total_cost * 0.4)
        output_costs.append(a.direct_total_cost * 0.6)

    if a.rag_total_cost > 0:
        modes.append("RAG")
        input_costs.append(a.rag_total_cost * 0.4)
        output_costs.append(a.rag_total_cost * 0.6)

    fig.add_trace(go.Bar(
        name="Input Cost", x=modes, y=input_costs,
        marker_color=COLOR_INPUT,
        text=[f"${c:.4f}" for c in input_costs], textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Output Cost", x=modes, y=output_costs,
        marker_color=COLOR_OUTPUT,
        text=[f"${c:.4f}" for c in output_costs], textposition="auto",
    ))

    fig.update_layout(
        title="Cost Breakdown by Mode",
        xaxis_title="Mode",
        yaxis_title="Cost (USD)",
        barmode="stack",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_memory_timeline(a: SessionAnalytics) -> None:
    """Area chart: peak memory per query."""
    if not a.memory_timeline:
        return

    indices = [t["index"] for t in a.memory_timeline]
    peaks = [t["memory_peak"] for t in a.memory_timeline]

    if all(p == 0 for p in peaks):
        st.caption("No memory data available")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=indices, y=peaks, mode="lines+markers",
        fill="tozeroy", name="Peak Memory",
        line=dict(color="#9b59b6"),
    ))
    fig.update_layout(
        title="Memory Usage Timeline",
        xaxis_title="Query #",
        yaxis_title="Memory (MB)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_request_breakdown(a: SessionAnalytics) -> None:
    """Donut chart: request count by mode."""
    labels = []
    values = []

    if a.rlm_count > 0:
        labels.append("RLM")
        values.append(a.rlm_count)
    if a.direct_count > 0:
        labels.append("Direct")
        values.append(a.direct_count)
    if a.rag_count > 0:
        labels.append("RAG")
        values.append(a.rag_count)

    if not values:
        st.caption("No request data")
        return

    color_map = {"RLM": COLOR_RLM, "Direct": COLOR_DIRECT, "RAG": COLOR_RAG}
    colors = [color_map.get(l, "#95a5a6") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.4, marker_colors=colors,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Request Distribution",
        height=400,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


def _chart_performance_radar(a: SessionAnalytics) -> None:
    """Radar chart: RLM vs Direct normalized comparison."""
    if a.rlm_count == 0 or a.direct_count == 0:
        return

    # Compute averages
    rlm_avg_tokens = a.rlm_total_tokens / a.rlm_count
    direct_avg_tokens = a.direct_total_tokens / a.direct_count
    rlm_avg_cost = a.rlm_total_cost / a.rlm_count
    direct_avg_cost = a.direct_total_cost / a.direct_count

    # Compute avg time from time_trends
    rlm_times = [t["rlm_time"] for t in a.time_trends if t["rlm_time"] > 0]
    direct_times = [t["direct_time"] for t in a.time_trends if t["direct_time"] > 0]
    rlm_avg_time = sum(rlm_times) / len(rlm_times) if rlm_times else 0
    direct_avg_time = sum(direct_times) / len(direct_times) if direct_times else 0

    # Normalize: higher is better, so invert metrics where lower is better
    def _norm_inv(val, max_val):
        """1 = best, 0 = worst. Lower original value → higher score."""
        if max_val == 0:
            return 0.5
        return 1.0 - (val / max_val)

    max_tokens = max(rlm_avg_tokens, direct_avg_tokens, 1)
    max_cost = max(rlm_avg_cost, direct_avg_cost, 0.001)
    max_time = max(rlm_avg_time, direct_avg_time, 0.1)

    categories = ["Token Efficiency", "Cost Efficiency", "Speed"]

    rlm_vals = [
        _norm_inv(rlm_avg_tokens, max_tokens),
        _norm_inv(rlm_avg_cost, max_cost),
        _norm_inv(rlm_avg_time, max_time),
    ]
    direct_vals = [
        _norm_inv(direct_avg_tokens, max_tokens),
        _norm_inv(direct_avg_cost, max_cost),
        _norm_inv(direct_avg_time, max_time),
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=rlm_vals + [rlm_vals[0]],
        theta=categories + [categories[0]],
        fill="toself", name="RLM",
        line_color=COLOR_RLM,
    ))
    fig.add_trace(go.Scatterpolar(
        r=direct_vals + [direct_vals[0]],
        theta=categories + [categories[0]],
        fill="toself", name="Direct",
        line_color=COLOR_DIRECT,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Performance Radar (RLM vs Direct)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Section C: Query-Level Insights
# ---------------------------------------------------------------------------

def _render_query_insights(a: SessionAnalytics) -> None:
    """Render per-query metrics table."""
    st.subheader("Query-Level Insights")

    if not a.query_details:
        st.info("No query data")
        return

    if not PANDAS_AVAILABLE:
        # Fallback: render as simple markdown
        for i, d in enumerate(a.query_details, 1):
            st.write(
                f"**Q{i}** | {d['mode']} | {d['status']} | "
                f"{d['tokens']:,} tokens | ${d['cost']:.4f} | "
                f"{d['time']:.2f}s | {d['steps']} steps"
            )
        return

    rows = []
    for i, d in enumerate(a.query_details, 1):
        query_text = d.get("query", "")
        if len(query_text) > 60:
            query_text = query_text[:57] + "..."
        rows.append({
            "#": i,
            "Query": query_text,
            "Mode": d["mode"],
            "Status": d["status"],
            "Tokens": f"{d['tokens']:,}",
            "Cost": f"${d['cost']:.4f}",
            "Time": f"{d['time']:.2f}s",
            "Steps": d["steps"],
            "Memory": format_memory(d['memory']),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Section D: Advanced Analytics
# ---------------------------------------------------------------------------

def _render_advanced_analytics(a: SessionAnalytics) -> None:
    """Render advanced analytics in a collapsible section."""
    with st.expander("Advanced Analytics", expanded=False):
        _render_mode_averages(a)
        _render_cost_speed_scatter(a)
        _render_outliers(a)


def _render_mode_averages(a: SessionAnalytics) -> None:
    """Show average metrics per mode."""
    st.markdown("**Mode Performance Averages**")

    rows = []
    if a.rlm_count > 0:
        rlm_times = [t["rlm_time"] for t in a.time_trends if t["rlm_time"] > 0]
        avg_time = sum(rlm_times) / len(rlm_times) if rlm_times else 0
        rows.append({
            "Mode": "RLM",
            "Queries": a.rlm_count,
            "Avg Tokens": f"{a.rlm_total_tokens / a.rlm_count:,.0f}",
            "Avg Cost": f"${a.rlm_total_cost / a.rlm_count:.4f}",
            "Avg Time": f"{avg_time:.2f}s",
        })
    if a.direct_count > 0:
        direct_times = [t["direct_time"] for t in a.time_trends if t["direct_time"] > 0]
        avg_time = sum(direct_times) / len(direct_times) if direct_times else 0
        rows.append({
            "Mode": "Direct",
            "Queries": a.direct_count,
            "Avg Tokens": f"{a.direct_total_tokens / a.direct_count:,.0f}",
            "Avg Cost": f"${a.direct_total_cost / a.direct_count:.4f}",
            "Avg Time": f"{avg_time:.2f}s",
        })
    if a.rag_count > 0:
        rag_times = [t["rag_time"] for t in a.time_trends if t.get("rag_time", 0) > 0]
        avg_time = sum(rag_times) / len(rag_times) if rag_times else 0
        rows.append({
            "Mode": "RAG",
            "Queries": a.rag_count,
            "Avg Tokens": f"{a.rag_total_tokens / a.rag_count:,.0f}",
            "Avg Cost": f"${a.rag_total_cost / a.rag_count:.4f}",
            "Avg Time": f"{avg_time:.2f}s",
        })

    if rows and PANDAS_AVAILABLE:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    elif rows:
        for r in rows:
            st.write(f"**{r['Mode']}**: {r['Queries']} queries, "
                     f"{r['Avg Tokens']} tokens, {r['Avg Cost']}, {r['Avg Time']}")


def _render_cost_speed_scatter(a: SessionAnalytics) -> None:
    """Scatter plot: cost vs time per query."""
    if not PLOTLY_AVAILABLE or not a.query_details:
        return

    st.markdown("**Cost vs Speed Trade-off**")

    costs = [d["cost"] for d in a.query_details]
    times = [d["time"] for d in a.query_details]
    modes = [d["mode"] for d in a.query_details]

    if all(c == 0 for c in costs) and all(t == 0 for t in times):
        st.caption("Insufficient data for scatter plot")
        return

    mode_colors = {
        "rlm_only": COLOR_RLM,
        "direct_only": COLOR_DIRECT,
        "rag_only": COLOR_RAG,
        "compare": "#9b59b6",
    }

    fig = go.Figure()
    for mode_key, color in mode_colors.items():
        idx = [i for i, m in enumerate(modes) if m == mode_key]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x=[times[i] for i in idx],
            y=[costs[i] for i in idx],
            mode="markers",
            name=mode_key,
            marker=dict(color=color, size=10),
            text=[f"Q{i+1}" for i in idx],
        ))

    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="Cost (USD)",
        height=400,
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_outliers(a: SessionAnalytics) -> None:
    """Show flagged outlier queries."""
    st.markdown("**Outlier Detection**")

    if not a.outlier_indices:
        st.caption("No outliers detected (all queries within normal range)")
        return

    for idx in a.outlier_indices:
        if idx < len(a.query_details):
            d = a.query_details[idx]
            st.warning(
                f"**Query {idx + 1}** — {d['tokens']:,} tokens, "
                f"${d['cost']:.4f}, {d['time']:.2f}s "
                f"(significantly above average)"
            )


# ---------------------------------------------------------------------------
# Section E: Export
# ---------------------------------------------------------------------------

def _render_export_section(a: SessionAnalytics) -> None:
    """Render export buttons."""
    st.subheader("Export")

    col1, col2 = st.columns(2)

    with col1:
        csv_data = _build_csv(a)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="rlmkit_metrics.csv",
            mime="text/csv",
        )

    with col2:
        st.button(
            "Export PDF Report",
            disabled=True,
            help="PDF export is planned for a future release",
        )


def _build_csv(a: SessionAnalytics) -> str:
    """Build CSV string from query details."""
    output = io.StringIO()

    if PANDAS_AVAILABLE:
        rows = []
        for i, d in enumerate(a.query_details, 1):
            rows.append({
                "Query #": i,
                "Query": d.get("query", ""),
                "Mode": d["mode"],
                "Status": d["status"],
                "Tokens": d["tokens"],
                "Cost (USD)": d["cost"],
                "Time (s)": d["time"],
                "Steps": d["steps"],
                "Memory (MB)": d["memory"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(output, index=False)
    else:
        import csv
        writer = csv.writer(output)
        writer.writerow([
            "Query #", "Query", "Mode", "Status",
            "Tokens", "Cost (USD)", "Time (s)", "Steps", "Memory (MB)",
        ])
        for i, d in enumerate(a.query_details, 1):
            writer.writerow([
                i, d.get("query", ""), d["mode"], d["status"],
                d["tokens"], d["cost"], d["time"], d["steps"], d["memory"],
            ])

    return output.getvalue()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    render_metrics_dashboard()
