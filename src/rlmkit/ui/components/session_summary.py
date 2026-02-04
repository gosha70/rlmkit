# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""Sidebar session summary component.

Renders a compact summary of session-wide metrics in the sidebar.
Designed to be called from every page's sidebar rendering code.
"""
from __future__ import annotations

from datetime import datetime

import streamlit as st

from rlmkit.ui.services.analytics_engine import AnalyticsEngine
from rlmkit.ui.utils import format_memory


def render_session_summary() -> None:
    """Render session metrics summary in the sidebar.

    Should be called inside a ``with st.sidebar:`` block (or when the
    sidebar context is already active).  Reads messages from
    ``st.session_state.chat_messages`` and computes aggregates via
    :class:`AnalyticsEngine`.
    """
    messages = st.session_state.get("chat_messages", [])
    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

    if not assistant_msgs:
        st.caption("No session metrics yet")
        return

    analytics = AnalyticsEngine(messages).compute()

    # Duration
    start = st.session_state.get("session_start_time")
    if start and isinstance(start, datetime):
        duration_s = (datetime.now() - start).total_seconds()
        if duration_s >= 3600:
            duration_label = f"{duration_s / 3600:.1f}h"
        elif duration_s >= 60:
            duration_label = f"{duration_s / 60:.0f}m"
        else:
            duration_label = f"{duration_s:.0f}s"
    else:
        duration_label = "-"

    st.markdown("**Session Summary**")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", analytics.total_requests)
    with col2:
        st.metric("Duration", duration_label)

    col3, col4 = st.columns(2)
    with col3:
        st.metric("Total Tokens", f"{analytics.total_tokens:,}")
    with col4:
        st.metric("Total Cost", f"${analytics.total_cost_usd:.4f}")

    col5, col6 = st.columns(2)
    with col5:
        st.metric("Avg Speed", f"{analytics.avg_response_time_seconds:.1f}s")
    with col6:
        st.metric("Peak Mem", format_memory(analytics.peak_memory_mb))

    # RLM savings (only show if there are savings)
    if analytics.rlm_savings_tokens > 0 or analytics.rlm_savings_cost > 0:
        st.caption(
            f"RLM savings: {analytics.rlm_savings_tokens:,} tokens "
            f"(${analytics.rlm_savings_cost:.4f})"
        )
