# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Custom navigation component for RLM Studio."""

import streamlit as st
from pathlib import Path
import logging
logger = logging.getLogger("rlmkit.nav")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

def _nav_log(msg: str) -> None:
    logger.info(msg)

def get_current_page() -> str:
    """
    Detect the current page being displayed.

    Returns:
        Page identifier: 'chat', 'analysis', or 'configuration'
    """
    # Use session state to track current page (more reliable than script_path)
    if 'current_nav_page' not in st.session_state:
        st.session_state.current_nav_page = 'chat'

    current = st.session_state.current_nav_page
    _nav_log(f"get_current_page returning: {current}")
    return current


def render_custom_navigation():
    """
    Render custom navigation sidebar with clean labels and icons.

    Replaces Streamlit's default auto-generated navigation with custom buttons.
    Note: Default nav is hidden via CSS in styles.css.
    """
    
    # Detect current page
    current_page = get_current_page()
    _nav_log(f"render_custom_navigation current_page={current_page}")
    
    # Render navigation buttons in sidebar
    with st.sidebar:
        # Chat button  
        chat_class = "nav-button-active" if current_page == 'chat' else "nav-button"
        st.markdown(f'<div class="{chat_class}">', unsafe_allow_html=True)
        chat_btn = st.button("💬 Chat", key="nav_chat", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis button
        analysis_class = "nav-button-active" if current_page == 'analysis' else "nav-button"
        st.markdown(f'<div class="{analysis_class}">', unsafe_allow_html=True)
        analysis_btn = st.button("📊 Analysis", key="nav_analysis", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Configuration button
        config_class = "nav-button-active" if current_page == 'configuration' else "nav-button"
        st.markdown(f'<div class="{config_class}">', unsafe_allow_html=True)
        config_btn = st.button("⚙️ Configuration", key="nav_config", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
    
    # Handle navigation after rendering
    _nav_log(f"buttons: chat={chat_btn} analysis={analysis_btn} config={config_btn}")
    if chat_btn:
        _nav_log("CLICK chat")
        st.session_state.current_nav_page = 'chat'
        st.switch_page("app.py")
    elif analysis_btn:
        _nav_log("CLICK analysis")
        st.session_state.current_nav_page = 'analysis'
        st.switch_page("pages/analysis_panel.py")
    elif config_btn:
        _nav_log("CLICK config")
        st.session_state.current_nav_page = 'configuration'
        st.switch_page("pages/configuration_panel.py")
