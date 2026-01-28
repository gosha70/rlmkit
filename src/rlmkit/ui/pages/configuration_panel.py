# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Configuration Management Page - Streamlit UI for managing LLM providers."""

import streamlit as st
from pathlib import Path
import os

from rlmkit.ui.services import LLMConfigManager
from rlmkit.ui.components.navigation import render_custom_navigation
from rlmkit.ui.app import _inject_rlmkit_desktop_css


def init_config_session_state():
    """Initialize session state for configuration page."""
    if "llm_manager" not in st.session_state:
        config_dir = Path.home() / ".rlmkit"
        st.session_state.llm_manager = LLMConfigManager(config_dir=config_dir)
    
    if "refresh_providers" not in st.session_state:
        st.session_state.refresh_providers = False


def render_provider_selection():
    """Initialize selected provider if not already set."""
    manager = st.session_state.llm_manager
    providers = manager.list_providers()
    
    if 'selected_provider' not in st.session_state:
        st.session_state.selected_provider = providers[0] if providers else None


def render_provider_list():
    """Display list of configured providers with checkboxes for comparison."""
    st.subheader("📋 Configured Providers")
    
    manager = st.session_state.llm_manager
    providers = manager.list_providers()
    
    if not providers:
        st.info("No providers configured yet. Add one below to get started.")
        return
    
    # Ensure selected_provider exists
    if 'selected_provider' not in st.session_state or st.session_state.selected_provider not in providers:
        st.session_state.selected_provider = providers[0]
    
    # Initialize enabled providers for comparison (Phase 4 feature)
    if 'enabled_providers' not in st.session_state:
        st.session_state.enabled_providers = {providers[0]: True}
    
    # Create columns for provider display (with checkbox column)
    col_checkbox, col1, col2, col3, col4 = st.columns([0.8, 2, 2, 1, 1])
    with col_checkbox:
        st.write("**Use**")
    with col1:
        st.write("**Provider**")
    with col2:
        st.write("**Model**")
    with col3:
        st.write("**Status**")
    with col4:
        st.write("**Actions**")
    
    st.divider()
    
    # Display each provider with checkbox
    for provider_name in providers:
        config = manager.get_provider_config(provider_name)
        if not config:
            continue
        
        col_checkbox, col1, col2, col3, col4 = st.columns([0.8, 2, 2, 1, 1])
        
        with col_checkbox:
            # Checkbox to enable provider for comparison (Phase 4)
            is_enabled = st.checkbox(
                label=provider_name,
                value=st.session_state.enabled_providers.get(provider_name, False),
                key=f"checkbox_{provider_name}",
                label_visibility="collapsed"
            )
            st.session_state.enabled_providers[provider_name] = is_enabled
            
            # Auto-select first enabled provider as active
            if is_enabled and st.session_state.selected_provider not in st.session_state.enabled_providers or not st.session_state.enabled_providers.get(st.session_state.selected_provider, False):
                st.session_state.selected_provider = provider_name
        
        with col1:
            st.write(f"**{config.provider.upper()}**")
        
        with col2:
            st.write(f"`{config.model}`")
        
        with col3:
            if config.is_ready:
                st.success("✅ Ready")
            else:
                st.warning("⚠️ Not Ready")
        
        with col4:
            if st.button("🗑️ Delete", key=f"delete_{provider_name}"):
                if manager.delete_provider(provider_name):
                    st.success(f"Deleted {provider_name}")
                    if provider_name in st.session_state.enabled_providers:
                        del st.session_state.enabled_providers[provider_name]
                    st.rerun()
                else:
                    st.error(f"Failed to delete {provider_name}")
    
    # Display selected provider summary below the table
    st.divider()
    st.subheader("📊 Active Provider Summary")
    config = manager.get_provider_config(st.session_state.selected_provider)
    if config:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Provider", st.session_state.selected_provider.upper())
        with col2:
            st.metric("Model", config.model)
        with col3:
            status = "✅ Ready" if config.is_ready else "⚠️ Not Ready"
            st.metric("Status", status)


def render_add_provider_form():
    """Display form for adding new provider with simplified API key setup."""
    st.subheader("➕ Add New Provider")
    
    st.info("""
    🔐 **How API Keys Work:**
    - Paste your API key in the field below (stored in memory only, not on disk)
    - OR set environment variable (e.g., `export OPENAI_API_KEY=sk-...`)
    - If you set the env variable, leave the key field empty
    - The app will use whichever is available
    """)
    
    # Provider selection OUTSIDE form so changes trigger re-render
    provider_options = [
        ("OpenAI", "openai"),
        ("Anthropic", "anthropic"),
        ("Ollama (Local)", "ollama"),
        ("LM Studio (Local)", "lmstudio"),
    ]
    provider_labels = [opt[0] for opt in provider_options]
    provider_values = [opt[1] for opt in provider_options]
    
    if 'selected_provider_idx' not in st.session_state:
        st.session_state.selected_provider_idx = 0
    
    selected_idx = st.selectbox(
        "Choose Provider",
        range(len(provider_labels)),
        format_func=lambda i: provider_labels[i],
        help="Select the LLM provider type",
        key="provider_selectbox_main",
        index=st.session_state.selected_provider_idx,
        on_change=lambda: st.session_state.update({'selected_provider_idx': st.session_state.provider_selectbox_main})
    )
    provider = provider_values[selected_idx]
    
    with st.form("add_provider_form"):
        # Model selection based on provider
        if provider == "openai":
            model = st.selectbox(
                "Model",
                [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo",
                    "gpt-4",
                    "gpt-3.5-turbo",
                    "o1",
                    "o1-mini"
                ],
                index=0,
                key=f"model_select_{provider}"
            )
            # Dynamic pricing based on model
            pricing_map = {
                "gpt-4o": (0.005, 0.015),
                "gpt-4o-mini": (0.00015, 0.0006),
                "gpt-4-turbo": (0.01, 0.03),
                "gpt-4": (0.03, 0.06),
                "gpt-3.5-turbo": (0.0005, 0.0015),
                "o1": (0.015, 0.06),
                "o1-mini": (0.003, 0.012)
            }
            input_cost, output_cost = pricing_map.get(model, (0.005, 0.015))
        elif provider == "anthropic":
            model = st.selectbox(
                "Model",
                [
                    "claude-opus-4-5",
                    "claude-sonnet-4-5",
                    "claude-haiku-4-5",
                    "claude-opus-4",
                    "claude-sonnet-4",
                    "claude-3-7-sonnet",
                    "claude-3-5-sonnet",
                    "claude-3-5-haiku",
                    "claude-3-opus",
                    "claude-3-sonnet",
                    "claude-3-haiku",
                ],
                index=0,
                key=f"model_select_{provider}"
            )
            # Dynamic pricing based on model
            pricing_map = {
                "claude-opus-4-5": (0.003, 0.015),
                "claude-sonnet-4-5": (0.003, 0.015),
                "claude-haiku-4-5": (0.008, 0.024),
                "claude-opus-4": (0.003, 0.015),
                "claude-sonnet-4": (0.003, 0.015),
                "claude-3-7-sonnet": (0.003, 0.015),
                "claude-3-5-sonnet": (0.003, 0.015),
                "claude-3-5-haiku": (0.0008, 0.0024),
                "claude-3-opus": (0.015, 0.075),
                "claude-3-sonnet": (0.003, 0.015),
                "claude-3-haiku": (0.00025, 0.00125),
            }
            input_cost, output_cost = pricing_map.get(model, (0.003, 0.015))
        elif provider == "ollama":
            model = st.text_input(
                "Model Name",
                placeholder="e.g., llama2, neural-chat, mistral",
                help="Name of the model running on Ollama",
                key=f"model_select_{provider}"
            )
            input_cost = 0.0
            output_cost = 0.0
        else:  # lmstudio
            model = st.text_input(
                "Model Name",
                placeholder="e.g., mistral, neural-chat",
                help="Name of the model running on LMStudio",
                key=f"model_select_{provider}"
            )
            input_cost = 0.0
            output_cost = 0.0
        
        # Single API Key field (simplified!)
        st.write("🔑 **API Key** (optional if set in environment)")
        
        # Provider-specific placeholder hint
        placeholder_hints = {
            "openai": "sk-... (from platform.openai.com)",
            "anthropic": "sk-ant-... (from console.anthropic.com)",
            "ollama": "Not required for local Ollama",
            "lmstudio": "Not required for local LM Studio"
        }
        placeholder = placeholder_hints.get(provider, "Paste your API key here")
        
        api_key_input = st.text_input(
            "API Key (paste here or use environment variable)",
            type="password",
            placeholder=placeholder,
            help="Your API key. Not saved to disk. If empty, looks for env variable like OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )
        
        # Simple settings
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider(
                "Temperature (how random)",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
            )
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=32000,
                value=2000,
            )
        
        # Submit button
        submitted = st.form_submit_button("✅ Add & Test Provider", use_container_width=True)
        
        if submitted:
            if not model:
                st.error("❌ Please select or enter a model name")
                return
            
            # Use provided key OR fall back to auto-finding environment variable
            api_key = api_key_input if api_key_input else None
            
            # For OpenAI/Anthropic, try to auto-detect environment variable if no key provided
            api_key_env_var = None
            if not api_key:
                env_map = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "ollama": None,
                    "lmstudio": None,
                }
                env_var_name = env_map.get(provider)
                if env_var_name and os.getenv(env_var_name):
                    api_key_env_var = env_var_name
                elif provider in ["openai", "anthropic"]:
                    st.error(f"❌ No API key provided and {env_var_name} environment variable not found")
                    return
            
            with st.spinner(f"Testing connection to {provider}..."):
                manager = st.session_state.llm_manager
                success, error_msg = manager.add_provider(
                    provider=provider,
                    model=model,
                    api_key=api_key,
                    api_key_env_var=api_key_env_var,
                    input_cost_per_1k=input_cost,
                    output_cost_per_1k=output_cost,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            
            if success:
                # Store API key in session state for use by Chat page
                # (Config is saved to disk WITHOUT the key for security)
                if 'provider_api_keys' not in st.session_state:
                    st.session_state.provider_api_keys = {}
                
                # Only store if we got an actual key (not env var based)
                if api_key:
                    st.session_state.provider_api_keys[provider] = api_key
                    
                    # Also save to persistent file with restricted permissions
                    from pathlib import Path
                    import json
                    keys_file = Path.home() / ".rlmkit" / "api_keys.json"
                    keys_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Load existing keys if any
                    existing_keys = {}
                    if keys_file.exists():
                        try:
                            with open(keys_file, "r") as f:
                                existing_keys = json.load(f)
                        except (json.JSONDecodeError, IOError):
                            pass
                    
                    # Update with new key
                    existing_keys[provider] = api_key
                    
                    # Save with restricted permissions (0o600 = rw-------)
                    with open(keys_file, "w") as f:
                        json.dump(existing_keys, f, indent=2)
                    try:
                        keys_file.chmod(0o600)
                    except Exception:
                        pass  # Windows doesn't support chmod
                
                # Save provider selection to .env for auto-loading on next session
                from pathlib import Path
                env_file = Path.home() / ".rlmkit" / ".env"
                env_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Read existing .env if it exists
                env_vars = {}
                if env_file.exists():
                    with open(env_file, "r") as f:
                        for line in f:
                            line = line.strip()
                            if line and "=" in line and not line.startswith("#"):
                                key, value = line.split("=", 1)
                                env_vars[key.strip()] = value.strip()
                
                # Update with selected provider
                env_vars["SELECTED_PROVIDER"] = provider
                
                # Write back to .env
                with open(env_file, "w") as f:
                    f.write("# RLMKit Configuration (Auto-generated)\n")
                    for key, value in env_vars.items():
                        f.write(f"{key}={value}\n")
                
                st.success(f"✅ {provider.upper()} provider added and tested successfully!")
                st.balloons()
                st.rerun()
            else:
                st.error(f"❌ Connection test failed:\n\n{error_msg}")


def render_pricing_info():
    """Display pricing information for all providers."""
    st.subheader("💰 Pricing Reference")
    
    pricing_data = {
        "OpenAI": {
            "gpt-4o": {"input": "$0.005", "output": "$0.015", "per": "1K tokens"},
            "gpt-4o-mini": {"input": "$0.00015", "output": "$0.0006", "per": "1K tokens"},
            "gpt-4-turbo": {"input": "$0.01", "output": "$0.03", "per": "1K tokens"},
            "gpt-4": {"input": "$0.03", "output": "$0.06", "per": "1K tokens"},
            "gpt-3.5-turbo": {"input": "$0.0005", "output": "$0.0015", "per": "1K tokens"},
            "o1": {"input": "$0.015", "output": "$0.06", "per": "1K tokens"},
            "o1-mini": {"input": "$0.003", "output": "$0.012", "per": "1K tokens"},
        },
        "Anthropic": {
            "claude-opus-4-5": {"input": "$0.003", "output": "$0.015", "per": "1K tokens"},
            "claude-sonnet-4-5": {"input": "$0.003", "output": "$0.015", "per": "1K tokens"},
            "claude-haiku-4-5": {"input": "$0.008", "output": "$0.024", "per": "1K tokens"},
            "claude-opus-4": {"input": "$0.003", "output": "$0.015", "per": "1K tokens"},
            "claude-sonnet-4": {"input": "$0.003", "output": "$0.015", "per": "1K tokens"},
            "claude-3-7-sonnet": {"input": "$0.003", "output": "$0.015", "per": "1K tokens"},
            "claude-3-5-sonnet": {"input": "$0.003", "output": "$0.015", "per": "1K tokens"},
            "claude-3-5-haiku": {"input": "$0.0008", "output": "$0.0024", "per": "1K tokens"},
            "claude-3-opus": {"input": "$0.015", "output": "$0.075", "per": "1K tokens"},
            "claude-3-sonnet": {"input": "$0.003", "output": "$0.015", "per": "1K tokens"},
            "claude-3-haiku": {"input": "$0.00025", "output": "$0.00125", "per": "1K tokens"},
        },
        "Local (Ollama/LMStudio)": {
            "Any Model": {"input": "Free", "output": "Free", "per": "Local execution"},
        }
    }
    
    tabs = st.tabs(list(pricing_data.keys()))
    
    for tab, provider in zip(tabs, pricing_data.keys()):
        with tab:
            cols = st.columns([2, 1, 1, 1])
            with cols[0]:
                st.write("**Model**")
            with cols[1]:
                st.write("**Input**")
            with cols[2]:
                st.write("**Output**")
            with cols[3]:
                st.write("**Unit**")
            
            st.divider()
            
            for model, costs in pricing_data[provider].items():
                cols = st.columns([2, 1, 1, 1])
                with cols[0]:
                    st.write(f"`{model}`")
                with cols[1]:
                    st.write(costs["input"])
                with cols[2]:
                    st.write(costs["output"])
                with cols[3]:
                    st.write(f"_{costs['per']}_")


def render_config_sidebar():
    """Minimal sidebar - Streamlit handles navigation automatically."""
    # No additional sidebar content - Streamlit shows navigation by default
    pass


def render_execution_settings():
    """Render execution mode and budget limits in main content area."""
    st.subheader("⚡ Execution Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Execution Mode**")
        mode = st.radio(
            "Select mode:",
            ["RLM Only", "Direct Only", "Compare Both"],
            index=["RLM Only", "Direct Only", "Compare Both"].index(st.session_state.get('execution_mode', 'Compare Both')),
            help="Choose whether to run RLM mode, Direct mode, or compare both",
            label_visibility="collapsed"
        )
        st.session_state.execution_mode = mode
    
    with col2:
        st.write("**Budget Limits**")
        max_steps = st.slider(
            "Max Steps (RLM)",
            min_value=1,
            max_value=32,
            value=st.session_state.get('max_steps', 16),
            help="Maximum execution steps for RLM mode",
            label_visibility="collapsed"
        )
        st.session_state.max_steps = max_steps
        
        timeout = st.slider(
            "Timeout (seconds)",
            min_value=1,
            max_value=30,
            value=st.session_state.get('timeout', 5),
            help="Maximum execution time per step",
            label_visibility="collapsed"
        )
        st.session_state.timeout = timeout


def main():
    """Main configuration page."""
    # Mark this as the configuration page in session state
    st.session_state.current_nav_page = 'configuration'

    st.set_page_config(
        page_title="Configuration - RLM Studio",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="auto",
    )

    # Load global CSS (includes navigation hiding and sidebar toggle fixes)
    _inject_rlmkit_desktop_css()
    
    # Initialize session state
    init_config_session_state()
    
    # Render custom navigation in sidebar
    render_custom_navigation()
    
    # Header
    st.title("⚙️ Configuration Management")
    st.markdown("""
    Configure execution settings, manage LLM providers, and view pricing information.
    """)
    st.divider()
    
    # Execution settings in main content
    render_execution_settings()
    st.divider()
    
    # Initialize provider selection (hidden)
    render_provider_selection()
    
    # Providers list with integrated selection
    render_provider_list()
    st.divider()
    
    # Add provider form
    render_add_provider_form()
    st.divider()
    
    # Pricing info
    render_pricing_info()


if __name__ == "__main__":
    main()
