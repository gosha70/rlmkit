# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Configuration Management Page - Streamlit UI for managing LLM providers."""

import streamlit as st
from pathlib import Path
import os

from rlmkit.ui.services import LLMConfigManager
from rlmkit.ui.services.models import RAGConfig, ExecutionSlot, ExecutionPlan
from rlmkit.ui.services.secret_store import (
    get_secret_store, resolve_api_key, KEY_POLICY_OPTIONS,
)
from rlmkit.ui.services.profile_store import (
    RunProfile, RunProfileStore,
    SYSTEM_PROMPT_TEMPLATE_NAMES, SYSTEM_PROMPT_TEMPLATES,
    load_custom_prompts, save_custom_prompts,
)
from rlmkit.ui.components.navigation import render_custom_navigation
from rlmkit.ui.components.session_summary import render_session_summary
from rlmkit.ui.app import _inject_rlmkit_desktop_css
from rlmkit.ui.data.providers_catalog import (
    PROVIDERS, PROVIDER_OPTIONS, PROVIDERS_BY_KEY,
    get_provider, get_model_pricing, get_model_names, get_env_var,
    get_embedding_model_names, build_pricing_table,
)


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
    """Display configured providers as cards with Test / Set default / Delete actions."""
    st.subheader("Configured Providers")

    manager = st.session_state.llm_manager
    providers = manager.list_providers()

    if not providers:
        st.info("No providers configured yet. Add one below to get started.")
        return

    # Ensure selected_provider exists
    if "selected_provider" not in st.session_state or st.session_state.selected_provider not in providers:
        st.session_state.selected_provider = providers[0]

    for provider_name in providers:
        config = manager.get_provider_config(provider_name)
        if not config:
            continue

        entry = get_provider(config.provider)
        display_name = entry.display_name if entry else config.provider.upper()
        is_default = st.session_state.selected_provider == provider_name

        with st.container(border=True):
            # Header row: badge + model + status
            col_info, col_actions = st.columns([3, 2])

            with col_info:
                badge = "**[Default]** " if is_default else ""
                status_icon = "Ready" if config.is_ready else "Not tested"
                st.markdown(
                    f"### {badge}{display_name}  \n"
                    f"`{config.model}` — {status_icon}"
                )

            with col_actions:
                btn_cols = st.columns(3)
                with btn_cols[0]:
                    if st.button("Test", key=f"test_{provider_name}", use_container_width=True):
                        success, msg = manager.test_connection(
                            config.provider, config.model,
                            api_key=config.api_key,
                            api_key_env_var=config.api_key_env_var,
                        )
                        if success:
                            config.test_successful = True
                            manager._save_config(config)
                            st.success("Connection OK")
                        else:
                            st.error(msg)
                with btn_cols[1]:
                    if not is_default:
                        if st.button("Set default", key=f"default_{provider_name}", use_container_width=True):
                            st.session_state.selected_provider = provider_name
                            st.rerun()
                with btn_cols[2]:
                    if st.button("Delete", key=f"delete_{provider_name}", use_container_width=True):
                        manager.delete_provider(provider_name)
                        st.rerun()


def render_add_provider_form():
    """Display form for adding new provider with simplified API key setup."""
    st.subheader("➕ Add New Provider")

    # Key persistence policy selector
    if "key_policy" not in st.session_state:
        st.session_state.key_policy = "file"  # backward-compatible default

    policy_labels = [label for _, label in KEY_POLICY_OPTIONS]
    policy_keys = [k for k, _ in KEY_POLICY_OPTIONS]
    current_idx = policy_keys.index(st.session_state.key_policy) if st.session_state.key_policy in policy_keys else 0

    selected_policy_idx = st.selectbox(
        "🔐 Key Storage Policy",
        range(len(policy_labels)),
        format_func=lambda i: policy_labels[i],
        index=current_idx,
        help="Where API keys are persisted between sessions.",
        key="key_policy_selector",
    )
    st.session_state.key_policy = policy_keys[selected_policy_idx]

    if st.session_state.key_policy == "file":
        st.caption("⚠️ Keys stored unencrypted at `~/.rlmkit/api_keys.json` (chmod 600).")
    elif st.session_state.key_policy == "env":
        st.caption("Keys are read from environment variables only — nothing is written to disk.")

    # Provider selection OUTSIDE form so changes trigger re-render
    provider_labels = [opt[0] for opt in PROVIDER_OPTIONS]
    provider_values = [opt[1] for opt in PROVIDER_OPTIONS]

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
    provider_key = provider_values[selected_idx]
    catalog_entry = get_provider(provider_key)

    with st.form("add_provider_form"):
        # Model selection based on provider — driven by catalog
        model_names = get_model_names(provider_key)
        if model_names:
            model = st.selectbox(
                "Model",
                model_names,
                index=0,
                key=f"model_select_{provider_key}",
            )
            input_cost, output_cost = get_model_pricing(provider_key, model)
        else:
            # Local providers: free-text model input
            model = st.text_input(
                "Model Name",
                placeholder=catalog_entry.model_input_hint if catalog_entry else "",
                help=f"Name of the model running on {catalog_entry.display_name if catalog_entry else provider_key}",
                key=f"model_select_{provider_key}",
            )
            input_cost = 0.0
            output_cost = 0.0

        # Single API Key field (simplified!)
        st.write("🔑 **API Key** (optional if set in environment)")

        placeholder = catalog_entry.api_key_placeholder if catalog_entry else "Paste your API key here"

        api_key_input = st.text_input(
            "API Key (paste here or use environment variable)",
            type="password",
            placeholder=placeholder,
            help="Your API key. Not saved to disk. If empty, looks for env variable like OPENAI_API_KEY or ANTHROPIC_API_KEY"
        )

        # Submit button
        submitted = st.form_submit_button("✅ Add & Test Provider", use_container_width=True)

        if submitted:
            if not model:
                st.error("❌ Please select or enter a model name")
                return

            # Use provided key OR fall back to auto-finding environment variable
            api_key = api_key_input if api_key_input else None

            # Try to auto-detect environment variable if no key provided
            api_key_env_var = None
            if not api_key:
                env_var_name = get_env_var(provider_key)
                if env_var_name and os.getenv(env_var_name):
                    api_key_env_var = env_var_name
                elif catalog_entry and catalog_entry.requires_api_key:
                    st.error(f"❌ No API key provided and {env_var_name} environment variable not found")
                    return

            with st.spinner(f"Testing connection to {provider_key}..."):
                manager = st.session_state.llm_manager
                success, error_msg = manager.add_provider(
                    provider=provider_key,
                    model=model,
                    api_key=api_key,
                    api_key_env_var=api_key_env_var,
                    input_cost_per_1k=input_cost,
                    output_cost_per_1k=output_cost,
                )

            if success:
                # Persist key via the user's chosen SecretStore
                if api_key:
                    store = get_secret_store(st.session_state)
                    store.set(provider_key, api_key)

                    # Also keep in session-state cache for current session
                    if "provider_api_keys" not in st.session_state:
                        st.session_state.provider_api_keys = {}
                    st.session_state.provider_api_keys[provider_key] = api_key

                st.success(f"✅ {provider_key.upper()} provider added and tested successfully!")
                st.balloons()
                st.rerun()
            else:
                st.error(f"❌ Connection test failed:\n\n{error_msg}")


def render_pricing_info():
    """Display pricing information inside a collapsible expander."""
    with st.expander("Pricing reference (optional)", expanded=False):
        st.caption("Prices may drift; update in code/data if needed.")
        pricing_data = build_pricing_table()

        tabs = st.tabs(list(pricing_data.keys()))

        for tab, provider_display in zip(tabs, pricing_data.keys()):
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

                for model, costs in pricing_data[provider_display].items():
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


def _apply_profile(profile: RunProfile) -> None:
    """Apply a RunProfile's values to session state."""
    st.session_state.default_temperature = profile.temperature
    st.session_state.default_top_p = profile.top_p
    st.session_state.default_max_tokens = profile.max_output_tokens
    st.session_state.max_steps = profile.max_steps
    st.session_state.rlm_timeout = profile.rlm_timeout_seconds
    # Set selected_strategies from profile.strategy
    st.session_state.selected_strategies = [profile.strategy]
    if profile.default_provider:
        st.session_state.selected_provider = profile.default_provider
    st.session_state.rag_config = RAGConfig(
        chunk_size=profile.rag_chunk_size,
        chunk_overlap=profile.rag_chunk_overlap,
        top_k=profile.rag_top_k,
        embedding_model=profile.rag_embedding_model,
    )
    # System prompt
    st.session_state.system_prompt_mode = profile.system_prompt_mode
    st.session_state.system_prompt_template = profile.system_prompt_template
    st.session_state.system_prompt_custom = profile.system_prompt_custom


def _build_profile_from_session(name: str) -> RunProfile:
    """Capture current session settings into a RunProfile."""
    rc = st.session_state.get("rag_config") or RAGConfig()
    # Get primary strategy from selected_strategies (first one)
    strategies = st.session_state.get("selected_strategies", ["direct"])
    primary_strategy = strategies[0] if strategies else "direct"
    return RunProfile(
        name=name,
        run_mode="compare" if len(strategies) > 1 else "single",
        strategy=primary_strategy,
        default_provider=st.session_state.get("selected_provider"),
        temperature=st.session_state.get("default_temperature", 0.7),
        top_p=st.session_state.get("default_top_p", 1.0),
        max_output_tokens=st.session_state.get("default_max_tokens", 2000),
        max_steps=st.session_state.get("max_steps", 16),
        rlm_timeout_seconds=st.session_state.get("rlm_timeout", 60),
        rag_top_k=rc.top_k,
        rag_chunk_size=rc.chunk_size,
        rag_chunk_overlap=rc.chunk_overlap,
        rag_embedding_model=rc.embedding_model,
        system_prompt_mode=st.session_state.get("system_prompt_mode", "default"),
        system_prompt_template=st.session_state.get("system_prompt_template", "Default"),
        system_prompt_custom=st.session_state.get("system_prompt_custom"),
        key_policy=st.session_state.get("key_policy"),
    )


def render_execution_settings():
    """Render execution mode and budget limits in main content area."""
    st.subheader("Run Settings")

    manager = st.session_state.llm_manager
    providers = manager.list_providers()

    # --- Profile selector ---
    profile_store = RunProfileStore()
    all_profiles = profile_store.list_profiles()
    profile_names = ["(Custom)"] + [p.name for p in all_profiles]

    col_profile, col_save = st.columns([3, 1])
    with col_profile:
        active_name = st.session_state.get("active_profile", "(Custom)")
        if active_name not in profile_names:
            active_name = "(Custom)"
        chosen = st.selectbox(
            "Active Profile",
            profile_names,
            index=profile_names.index(active_name),
            key="profile_selector",
        )
        if chosen != st.session_state.get("active_profile"):
            st.session_state.active_profile = chosen
            profile = profile_store.get_profile(chosen)
            if profile:
                _apply_profile(profile)
                st.rerun()

    with col_save:
        st.write("")  # spacer
        with st.popover("Save current as profile"):
            new_name = st.text_input("Profile name", key="save_profile_name")
            if st.button("Save", key="save_profile_btn") and new_name:
                p = _build_profile_from_session(new_name)
                profile_store.save_profile(p)
                st.session_state.active_profile = new_name
                st.success(f"Saved '{new_name}'")
                st.rerun()

    st.divider()

    # --- Mode Selection: Simple checklist ---
    # Initialize selected strategies from session state
    if "selected_strategies" not in st.session_state:
        # Default based on existing execution_mode
        mode = st.session_state.get("execution_mode", "Direct Only")
        if mode == "RLM Only":
            st.session_state.selected_strategies = ["rlm"]
        elif mode == "RAG Only":
            st.session_state.selected_strategies = ["rag"]
        elif mode == "Compare Both":
            st.session_state.selected_strategies = ["rlm", "direct"]
        else:
            st.session_state.selected_strategies = ["direct"]

    # Check if OpenAI provider is configured (required for RAG embeddings)
    openai_configured = any(
        manager.get_provider_config(p).provider == "openai"
        for p in providers
        if manager.get_provider_config(p)
    )

    col_strategies, col_provider = st.columns(2)

    with col_strategies:
        st.write("**Strategies**")
        rlm_checked = st.checkbox("RLM", value="rlm" in st.session_state.selected_strategies, key="strat_rlm")
        direct_checked = st.checkbox("Direct", value="direct" in st.session_state.selected_strategies, key="strat_direct")

        # RAG requires OpenAI for embeddings
        if openai_configured:
            rag_checked = st.checkbox("RAG", value="rag" in st.session_state.selected_strategies, key="strat_rag")
        else:
            rag_checked = st.checkbox("RAG", value=False, disabled=True, key="strat_rag")
            st.caption("⚠️ To enable RAG, configure OpenAI first")

        # Build selected list
        selected_strategies = []
        if rlm_checked:
            selected_strategies.append("rlm")
        if direct_checked:
            selected_strategies.append("direct")
        if rag_checked and openai_configured:
            selected_strategies.append("rag")

        # Require at least one selection
        if not selected_strategies:
            st.warning("Select at least one strategy.")
            selected_strategies = ["direct"]  # fallback

        st.session_state.selected_strategies = selected_strategies

        # Show mode indicator
        if len(selected_strategies) == 1:
            st.caption("Single Mode")
        else:
            st.caption(f"Multi Mode ({len(selected_strategies)} strategies)")

    with col_provider:
        if providers:
            selected = st.selectbox(
                "Provider",
                providers,
                index=providers.index(st.session_state.selected_provider) if st.session_state.get("selected_provider") in providers else 0,
                format_func=lambda p: f"{p.upper()} ({manager.get_provider_config(p).model})" if manager.get_provider_config(p) else p,
                key="mode_provider",
            )
            st.session_state.selected_provider = selected
        else:
            st.warning("No providers configured. Add one in the Connections tab.")

    # Build ExecutionPlan from selected strategies
    rag_cfg = st.session_state.get("rag_config") or RAGConfig()
    provider_name = st.session_state.get("selected_provider", "")
    cfg = manager.get_provider_config(provider_name) if provider_name else None
    model_name = cfg.model if cfg else provider_name or "?"

    if len(selected_strategies) == 1:
        # Single mode - no ExecutionPlan needed (legacy behavior)
        st.session_state.execution_plan = None
        mode_map = {"rlm": "RLM Only", "direct": "Direct Only", "rag": "RAG Only"}
        st.session_state.execution_mode = mode_map.get(selected_strategies[0], "Direct Only")
    else:
        # Multi mode - build ExecutionPlan
        plan_slots = []
        for mode in selected_strategies:
            mode_label = {"rlm": "RLM", "direct": "Direct", "rag": "RAG"}[mode]
            plan_slots.append(ExecutionSlot(
                mode=mode,
                provider_name=provider_name,
                label=f"{mode_label} ({model_name})",
                rag_config=rag_cfg if mode == "rag" else None,
            ))
        st.session_state.execution_plan = ExecutionPlan(slots=plan_slots)

    # --- Part B: Strategy-specific settings ---
    st.divider()

    # Active strategies for showing relevant settings
    active_strategies = set(selected_strategies)

    # -- Direct settings (sampling) --
    if "direct" in active_strategies or "rlm" in active_strategies:
        with st.container(border=True):
            st.write("**Sampling Settings**")
            st.caption("Applies to Direct calls and each RLM step.")
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0, max_value=2.0,
                    value=st.session_state.get("default_temperature", 0.7),
                    step=0.1,
                    key="run_temperature",
                )
                st.session_state.default_temperature = temperature
            with col2:
                max_output_tokens = st.number_input(
                    "Max Output Tokens",
                    min_value=100, max_value=32000,
                    value=st.session_state.get("default_max_tokens", 2000),
                    key="run_max_tokens",
                )
                st.session_state.default_max_tokens = max_output_tokens

            with st.expander("Advanced sampling", expanded=False):
                top_p = st.slider(
                    "Top-p",
                    min_value=0.0, max_value=1.0,
                    value=st.session_state.get("default_top_p", 1.0),
                    step=0.05,
                    key="run_top_p",
                )
                st.session_state.default_top_p = top_p

                timeout_direct = st.slider(
                    "Timeout (seconds)",
                    min_value=1, max_value=120,
                    value=st.session_state.get("timeout", 30),
                    key="run_timeout",
                )
                st.session_state.timeout = timeout_direct

    # -- RLM settings --
    if "rlm" in active_strategies:
        with st.container(border=True):
            st.write("**RLM Settings**")
            col1, col2 = st.columns(2)
            with col1:
                max_steps = st.slider(
                    "Max Steps",
                    min_value=1, max_value=32,
                    value=st.session_state.get("max_steps", 16),
                    help="Maximum exploration steps for RLM",
                    key="run_max_steps",
                )
                st.session_state.max_steps = max_steps
            with col2:
                rlm_timeout = st.slider(
                    "RLM Timeout (seconds)",
                    min_value=5, max_value=300,
                    value=st.session_state.get("rlm_timeout", 60),
                    help="Total time budget for the RLM run",
                    key="run_rlm_timeout",
                )
                st.session_state.rlm_timeout = rlm_timeout

    # -- RAG settings --
    if "rag" in active_strategies:
        with st.container(border=True):
            st.write("**RAG Settings**")
            _render_rag_settings_inline()

    # -- System Prompt Settings --
    with st.expander("System Prompt Settings", expanded=False):
        _render_system_prompt_settings()


def _render_system_prompt_settings():
    """Render system prompt configuration controls."""
    # Fix text area resize handle clipping
    st.markdown("""
        <style>
        .stTextArea textarea {
            resize: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.write("**System Prompt**")

    # Build options: template names + "Custom"
    options = SYSTEM_PROMPT_TEMPLATE_NAMES + ["Custom"]

    # Determine current selection
    if st.session_state.get("system_prompt_mode") == "custom":
        current_selection = "Custom"
    else:
        current_selection = st.session_state.get("system_prompt_template", "Default")
        if current_selection not in SYSTEM_PROMPT_TEMPLATE_NAMES:
            current_selection = "Default"

    idx = options.index(current_selection) if current_selection in options else 0

    selected = st.selectbox(
        "Template",
        options,
        index=idx,
        key="run_prompt_template_select",
    )

    # Update session state based on selection
    if selected == "Custom":
        st.session_state.system_prompt_mode = "custom"

        # Load current custom prompts (from session state or file)
        current_custom = st.session_state.get("system_prompt_custom")
        if current_custom is None or not isinstance(current_custom, dict):
            current_custom = load_custom_prompts()
            st.session_state.system_prompt_custom = current_custom

        st.caption("Provide custom system prompts for each execution mode.")

        # Three editable text areas with prominent titles
        st.markdown("**Direct**")
        direct_prompt = st.text_area(
            "Direct prompt",
            value=current_custom.get("direct", ""),
            height=100,
            key="run_prompt_custom_direct",
            placeholder="System prompt for Direct mode…",
            label_visibility="collapsed",
        )

        st.markdown("**RAG**")
        rag_prompt = st.text_area(
            "RAG prompt",
            value=current_custom.get("rag", ""),
            height=100,
            key="run_prompt_custom_rag",
            placeholder="System prompt for RAG mode…",
            label_visibility="collapsed",
        )

        st.markdown("**RLM**")
        rlm_prompt = st.text_area(
            "RLM prompt",
            value=current_custom.get("rlm", ""),
            height=100,
            key="run_prompt_custom_rlm",
            placeholder="System prompt for RLM mode…",
            label_visibility="collapsed",
        )

        # Update session state with current values
        st.session_state.system_prompt_custom = {
            "direct": direct_prompt,
            "rag": rag_prompt,
            "rlm": rlm_prompt,
        }

        # Save button
        if st.button("Save Custom Prompts", key="run_prompt_save", type="primary"):
            save_custom_prompts(st.session_state.system_prompt_custom)
            st.success("Custom prompts saved to ~/.rlmkit/system_prompt_custom.json")

    else:
        # Template mode
        st.session_state.system_prompt_mode = "default"
        st.session_state.system_prompt_template = selected

        tpl_info = SYSTEM_PROMPT_TEMPLATES.get(selected, {})
        if tpl_info.get("description"):
            st.caption(tpl_info["description"])

        # Three read-only text areas with prominent titles
        st.markdown("**Direct**")
        st.text_area(
            "Direct prompt",
            value=tpl_info.get("direct", "(built-in default)"),
            height=100,
            disabled=True,
            key=f"run_prompt_tpl_direct_{selected}",
            label_visibility="collapsed",
        )

        st.markdown("**RAG**")
        st.text_area(
            "RAG prompt",
            value=tpl_info.get("rag", "(built-in default)"),
            height=100,
            disabled=True,
            key=f"run_prompt_tpl_rag_{selected}",
            label_visibility="collapsed",
        )

        st.markdown("**RLM**")
        st.text_area(
            "RLM prompt",
            value=tpl_info.get("rlm", "(built-in default)"),
            height=100,
            disabled=True,
            key=f"run_prompt_tpl_rlm_{selected}",
            label_visibility="collapsed",
        )


def _render_rag_settings_inline():
    """Render RAG settings inline within Run Settings (no subheader)."""
    if "rag_config" not in st.session_state or st.session_state.rag_config is None:
        st.session_state.rag_config = RAGConfig()

    rc = st.session_state.rag_config

    embedding_models = get_embedding_model_names()
    if not embedding_models:
        embedding_models = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]

    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=4000,
            value=rc.chunk_size,
            step=100,
            help="Number of characters per document chunk",
        )
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=1000,
            value=rc.chunk_overlap,
            step=50,
            help="Overlap between consecutive chunks",
        )

    with col2:
        top_k = st.slider(
            "Top-K Chunks",
            min_value=1,
            max_value=20,
            value=rc.top_k,
            help="Number of most relevant chunks to retrieve",
        )
        embedding_model = st.selectbox(
            "Embedding Model",
            embedding_models,
            index=embedding_models.index(rc.embedding_model) if rc.embedding_model in embedding_models else 0,
            help="OpenAI embedding model for chunk similarity",
        )

    st.session_state.rag_config = RAGConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        embedding_model=embedding_model,
    )


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

    # Session metrics summary in sidebar
    with st.sidebar:
        render_session_summary()

    # Header
    st.title("Configuration")
    st.caption("Manage provider connections and configure how runs execute.")
    st.divider()

    # Initialize provider selection (hidden)
    render_provider_selection()

    # Two tabs per plan: Connections + Run Settings
    tab_connections, tab_run = st.tabs(["Connections", "Run Settings"])

    with tab_connections:
        render_provider_list()
        st.divider()
        render_add_provider_form()
        render_pricing_info()

    with tab_run:
        render_execution_settings()


if __name__ == "__main__":
    main()
