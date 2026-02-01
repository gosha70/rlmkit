"""Session state initialization for RLMKit Chat UI."""

from typing import Dict, Any
import streamlit as st
from uuid import uuid4
from datetime import datetime

from .services import ChatManager, MetricsCollector, MemoryMonitor, LLMConfigManager
from .services.models import SessionMetrics
from rlmkit.storage import Database, ConversationStore, VectorStore


def init_session_state() -> Dict[str, Any]:
    """
    Initialize Streamlit session state for chat UI.

    Returns:
        Session state dict
    """
    # Initialize if not already done
    if "initialized" not in st.session_state:
        st.session_state.initialized = True

        # Core managers
        st.session_state.chat_manager = ChatManager(st.session_state)
        st.session_state.metrics_collector = MetricsCollector()
        st.session_state.memory_monitor = MemoryMonitor()
        st.session_state.config_manager = LLMConfigManager()

        # Persistent storage
        db = Database()
        st.session_state.db = db
        st.session_state.conversation_store = ConversationStore(db)
        st.session_state.vector_store = VectorStore(db)

        # Conversation state
        st.session_state.conversation_id = str(uuid4())
        st.session_state.messages = []
        st.session_state.session_start_time = datetime.now()
        
        # UI state
        st.session_state.current_mode = "compare"
        st.session_state.show_settings = False
        st.session_state.show_metrics = True
        st.session_state.file_context = None
        st.session_state.file_info = None
        
        # Configuration
        st.session_state.active_provider = None
        st.session_state.active_model = None
        
        # Session metrics
        st.session_state.session_metrics = SessionMetrics()
    
    return st.session_state


def reset_session() -> None:
    """Reset session state to initial state."""
    st.session_state.clear()
    init_session_state()


def get_session_info() -> Dict[str, Any]:
    """Get current session information."""
    return {
        "conversation_id": st.session_state.conversation_id,
        "message_count": len(st.session_state.messages),
        "session_start": st.session_state.session_start_time,
        "active_provider": st.session_state.active_provider,
        "active_model": st.session_state.active_model,
        "current_mode": st.session_state.current_mode,
    }


__all__ = [
    "init_session_state",
    "reset_session",
    "get_session_info",
]
