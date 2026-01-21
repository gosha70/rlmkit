"""ChatManager - Core message management and execution routing."""

from typing import Optional, List, Dict, Any
from uuid import uuid4
import asyncio

from .models import ChatMessage, ExecutionMetrics, ComparisonMetrics


class ChatManager:
    """
    Manage chat messages and execution flow.
    
    Responsibilities:
    - Maintain message history
    - Route execution to RLM or Direct based on mode
    - Collect metrics from both execution paths
    - Generate comparison insights
    """
    
    def __init__(self, session_state: Dict[str, Any] = None):
        """
        Initialize ChatManager.
        
        Args:
            session_state: Streamlit session state dict (usually st.session_state)
        """
        self.session_state = session_state or {}
        self._init_session()
    
    def _init_session(self) -> None:
        """Initialize session state if needed."""
        if "messages" not in self.session_state:
            self.session_state["messages"] = []
        if "conversation_id" not in self.session_state:
            self.session_state["conversation_id"] = str(uuid4())
    
    async def process_message(
        self,
        user_query: str,
        mode: str = "compare",
        file_context: Optional[str] = None,
        file_info: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """
        Process user message and generate response(s).
        
        Args:
            user_query: The user's question
            mode: "rlm_only", "direct_only", or "compare"
            file_context: Optional document text for context
            file_info: Metadata about the file (name, size, tokens)
        
        Returns:
            ChatMessage with response(s) and metrics
            
        Implementation notes:
        - Should execute RLM and/or Direct based on mode
        - Should collect metrics using MetricsCollector
        - Should handle errors gracefully
        - Should update session state with new message
        """
        message = ChatMessage(
            user_query=user_query,
            mode=mode,
            file_context=file_context,
            file_info=file_info,
        )
        
        # TODO: Execute based on mode
        # if mode in ("rlm_only", "compare"):
        #     rlm_result = await self._execute_rlm(user_query, file_context)
        #     message.rlm_response = rlm_result.response
        #     message.rlm_metrics = rlm_result.metrics
        #     message.rlm_trace = rlm_result.trace
        
        # if mode in ("direct_only", "compare"):
        #     direct_result = await self._execute_direct(user_query, file_context)
        #     message.direct_response = direct_result.response
        #     message.direct_metrics = direct_result.metrics
        
        # if mode == "compare":
        #     message.comparison_metrics = self._compare_metrics(
        #         message.rlm_metrics,
        #         message.direct_metrics
        #     )
        
        # Add to history
        self.session_state["messages"].append(message)
        
        return message
    
    async def _execute_rlm(
        self,
        user_query: str,
        file_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute RLM (with exploration steps).
        
        Returns dict with:
            - response: Response object
            - metrics: ExecutionMetrics
            - trace: List of step dicts
        
        Implementation notes:
        - Should use existing RLM controller from core
        - Should track execution trace
        - Should collect memory metrics
        """
        raise NotImplementedError("To be implemented")
    
    async def _execute_direct(
        self,
        user_query: str,
        file_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute Direct LLM (no exploration).
        
        Returns dict with:
            - response: Response object
            - metrics: ExecutionMetrics
        
        Implementation notes:
        - Should call LLM directly without RLM steps
        - Should still collect metrics
        """
        raise NotImplementedError("To be implemented")
    
    def _compare_metrics(
        self,
        rlm_metrics: ExecutionMetrics,
        direct_metrics: ExecutionMetrics,
    ) -> ComparisonMetrics:
        """
        Compare metrics from RLM and Direct execution.
        
        Returns ComparisonMetrics with savings and recommendation.
        
        Implementation notes:
        - Should calculate token/cost savings
        - Should determine recommendation based on trade-offs
        """
        raise NotImplementedError("To be implemented")
    
    def get_messages(self) -> List[ChatMessage]:
        """Get all messages in current conversation."""
        return self.session_state.get("messages", [])
    
    def get_message(self, message_id: str) -> Optional[ChatMessage]:
        """Get a specific message by ID."""
        for msg in self.get_messages():
            if msg.id == message_id:
                return msg
        return None
    
    def clear_history(self) -> None:
        """Clear all messages from this session."""
        self.session_state["messages"] = []
    
    def export_conversation(self, format: str = "json") -> str:
        """
        Export conversation to JSON, Markdown, or CSV.
        
        Args:
            format: "json", "markdown", or "csv"
        
        Returns:
            String representation of conversation
            
        Implementation notes:
        - Should handle all 3 formats
        - Should include all metrics
        - Should be human-readable (especially markdown)
        """
        raise NotImplementedError("To be implemented")
    
    @property
    def conversation_id(self) -> str:
        """Get current conversation ID."""
        return self.session_state.get("conversation_id", "")
