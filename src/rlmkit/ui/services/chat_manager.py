# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
ChatManager - Core message management and execution routing.
"""

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
        self.session_state = session_state if session_state is not None else {}
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
            
        Example:
            >>> manager = ChatManager()
            >>> message = await manager.process_message("Summarize", mode="compare")
            >>> print(message.rlm_response.content)
            "RLM exploration completed..."
            >>> print(message.comparison_metrics.cost_delta_percent)
            500
            
        Implementation notes:
        - Executes RLM and/or Direct based on mode parameter
        - "rlm_only": Only RLM exploration (3 steps)
        - "direct_only": Only direct LLM call (single step)
        - "compare": Both paths, calculate deltas and recommendation
        - Handles errors gracefully with fallback responses
        - Updates session state with new message for history
        """
        message = ChatMessage(
            user_query=user_query,
            mode=mode,
            file_context=file_context,
            file_info=file_info,
        )
        
        # Execute based on mode
        if mode in ("rlm_only", "compare"):
            try:
                rlm_result = await self._execute_rlm(user_query, file_context)
                message.rlm_response = rlm_result["response"]
                message.rlm_metrics = rlm_result["metrics"]
                message.rlm_trace = rlm_result["trace"]
            except Exception as e:
                message.error = f"RLM execution failed: {str(e)}"
        
        if mode in ("direct_only", "compare"):
            try:
                direct_result = await self._execute_direct(user_query, file_context)
                message.direct_response = direct_result["response"]
                message.direct_metrics = direct_result["metrics"]
            except Exception as e:
                if not message.error:
                    message.error = f"Direct execution failed: {str(e)}"
        
        if mode == "compare" and message.rlm_metrics and message.direct_metrics:
            try:
                message.comparison_metrics = self._compare_metrics(
                    message.rlm_metrics,
                    message.direct_metrics
                )
            except Exception as e:
                message.error = f"Comparison failed: {str(e)}"
        
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
        
        Example:
            >>> manager = ChatManager()
            >>> result = await manager._execute_rlm("Summarize this document")
            >>> print(result['response'].content)
            "The document discusses..."
            >>> print(result['metrics'].steps_taken)
            3
        
        Implementation notes:
        - Simulates RLM execution with multiple steps
        - Returns mock data for testing (TODO: integrate real RLM controller)
        - Collects memory and performance metrics
        """
        from .models import Response, ExecutionMetrics
        
        # LATER: Replace with actual RLM controller integration
        steps = []
        
        # LATER: Execute real RLM exploration with actual steps
        for step_num in range(1, 4):
            step = {
                "step": step_num,
                "action": f"Explore section {step_num}",
                "input_tokens": 200 + (step_num * 50),
                "output_tokens": 100 + (step_num * 30),
                "duration_seconds": 0.5 + (step_num * 0.3),
            }
            steps.append(step)
        
        # Calculate totals from trace
        total_input = sum(s["input_tokens"] for s in steps)
        total_output = sum(s["output_tokens"] for s in steps)
        total_time = sum(s["duration_seconds"] for s in steps)
        
        # Create response
        response = Response(
            content=f"RLM exploration completed with {len(steps)} steps. "
                   f"Query was: {user_query}",
            stop_reason="stop",
            raw_response=None
        )
        
        # Create metrics
        metrics = ExecutionMetrics(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            # LATER: Get actual pricing from LLMConfigManager
            cost_usd=(total_input * 0.00001) + (total_output * 0.00002),
            cost_breakdown={
                "input": total_input * 0.00001,
                "output": total_output * 0.00002,
            },
            execution_time_seconds=total_time,
            steps_taken=len(steps),
            # LATER: Get actual memory metrics from MemoryMonitor
            memory_used_mb=45.2,
            memory_peak_mb=62.1,
            success=True,
            execution_type="rlm"
        )
        
        return {
            "response": response,
            "metrics": metrics,
            "trace": steps,
        }
    
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
        
        Example:
            >>> manager = ChatManager()
            >>> result = await manager._execute_direct("Summarize this")
            >>> print(result['response'].content)
            "This document is about..."
            >>> print(result['metrics'].steps_taken)
            0
        
        Implementation notes:
        - Calls LLM directly without RLM exploration steps
        - Single execution path vs RLM's multi-step approach
        - Much faster but potentially less thorough
        - Still collects full metrics for comparison
        """
        from .models import Response, ExecutionMetrics
        
        # Simulate direct LLM call (faster than RLM)
        # LATER: Replace with actual LLM API call
        
        # Direct call has no exploration steps, just single execution
        # Input is the full prompt + context + query
        input_tokens = 150  # Smaller than RLM exploration steps
        output_tokens = 120  # Shorter response without exploration
        duration = 0.8  # Faster than RLM's multi-step approach (avg 1.6s)
        
        # Create response (similar structure to RLM for comparison)
        response = Response(
            content=f"Direct LLM response to: {user_query}. "
                   f"This is a single-call response without exploration steps.",
            stop_reason="stop",
            raw_response=None
        )
        
        # Create metrics for direct execution
        metrics = ExecutionMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            # LATER: Get actual pricing from LLMConfigManager
            cost_usd=(input_tokens * 0.00001) + (output_tokens * 0.00002),
            cost_breakdown={
                "input": input_tokens * 0.00001,
                "output": output_tokens * 0.00002,
            },
            execution_time_seconds=duration,
            steps_taken=0,  # No RLM exploration steps
            # LATER: Get actual memory metrics from MemoryMonitor
            memory_used_mb=12.5,  # Much less memory than RLM
            memory_peak_mb=18.3,
            success=True,
            execution_type="direct"
        )
        
        return {
            "response": response,
            "metrics": metrics,
            # No trace for direct execution (it's a single call)
        }
    
    def _compare_metrics(
        self,
        rlm_metrics: ExecutionMetrics,
        direct_metrics: ExecutionMetrics,
    ) -> ComparisonMetrics:
        """
        Compare metrics from RLM and Direct execution.
        
        Returns ComparisonMetrics with savings and recommendation.
        
        Example:
            >>> rlm_m = ExecutionMetrics(cost_usd=0.001, execution_time_seconds=1.6, steps_taken=3)
            >>> direct_m = ExecutionMetrics(cost_usd=0.0003, execution_time_seconds=0.8, steps_taken=0)
            >>> comparison = manager._compare_metrics(rlm_m, direct_m)
            >>> print(comparison.cost_delta)
            0.0007
            >>> print(comparison.recommendation)
            "Direct is 70% cheaper but RLM is 3x more thorough"
        
        Implementation notes:
        - Calculates cost delta (RLM cost - Direct cost)
        - Calculates time delta (RLM time - Direct time)
        - Generates recommendation based on trade-offs
        - RLM is slower but more thorough (explores multiple paths)
        - Direct is faster and cheaper but single-path
        """
        from .models import ComparisonMetrics
        
        # Calculate deltas
        cost_delta = rlm_metrics.cost_usd - direct_metrics.cost_usd
        time_delta = rlm_metrics.execution_time_seconds - direct_metrics.execution_time_seconds
        token_delta = rlm_metrics.total_tokens - direct_metrics.total_tokens
        
        # Calculate percentage differences
        cost_pct = (cost_delta / direct_metrics.cost_usd * 100) if direct_metrics.cost_usd > 0 else 0
        time_pct = (time_delta / direct_metrics.execution_time_seconds * 100) if direct_metrics.execution_time_seconds > 0 else 0
        
        # Generate recommendation
        if cost_delta > 0.0001:  # RLM is significantly more expensive
            if time_pct > 50:  # And significantly slower
                recommendation = (
                    f"Direct LLM is {abs(cost_pct):.0f}% cheaper and "
                    f"{abs(time_pct):.0f}% faster. "
                    f"RLM explores {rlm_metrics.steps_taken} steps for more thorough analysis."
                )
            else:
                recommendation = (
                    f"Direct LLM is {abs(cost_pct):.0f}% cheaper. "
                    f"Use Direct for cost-sensitive queries."
                )
        else:
            recommendation = (
                f"RLM and Direct have similar costs. "
                f"Use RLM for complex analysis, Direct for speed."
            )
        
        return ComparisonMetrics(
            rlm_cost_usd=rlm_metrics.cost_usd,
            direct_cost_usd=direct_metrics.cost_usd,
            cost_delta_usd=cost_delta,
            cost_delta_percent=cost_pct,
            rlm_time_seconds=rlm_metrics.execution_time_seconds,
            direct_time_seconds=direct_metrics.execution_time_seconds,
            time_delta_seconds=time_delta,
            time_delta_percent=time_pct,
            rlm_tokens=rlm_metrics.total_tokens,
            direct_tokens=direct_metrics.total_tokens,
            token_delta=token_delta,
            rlm_steps=rlm_metrics.steps_taken,
            direct_steps=direct_metrics.steps_taken,
            recommendation=recommendation,
        )
    
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
            
        Example:
            >>> manager = ChatManager()
            >>> await manager.process_message("Test query")
            >>> json_export = manager.export_conversation("json")
            >>> print(json_export)
            '{"conversation_id": "...", "messages": [...]}'
            
            >>> markdown_export = manager.export_conversation("markdown")
            >>> print(markdown_export)
            '# Conversation\\n\\n## Query 1\\n...'
            
        Implementation notes:
        - JSON: Serializable format for data processing
        - Markdown: Human-readable with clear sections
        - CSV: Tabular format with metrics in columns
        - All formats include timestamp, mode, and metrics
        """
        messages = self.get_messages()
        
        if format == "json":
            import json
            export_data = {
                "conversation_id": self.conversation_id,
                "message_count": len(messages),
                "messages": [
                    {
                        "id": msg.id,
                        "timestamp": msg.timestamp.isoformat(),
                        "user_query": msg.user_query,
                        "mode": msg.mode,
                        "file_context_length": len(msg.file_context or ""),
                        "rlm_response": msg.rlm_response.content if msg.rlm_response else None,
                        "rlm_metrics": {
                            "total_tokens": msg.rlm_metrics.total_tokens,
                            "cost_usd": float(msg.rlm_metrics.cost_usd),
                            "execution_time_seconds": msg.rlm_metrics.execution_time_seconds,
                            "steps_taken": msg.rlm_metrics.steps_taken,
                        } if msg.rlm_metrics else None,
                        "direct_response": msg.direct_response.content if msg.direct_response else None,
                        "direct_metrics": {
                            "total_tokens": msg.direct_metrics.total_tokens,
                            "cost_usd": float(msg.direct_metrics.cost_usd),
                            "execution_time_seconds": msg.direct_metrics.execution_time_seconds,
                            "steps_taken": msg.direct_metrics.steps_taken,
                        } if msg.direct_metrics else None,
                        "comparison": {
                            "cost_delta_usd": float(msg.comparison_metrics.cost_delta_usd),
                            "cost_delta_percent": float(msg.comparison_metrics.cost_delta_percent),
                            "token_delta": msg.comparison_metrics.token_delta,
                            "time_delta_seconds": msg.comparison_metrics.time_delta_seconds,
                            "recommendation": msg.comparison_metrics.recommendation,
                        } if msg.comparison_metrics else None,
                    }
                    for msg in messages
                ]
            }
            return json.dumps(export_data, indent=2)
        
        elif format == "markdown":
            lines = [
                f"# Conversation Report",
                f"**ID:** {self.conversation_id}",
                f"**Messages:** {len(messages)}",
                "",
            ]
            
            for i, msg in enumerate(messages, 1):
                lines.append(f"## Query {i}")
                lines.append(f"**Mode:** {msg.mode}")
                lines.append(f"**Time:** {msg.timestamp.isoformat()}")
                lines.append("")
                
                lines.append("### User Query")
                lines.append(f"```\n{msg.user_query}\n```")
                lines.append("")
                
                if msg.rlm_response:
                    lines.append("### RLM Response")
                    lines.append(f"```\n{msg.rlm_response.content}\n```")
                    lines.append("")
                    if msg.rlm_metrics:
                        lines.append("**RLM Metrics:**")
                        lines.append(f"- Tokens: {msg.rlm_metrics.total_tokens}")
                        lines.append(f"- Cost: ${msg.rlm_metrics.cost_usd:.6f}")
                        lines.append(f"- Time: {msg.rlm_metrics.execution_time_seconds:.2f}s")
                        lines.append(f"- Steps: {msg.rlm_metrics.steps_taken}")
                        lines.append("")
                
                if msg.direct_response:
                    lines.append("### Direct Response")
                    lines.append(f"```\n{msg.direct_response.content}\n```")
                    lines.append("")
                    if msg.direct_metrics:
                        lines.append("**Direct Metrics:**")
                        lines.append(f"- Tokens: {msg.direct_metrics.total_tokens}")
                        lines.append(f"- Cost: ${msg.direct_metrics.cost_usd:.6f}")
                        lines.append(f"- Time: {msg.direct_metrics.execution_time_seconds:.2f}s")
                        lines.append(f"- Steps: {msg.direct_metrics.steps_taken}")
                        lines.append("")
                
                if msg.comparison_metrics:
                    lines.append("### Comparison")
                    lines.append(f"{msg.comparison_metrics.recommendation}")
                    lines.append("")
                    lines.append("**Deltas:**")
                    lines.append(f"- Cost: ${msg.comparison_metrics.cost_delta_usd:.6f} ({msg.comparison_metrics.cost_delta_percent:.0f}%)")
                    lines.append(f"- Tokens: {msg.comparison_metrics.token_delta}")
                    lines.append(f"- Time: {msg.comparison_metrics.time_delta_seconds:.2f}s ({msg.comparison_metrics.time_delta_percent:.0f}%)")
                    lines.append("")
            
            return "\n".join(lines)
        
        elif format == "csv":
            lines = [
                "Query,Mode,RLM_Tokens,RLM_Cost,RLM_Time,RLM_Steps,Direct_Tokens,Direct_Cost,Direct_Time,Direct_Steps,Cost_Delta,Token_Delta,Time_Delta",
            ]
            
            for msg in messages:
                rlm_tokens = msg.rlm_metrics.total_tokens if msg.rlm_metrics else ""
                rlm_cost = f"{msg.rlm_metrics.cost_usd:.6f}" if msg.rlm_metrics else ""
                rlm_time = f"{msg.rlm_metrics.execution_time_seconds:.2f}" if msg.rlm_metrics else ""
                rlm_steps = msg.rlm_metrics.steps_taken if msg.rlm_metrics else ""
                
                direct_tokens = msg.direct_metrics.total_tokens if msg.direct_metrics else ""
                direct_cost = f"{msg.direct_metrics.cost_usd:.6f}" if msg.direct_metrics else ""
                direct_time = f"{msg.direct_metrics.execution_time_seconds:.2f}" if msg.direct_metrics else ""
                direct_steps = msg.direct_metrics.steps_taken if msg.direct_metrics else ""
                
                cost_delta = f"{msg.comparison_metrics.cost_delta_usd:.6f}" if msg.comparison_metrics else ""
                token_delta = msg.comparison_metrics.token_delta if msg.comparison_metrics else ""
                time_delta = f"{msg.comparison_metrics.time_delta_seconds:.2f}" if msg.comparison_metrics else ""
                
                # Escape quotes in query
                query = msg.user_query.replace('"', '""')
                
                line = f'"{query}",{msg.mode},{rlm_tokens},{rlm_cost},{rlm_time},{rlm_steps},{direct_tokens},{direct_cost},{direct_time},{direct_steps},{cost_delta},{token_delta},{time_delta}'
                lines.append(line)
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json', 'markdown', or 'csv'.")
    
    @property
    def conversation_id(self) -> str:
        """Get current conversation ID."""
        return self.session_state.get("conversation_id", "")
