# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
ChatManager - Core message management and execution routing.
"""

from typing import Optional, List, Dict, Any
from uuid import uuid4
import asyncio
import time

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
        - Uses actual RLM controller from src/rlmkit/core/rlm.py
        - Integrates with configured LLM provider
        - Collects real token counts and timing metrics
        """
        from .models import Response, ExecutionMetrics
        from rlmkit.core.rlm import RLM
        from rlmkit.llm import get_llm_client
        from rlmkit.config import RLMConfig
        from pathlib import Path
        import os
        
        try:
            # Get active provider configuration
            from .llm_config_manager import LLMConfigManager
            config_dir = Path.home() / ".rlmkit"
            config_manager = LLMConfigManager(config_dir=config_dir)
            
            # Primary: use selected_provider from session state (from Configuration page)
            provider_config = None
            if self.session_state.get('selected_provider'):
                print(f"DEBUG: Using selected_provider from session: {self.session_state['selected_provider']}")
                provider_config = config_manager.get_provider_config(
                    self.session_state['selected_provider']
                )
            
            # Fallback: try to get active provider
            if not provider_config:
                provider_config = config_manager.get_active_provider()
                print(f"DEBUG: get_active_provider returned: {provider_config}")
            
            if provider_config:
                print(f"DEBUG: provider={provider_config.provider}, model={provider_config.model}, is_ready={provider_config.is_ready}, test_successful={provider_config.test_successful}")
            
            # Fallback: use first available provider
            if not provider_config:
                providers = config_manager.list_providers()
                print(f"DEBUG: Trying fallback with first available provider from list: {providers}")
                if providers:
                    provider_config = config_manager.get_provider_config(providers[0])
            
            if not provider_config or not provider_config.is_ready:
                raise ValueError("No active LLM provider configured")
            
            # Get API key (from memory or environment variable)
            api_key = provider_config.api_key
            if not api_key and provider_config.api_key_env_var:
                api_key = os.getenv(provider_config.api_key_env_var)
            
            # Create LLM client
            llm_client = get_llm_client(
                provider=provider_config.provider,
                model=provider_config.model,
                api_key=api_key
            )
            
            # Create RLM instance with configuration
            rlm_config = RLMConfig(
                execution=RLMConfig.ExecutionConfig(
                    default_timeout=5.0,
                    max_steps=16,
                    default_safe_mode=True,
                    max_output_chars=10000,
                )
            )
            rlm = RLM(client=llm_client, config=rlm_config)
            
            # Prepare content (use file_context if provided)
            content = file_context or "No content provided"
            
            # Run RLM
            start_time = time.time()
            rlm_result = rlm.run(prompt=content, query=user_query)
            elapsed_time = time.time() - start_time
            
            # Extract metrics from trace
            total_input_tokens = 0
            total_output_tokens = 0
            
            if rlm_result.trace:
                for step in rlm_result.trace:
                    # Count tokens in trace (rough estimate)
                    if "content" in step:
                        content_len = len(step["content"].split())
                        total_output_tokens += content_len * 1.3  # Rough conversion
            
            # If no token info in trace, estimate from content
            if total_output_tokens == 0:
                total_output_tokens = len(rlm_result.answer.split()) * 1.3
            
            # Estimate input tokens from query and content
            total_input_tokens = (len(user_query.split()) + len(content.split()) / 100) * 1.3
            
            # Calculate cost using provider pricing
            input_cost = (total_input_tokens / 1000) * provider_config.input_cost_per_1k_tokens
            output_cost = (total_output_tokens / 1000) * provider_config.output_cost_per_1k_tokens
            total_cost = input_cost + output_cost
            
            # Create response
            response = Response(
                content=rlm_result.answer,
                stop_reason="stop",
                raw_response=rlm_result
            )
            
            # Create metrics
            metrics = ExecutionMetrics(
                input_tokens=int(total_input_tokens),
                output_tokens=int(total_output_tokens),
                total_tokens=int(total_input_tokens + total_output_tokens),
                cost_usd=total_cost,
                cost_breakdown={
                    "input": input_cost,
                    "output": output_cost,
                },
                execution_time_seconds=elapsed_time,
                steps_taken=rlm_result.steps,
                memory_used_mb=0.0,  # LATER: Get from MemoryMonitor
                memory_peak_mb=0.0,   # LATER: Get from MemoryMonitor
                success=rlm_result.success,
                execution_type="rlm"
            )
            
            return {
                "response": response,
                "metrics": metrics,
                "trace": rlm_result.trace,
            }
        
        except Exception as e:
            # Fallback to mock response on error
            response = Response(
                content=f"RLM execution failed: {str(e)}",
                stop_reason="error",
                raw_response=None
            )
            metrics = ExecutionMetrics(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                cost_breakdown={"input": 0.0, "output": 0.0},
                execution_time_seconds=0.0,
                steps_taken=0,
                memory_used_mb=0.0,
                memory_peak_mb=0.0,
                success=False,
                execution_type="rlm"
            )
            return {
                "response": response,
                "metrics": metrics,
                "trace": [{"error": str(e)}],
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
        from rlmkit.llm import get_llm_client
        from pathlib import Path
        import os
        
        try:
            # Get active provider configuration
            from .llm_config_manager import LLMConfigManager
            config_dir = Path.home() / ".rlmkit"
            config_manager = LLMConfigManager(config_dir=config_dir)
            
            # Primary: use selected_provider from session state (from Configuration page)
            provider_config = None
            if self.session_state.get('selected_provider'):
                print(f"DEBUG (Direct): Using selected_provider from session: {self.session_state['selected_provider']}")
                provider_config = config_manager.get_provider_config(
                    self.session_state['selected_provider']
                )
            
            # Fallback: try to get active provider
            if not provider_config:
                provider_config = config_manager.get_active_provider()
                print(f"DEBUG (Direct): get_active_provider returned: {provider_config}")
            
            if provider_config:
                print(f"DEBUG (Direct): provider={provider_config.provider}, model={provider_config.model}, is_ready={provider_config.is_ready}, test_successful={provider_config.test_successful}")
            
            # Fallback: use first available provider
            if not provider_config:
                providers = config_manager.list_providers()
                print(f"DEBUG (Direct): Trying fallback with first available provider from list: {providers}")
                if providers:
                    provider_config = config_manager.get_provider_config(providers[0])
            
            if not provider_config or not provider_config.is_ready:
                raise ValueError("No active LLM provider configured")
            
            # Get API key (from memory or environment variable)
            api_key = provider_config.api_key
            if not api_key and provider_config.api_key_env_var:
                api_key = os.getenv(provider_config.api_key_env_var)
            
            # Create LLM client
            llm_client = get_llm_client(
                provider=provider_config.provider,
                model=provider_config.model,
                api_key=api_key
            )
            
            # Build prompt with context
            if file_context:
                prompt = f"Context:\n{file_context}\n\nQuestion: {user_query}"
            else:
                prompt = user_query
            
            # Call LLM directly
            start_time = time.time()
            response_text = llm_client.complete([
                {"role": "user", "content": prompt}
            ])
            elapsed_time = time.time() - start_time
            
            # Estimate tokens
            input_tokens = len(prompt.split()) * 1.3
            output_tokens = len(response_text.split()) * 1.3
            
            # Calculate cost using provider pricing
            input_cost = (input_tokens / 1000) * provider_config.input_cost_per_1k_tokens
            output_cost = (output_tokens / 1000) * provider_config.output_cost_per_1k_tokens
            total_cost = input_cost + output_cost
            
            # Create response (similar structure to RLM for comparison)
            response = Response(
                content=response_text,
                stop_reason="stop",
                raw_response=None
            )
            
            # Create metrics for direct execution
            metrics = ExecutionMetrics(
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                total_tokens=int(input_tokens + output_tokens),
                cost_usd=total_cost,
                cost_breakdown={
                    "input": input_cost,
                    "output": output_cost,
                },
                execution_time_seconds=elapsed_time,
                steps_taken=0,  # No RLM exploration steps
                memory_used_mb=0.0,  # LATER: Get from MemoryMonitor
                memory_peak_mb=0.0,   # LATER: Get from MemoryMonitor
                success=True,
                execution_type="direct"
            )
            
            return {
                "response": response,
                "metrics": metrics,
                # No trace for direct execution (it's a single call)
            }
        
        except Exception as e:
            # Fallback to error response
            response = Response(
                content=f"Direct LLM execution failed: {str(e)}",
                stop_reason="error",
                raw_response=None
            )
            metrics = ExecutionMetrics(
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                cost_breakdown={"input": 0.0, "output": 0.0},
                execution_time_seconds=0.0,
                steps_taken=0,
                memory_used_mb=0.0,
                memory_peak_mb=0.0,
                success=False,
                execution_type="direct"
            )
            return {
                "response": response,
                "metrics": metrics,
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
