# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.
"""
ChatManager - Core message management and execution routing.
"""

from typing import Optional, List, Dict, Any, Tuple
from uuid import uuid4
import asyncio
import time

from .models import (
    ChatMessage, ExecutionMetrics, ComparisonMetrics,
    ExecutionPlan, ExecutionSlot, ExecutionResult, GeneralizedComparison,
    RAGConfig, Response, LLMProviderConfig,
)
from .memory_monitor import MemoryMonitor
from .profile_store import resolve_system_prompt


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

    def _resolve_provider(
        self,
        provider_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Tuple[LLMProviderConfig, str]:
        """Resolve provider config and effective API key.

        Returns:
            (provider_config, effective_api_key)

        Raises:
            ValueError: if no provider is configured or API key is missing
        """
        from .llm_config_manager import LLMConfigManager
        from .secret_store import resolve_api_key
        from pathlib import Path

        config_dir = Path.home() / ".rlmkit"
        config_manager = LLMConfigManager(config_dir=config_dir)

        provider_config = None
        if provider_name:
            provider_config = config_manager.get_provider_config(provider_name)
        if not provider_config:
            provider_config = config_manager.get_active_provider()
        if not provider_config:
            providers = config_manager.list_providers()
            if providers:
                provider_config = config_manager.get_provider_config(providers[0])
        if not provider_config:
            raise ValueError(
                "No LLM provider configured - go to Configuration page to set one up"
            )

        # Resolve API key: explicit arg → session cache → SecretStore → env var
        effective_api_key = api_key if api_key else provider_config.api_key
        if not effective_api_key:
            effective_api_key = resolve_api_key(
                provider_config.provider, self.session_state
            )
        if not effective_api_key:
            raise ValueError(
                f"Provider {provider_config.provider} is configured but API key is missing"
            )

        return provider_config, effective_api_key

    async def process_message(
        self,
        user_query: str,
        mode: str = "compare",
        file_context: Optional[str] = None,
        file_info: Optional[Dict[str, Any]] = None,
        selected_provider: Optional[str] = None,
        api_key: Optional[str] = None,
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
                rlm_result = await self._execute_rlm(user_query, file_context, selected_provider, api_key)
                message.rlm_response = rlm_result["response"]
                message.rlm_metrics = rlm_result["metrics"]
                message.rlm_trace = rlm_result["trace"]
            except Exception as e:
                message.error = f"RLM execution failed: {str(e)}"
        
        if mode in ("direct_only", "compare"):
            try:
                direct_result = await self._execute_direct(user_query, file_context, selected_provider, api_key)
                message.direct_response = direct_result["response"]
                message.direct_metrics = direct_result["metrics"]
            except Exception as e:
                if not message.error:
                    message.error = f"Direct execution failed: {str(e)}"

        if mode == "rag_only":
            try:
                rag_result = await self._execute_rag(user_query, file_context, selected_provider, api_key)
                message.rag_response = rag_result["response"]
                message.rag_metrics = rag_result["metrics"]
                message.rag_trace = rag_result.get("trace")
            except Exception as e:
                if not message.error:
                    message.error = f"RAG execution failed: {str(e)}"
        
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

        # Auto-persist to storage
        store = self.session_state.get("conversation_store")
        conv_id = self.session_state.get("conversation_id")
        if store and conv_id:
            try:
                store.save_message(conv_id, message)
            except Exception:
                pass  # Non-fatal: don't break chat over persistence failure

        return message
    
    async def _execute_rlm(
        self,
        user_query: str,
        file_context: Optional[str] = None,
        selected_provider: Optional[str] = None,
        api_key: Optional[str] = None,
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
        from rlmkit.core.rlm import RLM
        from rlmkit.llm import get_llm_client
        from rlmkit.config import RLMConfig, ExecutionConfig

        try:
            provider_config, effective_api_key = self._resolve_provider(
                selected_provider, api_key
            )

            # Create LLM client
            llm_client = get_llm_client(
                provider=provider_config.provider,
                model=provider_config.model,
                api_key=effective_api_key
            )
            
            # Create RLM instance with configuration
            rlm_config = RLMConfig(
                execution=ExecutionConfig(
                    default_timeout=5.0,
                    max_steps=16,
                    default_safe_mode=True,
                    max_output_chars=10000,
                )
            )
            rlm = RLM(client=llm_client, config=rlm_config)
            
            # Prepare content (use file_context if provided)
            content = file_context or "No content provided"

            # Resolve system prompt for RLM mode
            sys_prompt = resolve_system_prompt("rlm", self.session_state)

            # Run RLM
            mem = MemoryMonitor()
            mem.reset()
            start_time = time.time()
            rlm_result = rlm.run(prompt=content, query=user_query, system_prompt=sys_prompt)
            elapsed_time = time.time() - start_time
            mem.capture()
            
            # Extract actual token counts from RLMResult (populated by API metadata)
            total_input_tokens = rlm_result.total_input_tokens
            total_output_tokens = rlm_result.total_output_tokens

            # Fallback to per-step trace counts if RLMResult totals are zero
            if total_input_tokens == 0 and total_output_tokens == 0:
                for step in rlm_result.trace:
                    total_input_tokens += step.get("input_tokens", 0)
                    total_output_tokens += step.get("output_tokens", 0)
            
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
                memory_used_mb=mem.current_mb(),
                memory_peak_mb=mem.peak_mb(),
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
        selected_provider: Optional[str] = None,
        api_key: Optional[str] = None,
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
        from rlmkit.llm import get_llm_client

        try:
            provider_config, effective_api_key = self._resolve_provider(
                selected_provider, api_key
            )

            # Create LLM client
            llm_client = get_llm_client(
                provider=provider_config.provider,
                model=provider_config.model,
                api_key=effective_api_key
            )
            
            # Build prompt with context
            if file_context:
                prompt = f"Context:\n{file_context}\n\nQuestion: {user_query}"
            else:
                prompt = user_query

            # Resolve system prompt for direct mode
            sys_prompt = resolve_system_prompt("direct", self.session_state)
            messages: list = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": prompt})

            # Call LLM directly (with metadata for accurate token counts)
            mem = MemoryMonitor()
            mem.reset()
            start_time = time.time()
            if hasattr(llm_client, 'complete_with_metadata'):
                try:
                    llm_response = llm_client.complete_with_metadata(messages)
                    response_text = llm_response.content
                    input_tokens = llm_response.input_tokens or 0
                    output_tokens = llm_response.output_tokens or 0
                except (NotImplementedError, AttributeError):
                    response_text = llm_client.complete(messages)
                    input_tokens = 0
                    output_tokens = 0
            else:
                response_text = llm_client.complete(messages)
                input_tokens = 0
                output_tokens = 0
            elapsed_time = time.time() - start_time
            mem.capture()
            
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
                steps_taken=1,  # Single LLM call (not 0 - that's confusing)
                memory_used_mb=mem.current_mb(),
                memory_peak_mb=mem.peak_mb(),
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
    
    async def _execute_rag(
        self,
        user_query: str,
        file_context: Optional[str] = None,
        selected_provider: Optional[str] = None,
        api_key: Optional[str] = None,
        rag_config: Optional[RAGConfig] = None,
    ) -> Dict[str, Any]:
        """Execute RAG strategy (embed chunks, retrieve, generate)."""
        from rlmkit.llm import get_llm_client
        from rlmkit.strategies.rag import RAGStrategy
        from rlmkit.strategies.embeddings import OpenAIEmbedder

        try:
            provider_config, effective_api_key = self._resolve_provider(
                selected_provider, api_key
            )

            llm_client = get_llm_client(
                provider=provider_config.provider,
                model=provider_config.model,
                api_key=effective_api_key,
            )

            # RAG config (from parameter, session state, or defaults)
            rc = rag_config or self.session_state.get("rag_config") or RAGConfig()

            # Resolve embedding API key: prefer OpenAI provider key for embeddings
            embedding_api_key = effective_api_key
            if provider_config.provider != "openai":
                # Try to find an OpenAI provider key for embeddings
                from .llm_config_manager import LLMConfigManager, load_env_file
                from pathlib import Path
                import os
                mgr = LLMConfigManager(config_dir=Path.home() / ".rlmkit")
                openai_cfg = mgr.get_provider_config("openai")
                if openai_cfg:
                    embedding_api_key = openai_cfg.api_key
                    if not embedding_api_key and openai_cfg.api_key_env_var:
                        load_env_file()
                        embedding_api_key = os.getenv(openai_cfg.api_key_env_var)
                if not embedding_api_key:
                    embedding_api_key = effective_api_key  # fallback

            embedder = OpenAIEmbedder(
                model=rc.embedding_model,
                api_key=embedding_api_key,
            )

            # Resolve system prompt for RAG mode
            sys_prompt = resolve_system_prompt("rag", self.session_state)

            # Use IndexedRAGStrategy when vector store is available
            vector_store = self.session_state.get("vector_store")
            conv_id = self.session_state.get("conversation_id")
            if vector_store and conv_id:
                from rlmkit.strategies.indexed_rag import IndexedRAGStrategy
                rag = IndexedRAGStrategy(
                    client=llm_client, embedder=embedder,
                    vector_store=vector_store,
                    collection=f"conv_{conv_id}_artifacts",
                    chunk_size=rc.chunk_size,
                    chunk_overlap=rc.chunk_overlap,
                    top_k=rc.top_k,
                    system_prompt=sys_prompt,
                )
            else:
                rag = RAGStrategy(
                    client=llm_client, embedder=embedder,
                    chunk_size=rc.chunk_size,
                    chunk_overlap=rc.chunk_overlap,
                    top_k=rc.top_k,
                    system_prompt=sys_prompt,
                )

            content = file_context or "No content provided"
            mem = MemoryMonitor()
            mem.reset()
            start_time = time.time()
            result = rag.run(content=content, query=user_query)
            elapsed_time = time.time() - start_time
            mem.capture()

            input_cost = (result.tokens.input_tokens / 1000) * provider_config.input_cost_per_1k_tokens
            output_cost = (result.tokens.output_tokens / 1000) * provider_config.output_cost_per_1k_tokens
            total_cost = input_cost + output_cost

            response = Response(
                content=result.answer,
                stop_reason="stop" if result.success else "error",
                raw_response=None,
            )

            metrics = ExecutionMetrics(
                input_tokens=result.tokens.input_tokens,
                output_tokens=result.tokens.output_tokens,
                total_tokens=result.tokens.total_tokens,
                cost_usd=total_cost,
                cost_breakdown={"input": input_cost, "output": output_cost},
                execution_time_seconds=elapsed_time,
                steps_taken=result.steps,
                memory_used_mb=mem.current_mb(),
                memory_peak_mb=mem.peak_mb(),
                success=result.success,
                error=result.error,
                execution_type="rag",
            )

            return {
                "response": response,
                "metrics": metrics,
                "trace": result.trace,
            }

        except Exception as e:
            response = Response(
                content=f"RAG execution failed: {str(e)}",
                stop_reason="error",
                raw_response=None,
            )
            metrics = ExecutionMetrics(
                input_tokens=0, output_tokens=0, total_tokens=0,
                cost_usd=0.0, cost_breakdown={"input": 0.0, "output": 0.0},
                execution_time_seconds=0.0, steps_taken=0,
                memory_used_mb=0.0, memory_peak_mb=0.0,
                success=False, execution_type="rag",
            )
            return {
                "response": response,
                "metrics": metrics,
                "trace": [{"error": str(e)}],
            }

    async def process_message_v2(
        self,
        user_query: str,
        execution_plan: ExecutionPlan,
        file_context: Optional[str] = None,
        file_info: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Process user message using an ExecutionPlan with N slots.

        Each slot is executed independently (RLM, Direct, or RAG with its
        own provider).  Results are collected and, when there are 2+ slots,
        a generalized comparison is generated.

        Legacy fields (rlm_response, direct_response, etc.) are also
        populated for backward compatibility with saved conversations and
        existing rendering code.
        """
        mode = "compare" if execution_plan.is_comparison else (
            execution_plan.slots[0].mode + "_only" if execution_plan.slots else "compare"
        )

        message = ChatMessage(
            user_query=user_query,
            mode=mode,
            file_context=file_context,
            file_info=file_info,
            execution_plan=execution_plan,
        )

        results: List[ExecutionResult] = []

        for slot in execution_plan.slots:
            # Auto-generate label if missing
            if not slot.label:
                from .llm_config_manager import LLMConfigManager
                from pathlib import Path
                mgr = LLMConfigManager(config_dir=Path.home() / ".rlmkit")
                cfg = mgr.get_provider_config(slot.provider_name)
                model_name = cfg.model if cfg else slot.provider_name
                mode_label = {"rlm": "RLM", "direct": "Direct", "rag": "RAG"}[slot.mode]
                slot.label = f"{mode_label} ({model_name})"

            try:
                if slot.mode == "rlm":
                    raw = await self._execute_rlm(
                        user_query, file_context, slot.provider_name
                    )
                elif slot.mode == "direct":
                    raw = await self._execute_direct(
                        user_query, file_context, slot.provider_name
                    )
                elif slot.mode == "rag":
                    raw = await self._execute_rag(
                        user_query, file_context, slot.provider_name,
                        rag_config=slot.rag_config,
                    )
                else:
                    continue

                results.append(ExecutionResult(
                    slot=slot,
                    response=raw["response"],
                    metrics=raw["metrics"],
                    trace=raw.get("trace"),
                ))
            except Exception as e:
                err_response = Response(
                    content=f"{slot.label} failed: {str(e)}",
                    stop_reason="error",
                )
                err_metrics = ExecutionMetrics(
                    input_tokens=0, output_tokens=0, total_tokens=0,
                    cost_usd=0.0, cost_breakdown={"input": 0.0, "output": 0.0},
                    execution_time_seconds=0.0, steps_taken=0,
                    memory_used_mb=0.0, memory_peak_mb=0.0,
                    success=False, execution_type=slot.mode,
                )
                results.append(ExecutionResult(
                    slot=slot, response=err_response, metrics=err_metrics,
                ))

        message.execution_results = results

        # Populate legacy fields for backward compat
        for r in results:
            if r.slot.mode == "rlm" and message.rlm_response is None:
                message.rlm_response = r.response
                message.rlm_metrics = r.metrics
                message.rlm_trace = r.trace
            elif r.slot.mode == "direct" and message.direct_response is None:
                message.direct_response = r.response
                message.direct_metrics = r.metrics
            elif r.slot.mode == "rag" and message.rag_response is None:
                message.rag_response = r.response
                message.rag_metrics = r.metrics
                message.rag_trace = r.trace

        # Build legacy comparison for 2-slot RLM+Direct case
        if (message.rlm_metrics and message.direct_metrics
                and len(results) == 2):
            try:
                message.comparison_metrics = self._compare_metrics(
                    message.rlm_metrics, message.direct_metrics
                )
            except Exception:
                pass

        # Build generalized comparison for all multi-slot cases
        if len(results) > 1:
            message.generalized_comparison = self._generalized_compare(results)

        # Add to history
        self.session_state["messages"].append(message)

        # Auto-persist
        store = self.session_state.get("conversation_store")
        conv_id = self.session_state.get("conversation_id")
        if store and conv_id:
            try:
                store.save_message(conv_id, message)
            except Exception:
                pass

        return message

    def _generalized_compare(
        self,
        results: List[ExecutionResult],
    ) -> GeneralizedComparison:
        """Compare N execution results and generate summary."""
        if not results:
            return GeneralizedComparison()

        successful = [r for r in results if r.metrics.success]
        if not successful:
            return GeneralizedComparison(
                results=results,
                recommendation="All executions failed.",
            )

        cheapest = min(successful, key=lambda r: r.metrics.cost_usd)
        fastest = min(successful, key=lambda r: r.metrics.execution_time_seconds)

        parts = []
        if cheapest.slot.label:
            parts.append(f"{cheapest.slot.label} is cheapest (${cheapest.metrics.cost_usd:.4f})")
        if fastest.slot.label and fastest.slot.label != cheapest.slot.label:
            parts.append(f"{fastest.slot.label} is fastest ({fastest.metrics.execution_time_seconds:.2f}s)")

        recommendation = ". ".join(parts) + "." if parts else ""

        return GeneralizedComparison(
            results=results,
            cheapest_label=cheapest.slot.label,
            fastest_label=fastest.slot.label,
            recommendation=recommendation,
        )

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
