"""Recursive Language Model (RLM) controller.

Based on the RLM paper (arXiv:2512.24601), this module implements the core
controller loop that:
1. Loads large prompts as variables in a REPL environment
2. Allows LLM to write Python code to explore the content
3. Executes code and feeds results back to LLM
4. Manages recursion and budgets
5. Extracts final answers
"""

from dataclasses import dataclass
from typing import Optional, Protocol, List, Dict, Any
from .parsing import parse_response, format_result_for_llm, ParsedResponse
from .errors import BudgetExceeded
from ..envs.pyrepl_env import PyReplEnv
from ..config import RLMConfig
from ..prompts import format_system_prompt


class LLMClient(Protocol):
    """Protocol for LLM providers.
    
    Any LLM client must implement this interface to work with RLM.
    """
    
    def complete(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate completion from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            
        Returns:
            Generated text response from LLM
        """
        ...


@dataclass
class RLMResult:
    """Result from RLM execution."""
    
    answer: str
    """Final answer extracted from LLM"""
    
    steps: int
    """Number of execution steps taken"""
    
    trace: List[Dict[str, Any]]
    """Full execution trace (messages + execution results)"""
    
    success: bool = True
    """Whether execution completed successfully"""
    
    error: Optional[str] = None
    """Error message if execution failed"""


class RLM:
    """
    Recursive Language Model controller.
    
    Implements the RLM paradigm from arXiv:2512.24601 where:
    - Large prompts are loaded as variables in a REPL environment
    - LLM writes Python code to explore the content
    - Code execution results feed back to LLM
    - Process repeats until final answer is found
    
    Example:
        >>> from rlmkit import RLM, RLMConfig
        >>> from rlmkit.llm import MockLLMClient
        >>> 
        >>> client = MockLLMClient(["```python\\nx = peek(0, 10)\\n```", "FINAL: Done"])
        >>> rlm = RLM(client=client, config=RLMConfig(max_steps=10))
        >>> result = rlm.run(prompt="Large content here...", query="Summarize this")
        >>> print(result.answer)  # "Done"
    """
    
    def __init__(
        self,
        client: LLMClient,
        config: Optional[RLMConfig] = None
    ):
        """
        Initialize RLM controller.
        
        Args:
            client: LLM client implementing LLMClient protocol
            config: Configuration for execution limits and security
        """
        self.client = client
        self.config = config or RLMConfig()
        self.env: Optional[PyReplEnv] = None
        
    def run(
        self,
        prompt: str,
        query: str,
        system_prompt: Optional[str] = None
    ) -> RLMResult:
        """
        Run RLM on a prompt to answer a query.
        
        This is the main entry point for RLM execution. It:
        1. Initializes a REPL environment with the prompt as variable 'P'
        2. Sends query to LLM with instructions about available tools
        3. Parses LLM response for code or final answer
        4. If code: executes it, shows results to LLM, repeats
        5. If final: extracts answer and returns
        6. Enforces budget limits (max_steps)
        
        Args:
            prompt: Large content to analyze (the "P" variable in paper)
            query: Question to answer about the prompt
            system_prompt: Optional custom system prompt (uses default if None)
            
        Returns:
            RLMResult with answer, execution trace, and statistics
            
        Raises:
            BudgetExceeded: If max_steps is reached without final answer
        """
        # Initialize REPL environment with prompt as 'P'
        self.env = PyReplEnv(
            safe_mode=self.config.execution.default_safe_mode,
            allowed_imports=list(self.config.security.safe_modules),
            max_exec_time_s=self.config.execution.default_timeout,
            max_stdout_chars=self.config.execution.max_output_chars,
        )
        self.env.set_content(prompt)
        
        # Build message history
        messages = []
        
        # Add system prompt
        if system_prompt is None:
            system_prompt = self._build_system_prompt(len(prompt))
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Main execution loop
        trace = []
        steps = 0
        
        try:
            while steps < self.config.execution.max_steps:
                steps += 1
                
                # Get LLM response
                response = self.client.complete(messages)
                
                # Add to trace
                trace.append({
                    "step": steps,
                    "role": "assistant",
                    "content": response
                })
                
                # Parse response
                parsed = parse_response(response)
                
                # Check if complete
                if parsed.is_complete:
                    # Extract final answer
                    answer = self._extract_final_answer(parsed)
                    
                    return RLMResult(
                        answer=answer,
                        steps=steps,
                        trace=trace,
                        success=True
                    )
                
                # Execute code if present
                if parsed.has_code:
                    # Execute in REPL
                    result = self.env.execute(parsed.code)
                    
                    # Format for LLM
                    formatted_result = format_result_for_llm(result)
                    
                    # Add to trace
                    trace.append({
                        "step": steps,
                        "role": "execution",
                        "content": formatted_result,
                        "raw_result": result
                    })
                    
                    # Add to message history
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "user",
                        "content": f"Execution result:\n{formatted_result}"
                    })
                else:
                    # No code and no final - unclear response
                    messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    messages.append({
                        "role": "user",
                        "content": "Please provide either Python code to execute or a FINAL answer."
                    })
            
            # Max steps exceeded
            raise BudgetExceeded(
                f"Maximum steps ({self.config.execution.max_steps}) exceeded without finding final answer"
            )
            
        except BudgetExceeded:
            raise
        except Exception as e:
            # Other errors
            return RLMResult(
                answer="",
                steps=steps,
                trace=trace,
                success=False,
                error=str(e)
            )
    
    def _extract_final_answer(self, parsed: ParsedResponse) -> str:
        """
        Extract final answer from parsed response.
        
        Handles both FINAL: direct answers and FINAL_VAR: variable references.
        
        Args:
            parsed: Parsed LLM response
            
        Returns:
            Final answer string
        """
        if parsed.final_answer:
            return parsed.final_answer
        
        if parsed.final_var and self.env:
            # Get variable from environment
            var_value = self.env.get_var(parsed.final_var)
            if var_value is not None:
                return str(var_value)
            else:
                return f"Error: Variable '{parsed.final_var}' not found in environment"
        
        return "Error: No final answer found"
    
    def _build_system_prompt(self, prompt_length: int) -> str:
        """
        Build default system prompt for RLM.
        
        Uses the templated prompt system for easy customization and versioning.
        
        Args:
            prompt_length: Length of the prompt content in characters
            
        Returns:
            System prompt string
        """
        return format_system_prompt(prompt_length=prompt_length)
