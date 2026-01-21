# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Python REPL execution environment for RLM."""

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional

from rlmkit.envs.sandbox import create_safe_globals
from rlmkit.envs.timeout import create_timeout, TimeoutError as ExecTimeoutError
from rlmkit.tools import peek, grep, chunk, select


class PyReplEnv:
    """
    Python REPL-like execution environment for RLM.
    
    Executes Python code in a controlled environment with:
    - Persistent globals (state across executions)
    - Stdout/stderr capture
    - Exception handling
    - Optional sandboxing (safe mode)
    - Timeout and output limits
    
    Example:
        >>> env = PyReplEnv()
        >>> result = env.execute("x = 1 + 1\\nprint(x)")
        >>> result['stdout']
        '2\\n'
    """
    
    def __init__(
        self,
        safe_mode: bool = False,
        allowed_imports: Optional[list[str]] = None,
        max_exec_time_s: float = 5.0,
        max_stdout_chars: int = 10000,
    ) -> None:
        """
        Initialize PyReplEnv.
        
        Args:
            safe_mode: Enable sandbox restrictions
            allowed_imports: List of allowed module names (only in safe mode)
            max_exec_time_s: Maximum execution time per code cell
            max_stdout_chars: Maximum characters in stdout/stderr
        """
        self.safe_mode = safe_mode
        self.allowed_imports = allowed_imports or ['json', 're', 'math', 'datetime']
        self.max_exec_time_s = max_exec_time_s
        self.max_stdout_chars = max_stdout_chars
        
        # Content storage (set via set_content)
        self._content: Optional[str] = None
        
        # Persistent environment globals
        if safe_mode:
            # Use restricted globals in safe mode
            self.env_globals = create_safe_globals(allowed_imports=self.allowed_imports)
        else:
            # Use full builtins in unsafe mode
            self.env_globals = {
                '__builtins__': __builtins__,
                '__name__': '__rlm__',
            }
        
        # Add content navigation tools
        self.env_globals['peek'] = peek
        self.env_globals['grep'] = grep
        self.env_globals['chunk'] = chunk
        self.env_globals['select'] = select
        
    def set_content(self, content: str) -> None:
        """
        Set the large content P that tools will operate on.
        
        Args:
            content: The large text content to analyze
        """
        self._content = content
        self.env_globals['P'] = content
        
        # Create wrapper functions that automatically pass content as first arg
        # This allows LLM to call peek(0, 10) instead of peek(P, 0, 10)
        self.env_globals['peek'] = lambda *args, **kwargs: peek(content, *args, **kwargs)
        self.env_globals['grep'] = lambda *args, **kwargs: grep(content, *args, **kwargs)
        self.env_globals['chunk'] = lambda *args, **kwargs: chunk(content, *args, **kwargs)
        self.env_globals['select'] = lambda *args, **kwargs: select(content, *args, **kwargs)
        
    def get_var(self, name: str) -> Any:
        """
        Get a variable value from the environment.
        
        Args:
            name: Variable name
            
        Returns:
            Variable value or None if not found
        """
        return self.env_globals.get(name)
    
    def execute(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code and return result.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary with keys:
                - stdout: str - captured stdout
                - stderr: str - captured stderr
                - exception: str | None - exception message if any
                - timeout: bool - whether execution timed out
                - truncated: bool - whether output was truncated
        """
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        exception_msg: Optional[str] = None
        timeout_occurred = False
        truncated = False
        
        try:
            # Create timeout context
            timeout_ctx = create_timeout(self.max_exec_time_s, use_signal=True)
            
            # Redirect stdout and stderr
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                # Execute code with timeout
                with timeout_ctx:
                    exec(code, self.env_globals)
                
        except ExecTimeoutError as e:
            # Timeout occurred
            timeout_occurred = True
            exception_msg = str(e)
        except Exception as e:
            # Other exception
            exception_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
        # Get output
        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()
        
        # Truncate if needed
        if len(stdout) > self.max_stdout_chars:
            stdout = stdout[:self.max_stdout_chars] + "\n... (truncated)"
            truncated = True
            
        if len(stderr) > self.max_stdout_chars:
            stderr = stderr[:self.max_stdout_chars] + "\n... (truncated)"
            truncated = True
        
        return {
            'stdout': stdout,
            'stderr': stderr,
            'exception': exception_msg,
            'timeout': timeout_occurred,
            'truncated': truncated,
        }
    
    def reset(self) -> None:
        """Reset the environment to initial state."""
        self.env_globals.clear()
        
        # Restore appropriate globals based on safe_mode
        if self.safe_mode:
            self.env_globals.update(create_safe_globals(allowed_imports=self.allowed_imports))
        else:
            self.env_globals['__builtins__'] = __builtins__
            self.env_globals['__name__'] = '__rlm__'
        
        # Restore content if set
        if self._content is not None:
            content = self._content  # Capture for lambda closure
            self.env_globals['P'] = content
            # Bind tools to content
            self.env_globals['peek'] = lambda *args, **kwargs: peek(content, *args, **kwargs)
            self.env_globals['grep'] = lambda *args, **kwargs: grep(content, *args, **kwargs)
            self.env_globals['chunk'] = lambda *args, **kwargs: chunk(content, *args, **kwargs)
            self.env_globals['select'] = lambda *args, **kwargs: select(content, *args, **kwargs)
        else:
            # No content set, use unbound tools
            self.env_globals['peek'] = peek
            self.env_globals['grep'] = grep
            self.env_globals['chunk'] = chunk
            self.env_globals['select'] = select
