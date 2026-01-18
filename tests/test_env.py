"""Tests for PyReplEnv execution environment."""

import pytest
from rlmkit.envs import PyReplEnv


class TestBasicExecution:
    """Test basic code execution."""
    
    def test_execute_simple_code(self):
        """Test executing simple arithmetic and print."""
        env = PyReplEnv()
        result = env.execute("x = 1 + 1\nprint(x)")
        
        assert result['stdout'] == "2\n"
        assert result['exception'] is None
        assert result['timeout'] is False
        
    def test_execute_multiple_statements(self):
        """Test executing multiple statements."""
        env = PyReplEnv()
        result = env.execute("a = 10\nb = 20\nprint(a + b)")
        
        assert result['stdout'] == "30\n"
        assert result['exception'] is None
        
    def test_persistent_globals(self):
        """Test that variables persist across executions."""
        env = PyReplEnv()
        
        # First execution
        result1 = env.execute("x = 42")
        assert result1['exception'] is None
        
        # Second execution can access x
        result2 = env.execute("print(x)")
        assert result2['stdout'] == "42\n"
        assert result2['exception'] is None
        
    def test_capture_exception(self):
        """Test that exceptions are captured properly."""
        env = PyReplEnv()
        result = env.execute("1 / 0")
        
        assert result['exception'] is not None
        assert "ZeroDivisionError" in result['exception']
        assert result['stdout'] == ""
        
    def test_capture_stderr(self):
        """Test capturing stderr output."""
        env = PyReplEnv()
        result = env.execute("import sys\nprint('error', file=sys.stderr)")
        
        assert result['stderr'] == "error\n"
        assert result['exception'] is None


class TestContentAccess:
    """Test accessing content P."""
    
    def test_set_and_access_content(self):
        """Test setting content and accessing via P."""
        env = PyReplEnv()
        env.set_content("Hello World!")
        
        result = env.execute("print(P)")
        assert result['stdout'] == "Hello World!\n"
        assert result['exception'] is None
        
    def test_get_var(self):
        """Test getting variable values from environment."""
        env = PyReplEnv()
        env.execute("result = 42")
        
        value = env.get_var('result')
        assert value == 42
        
    def test_get_var_not_found(self):
        """Test getting non-existent variable returns None."""
        env = PyReplEnv()
        value = env.get_var('nonexistent')
        assert value is None


class TestOutputLimits:
    """Test output truncation."""
    
    def test_truncates_huge_stdout(self):
        """Test that huge stdout is truncated."""
        env = PyReplEnv(max_stdout_chars=100)
        result = env.execute("print('x' * 10000)")
        
        assert len(result['stdout']) <= 120  # 100 + "\n... (truncated)"
        assert result['truncated'] is True
        assert "truncated" in result['stdout']
        
    def test_no_truncation_small_output(self):
        """Test that small output is not truncated."""
        env = PyReplEnv(max_stdout_chars=1000)
        result = env.execute("print('hello')")
        
        assert result['stdout'] == "hello\n"
        assert result['truncated'] is False


class TestReset:
    """Test environment reset."""
    
    def test_reset_clears_variables(self):
        """Test that reset clears variables but keeps content."""
        env = PyReplEnv()
        env.set_content("Test content")
        
        # Set a variable
        env.execute("x = 100")
        assert env.get_var('x') == 100
        
        # Reset
        env.reset()
        
        # Variable is gone
        assert env.get_var('x') is None
        
        # But P is still there
        assert env.get_var('P') == "Test content"
        
    def test_reset_without_content(self):
        """Test reset without content set."""
        env = PyReplEnv()
        env.execute("y = 200")
        
        env.reset()
        
        assert env.get_var('y') is None
        assert env.get_var('P') is None


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_syntax_error(self):
        """Test that syntax errors are captured."""
        env = PyReplEnv()
        result = env.execute("def broken(")
        
        assert result['exception'] is not None
        assert "SyntaxError" in result['exception']
        
    def test_empty_code(self):
        """Test executing empty code."""
        env = PyReplEnv()
        result = env.execute("")
        
        assert result['stdout'] == ""
        assert result['exception'] is None
        
    def test_multiline_code_with_functions(self):
        """Test executing multiline code with function definitions."""
        env = PyReplEnv()
        code = """
def add(a, b):
    return a + b

result = add(5, 7)
print(result)
"""
        result = env.execute(code)
        
        assert result['stdout'] == "12\n"
        assert result['exception'] is None
        assert env.get_var('result') == 12
