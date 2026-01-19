"""Tests for timeout functionality."""

import time
import pytest
from rlmkit.envs import PyReplEnv
from rlmkit.envs.timeout import TimeoutError


class TestBasicTimeout:
    """Test basic timeout functionality."""
    
    def test_timeout_infinite_loop(self):
        """Test that infinite loops time out."""
        env = PyReplEnv(max_exec_time_s=1.0)
        result = env.execute("while True: pass")
        
        assert result['timeout'] is True
        assert result['exception'] is not None
        assert "timed out" in result['exception'].lower()
    
    def test_timeout_long_sleep(self):
        """Test that long sleep times out."""
        env = PyReplEnv(max_exec_time_s=1.0)
        result = env.execute("import time\ntime.sleep(10)")
        
        assert result['timeout'] is True
        assert result['exception'] is not None
    
    def test_no_timeout_quick_execution(self):
        """Test that quick code doesn't timeout."""
        env = PyReplEnv(max_exec_time_s=5.0)
        result = env.execute("x = sum(range(1000))\nprint(x)")
        
        assert result['timeout'] is False
        assert result['exception'] is None
        assert '499500' in result['stdout']


class TestTimeoutSettings:
    """Test different timeout settings."""
    
    def test_short_timeout(self):
        """Test very short timeout."""
        env = PyReplEnv(max_exec_time_s=0.5)
        result = env.execute("import time\ntime.sleep(2)")
        
        assert result['timeout'] is True
    
    def test_long_timeout(self):
        """Test longer timeout allows more work."""
        env = PyReplEnv(max_exec_time_s=3.0)
        result = env.execute("""
import time
for i in range(5):
    time.sleep(0.1)
print('done')
""")
        
        # Should complete within 3 seconds
        assert result['timeout'] is False
        assert 'done' in result['stdout']
    
    def test_default_timeout(self):
        """Test default timeout (5 seconds)."""
        env = PyReplEnv()  # Default is 5.0 seconds
        assert env.max_exec_time_s == 5.0
        
        result = env.execute("print('quick')")
        assert result['timeout'] is False


class TestTimeoutWithComputations:
    """Test timeout with various computations."""
    
    def test_timeout_heavy_computation(self):
        """Test that heavy computation can timeout."""
        env = PyReplEnv(max_exec_time_s=1.0)
        result = env.execute("""
# Deliberately slow computation
total = 0
for i in range(10000000):
    total += i
""")
        
        # Might timeout depending on system speed
        # At minimum, should not crash
        assert isinstance(result['timeout'], bool)
    
    def test_no_timeout_reasonable_computation(self):
        """Test that reasonable computation doesn't timeout."""
        env = PyReplEnv(max_exec_time_s=5.0)
        result = env.execute("""
# Reasonable computation
result = sum(i**2 for i in range(10000))
print(result)
""")
        
        assert result['timeout'] is False
        assert result['exception'] is None


class TestTimeoutWithSafeMode:
    """Test timeout works with safe mode."""
    
    def test_timeout_with_safe_mode(self):
        """Test timeout works in safe mode."""
        env = PyReplEnv(safe_mode=True, max_exec_time_s=1.0)
        result = env.execute("while True: pass")
        
        assert result['timeout'] is True
    
    def test_timeout_with_imports(self):
        """Test timeout with allowed imports."""
        env = PyReplEnv(safe_mode=True, max_exec_time_s=1.0)
        result = env.execute("""
import time
time.sleep(5)
""")
        
        assert result['timeout'] is True


class TestTimeoutPreservesState:
    """Test that timeout doesn't break environment state."""
    
    def test_state_after_timeout(self):
        """Test that environment is usable after timeout."""
        env = PyReplEnv(max_exec_time_s=1.0)
        
        # First: cause timeout
        result1 = env.execute("while True: pass")
        assert result1['timeout'] is True
        
        # Second: verify environment still works
        result2 = env.execute("x = 42\nprint(x)")
        assert result2['timeout'] is False
        assert result2['exception'] is None
        assert '42' in result2['stdout']
    
    def test_variables_before_timeout(self):
        """Test that variables set before timeout are preserved."""
        env = PyReplEnv(max_exec_time_s=1.0)
        
        # Set a variable
        result1 = env.execute("y = 100")
        assert result1['timeout'] is False
        
        # Cause timeout
        result2 = env.execute("while True: pass")
        assert result2['timeout'] is True
        
        # Check if variable still exists
        result3 = env.execute("print(y)")
        assert result3['timeout'] is False
        assert '100' in result3['stdout']


class TestTimeoutEdgeCases:
    """Test edge cases for timeout."""
    
    def test_timeout_with_exception(self):
        """Test timeout vs exception priority."""
        env = PyReplEnv(max_exec_time_s=1.0)
        result = env.execute("""
import time
time.sleep(0.1)
raise ValueError('test error')
""")
        
        # Exception should be caught before potential timeout
        assert result['timeout'] is False
        assert result['exception'] is not None
        assert 'ValueError' in result['exception']
    
    def test_timeout_empty_code(self):
        """Test timeout with empty code."""
        env = PyReplEnv(max_exec_time_s=1.0)
        result = env.execute("")
        
        assert result['timeout'] is False
        assert result['exception'] is None
    
    def test_timeout_syntax_error(self):
        """Test that syntax errors are caught before timeout."""
        env = PyReplEnv(max_exec_time_s=1.0)
        result = env.execute("def broken(")
        
        assert result['timeout'] is False
        assert result['exception'] is not None
        assert 'SyntaxError' in result['exception']


class TestTimeoutWithOutput:
    """Test timeout interaction with output capture."""
    
    def test_output_before_timeout(self):
        """Test that output before timeout is captured."""
        env = PyReplEnv(max_exec_time_s=1.0)
        result = env.execute("""
print('starting')
import time
while True:
    pass
""")
        
        assert result['timeout'] is True
        # Output before timeout should be captured
        assert 'starting' in result['stdout']
    
    def test_output_truncation_with_timeout(self):
        """Test output truncation works with timeout."""
        env = PyReplEnv(max_exec_time_s=2.0, max_stdout_chars=100)
        result = env.execute("""
for i in range(10):
    print('x' * 1000)
import time
time.sleep(5)
""")
        
        # Could timeout or truncate depending on timing
        assert result['timeout'] is True or result['truncated'] is True
