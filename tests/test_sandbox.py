"""Tests for sandbox security restrictions."""

import pytest
from rlmkit.envs import PyReplEnv
from rlmkit import SecurityError


class TestBlockDangerousBuiltins:
    """Test that dangerous builtins are blocked in safe mode."""
    
    def test_blocks_open(self):
        """Test that open() is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("open('/etc/passwd', 'r')")
        
        assert result['exception'] is not None
        assert "open" in result['exception'] or "NameError" in result['exception']
        
    def test_blocks_eval(self):
        """Test that eval() is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("eval('1+1')")
        
        assert result['exception'] is not None
        assert "eval" in result['exception'] or "NameError" in result['exception']
        
    def test_blocks_exec(self):
        """Test that exec() is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("exec('x = 1')")
        
        assert result['exception'] is not None
        assert "exec" in result['exception'] or "NameError" in result['exception']
        
    def test_blocks_compile(self):
        """Test that compile() is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("compile('x=1', '<string>', 'exec')")
        
        assert result['exception'] is not None
        assert "compile" in result['exception'] or "NameError" in result['exception']


class TestBlockDangerousImports:
    """Test that dangerous imports are blocked in safe mode."""
    
    def test_blocks_os_import(self):
        """Test that 'import os' is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("import os")
        
        assert result['exception'] is not None
        assert "SecurityError" in result['exception'] or "not allowed" in result['exception']
        
    def test_blocks_sys_import(self):
        """Test that 'import sys' is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("import sys")
        
        assert result['exception'] is not None
        assert "SecurityError" in result['exception'] or "not allowed" in result['exception']
        
    def test_blocks_subprocess_import(self):
        """Test that 'import subprocess' is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("import subprocess")
        
        assert result['exception'] is not None
        assert "SecurityError" in result['exception'] or "not allowed" in result['exception']
        
    def test_blocks_socket_import(self):
        """Test that 'import socket' is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("import socket")
        
        assert result['exception'] is not None
        assert "SecurityError" in result['exception'] or "not allowed" in result['exception']
        
    def test_blocks_pathlib_import(self):
        """Test that 'import pathlib' is blocked."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("import pathlib")
        
        assert result['exception'] is not None
        assert "SecurityError" in result['exception'] or "not allowed" in result['exception']


class TestAllowSafeImports:
    """Test that safe imports are allowed in safe mode."""
    
    def test_allows_json_import(self):
        """Test that 'import json' is allowed."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("""
import json
x = json.dumps({'key': 'value'})
print(x)
""")
        
        assert result['exception'] is None
        assert '"key": "value"' in result['stdout'] or '"key":"value"' in result['stdout']
        
    def test_allows_re_import(self):
        """Test that 'import re' is allowed."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("""
import re
match = re.search(r'\\d+', 'abc123')
print(match.group())
""")
        
        assert result['exception'] is None
        assert '123' in result['stdout']
        
    def test_allows_math_import(self):
        """Test that 'import math' is allowed."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("""
import math
print(math.pi)
""")
        
        assert result['exception'] is None
        assert '3.14' in result['stdout']
        
    def test_allows_datetime_import(self):
        """Test that 'import datetime' is allowed."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("""
import datetime
now = datetime.datetime.now()
print(type(now).__name__)
""")
        
        assert result['exception'] is None
        assert 'datetime' in result['stdout']


class TestCustomAllowedImports:
    """Test custom import whitelisting."""
    
    def test_custom_allowed_import(self):
        """Test that custom allowed imports work."""
        env = PyReplEnv(safe_mode=True, allowed_imports=['collections'])
        result = env.execute("""
import collections
d = collections.defaultdict(int)
d['key'] += 1
print(d['key'])
""")
        
        assert result['exception'] is None
        assert '1' in result['stdout']
        
    def test_blocks_non_whitelisted_import(self):
        """Test that non-whitelisted imports are still blocked."""
        env = PyReplEnv(safe_mode=True, allowed_imports=['json'])
        result = env.execute("import os")
        
        assert result['exception'] is not None
        assert "SecurityError" in result['exception'] or "not allowed" in result['exception']


class TestSafeBuiltins:
    """Test that safe builtins work correctly."""
    
    def test_safe_builtins_work(self):
        """Test that safe builtins like print, len, etc. work."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("""
x = [1, 2, 3]
print(len(x))
print(sum(x))
print(max(x))
""")
        
        assert result['exception'] is None
        assert '3' in result['stdout']
        assert '6' in result['stdout']
        
    def test_list_comprehension_works(self):
        """Test that list comprehensions work in safe mode."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("""
squares = [x**2 for x in range(5)]
print(squares)
""")
        
        assert result['exception'] is None
        assert '[0, 1, 4, 9, 16]' in result['stdout']
        
    def test_lambda_functions_work(self):
        """Test that lambda functions work in safe mode."""
        env = PyReplEnv(safe_mode=True)
        result = env.execute("""
add = lambda a, b: a + b
print(add(2, 3))
""")
        
        assert result['exception'] is None
        assert '5' in result['stdout']


class TestUnsafeModeAllowsAll:
    """Test that unsafe mode allows all operations."""
    
    def test_unsafe_allows_imports(self):
        """Test that unsafe mode allows os import."""
        env = PyReplEnv(safe_mode=False)
        result = env.execute("""
import os
# Don't actually do anything dangerous, just verify import works
print('os imported')
""")
        
        # In unsafe mode, os import should succeed
        assert result['exception'] is None
        assert 'os imported' in result['stdout']
        
    def test_unsafe_allows_open(self):
        """Test that unsafe mode has open() available (but don't use it)."""
        env = PyReplEnv(safe_mode=False)
        result = env.execute("print(callable(open))")
        
        assert result['exception'] is None
        assert 'True' in result['stdout']


class TestResetPreservesSecurityMode:
    """Test that reset maintains security settings."""
    
    def test_reset_keeps_safe_mode(self):
        """Test that reset keeps safe mode restrictions."""
        env = PyReplEnv(safe_mode=True)
        env.execute("x = 42")
        
        env.reset()
        
        # Try blocked operation after reset
        result = env.execute("import os")
        assert result['exception'] is not None
        assert "SecurityError" in result['exception'] or "not allowed" in result['exception']
        
    def test_reset_keeps_allowed_imports(self):
        """Test that custom allowed imports persist after reset."""
        env = PyReplEnv(safe_mode=True, allowed_imports=['collections'])
        env.execute("x = 1")
        
        env.reset()
        
        # collections should still be allowed
        result = env.execute("import collections")
        assert result['exception'] is None
