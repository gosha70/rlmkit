"""Tests for RestrictedSandboxAdapter (RestrictedPython-based sandbox)."""

import pytest

from rlmkit.infrastructure.sandbox.restricted_sandbox import RestrictedSandboxAdapter


# ---------------------------------------------------------------------------
# Safe operations
# ---------------------------------------------------------------------------


class TestRestrictedSandboxSafeOps:
    """Verify that normal, safe code executes correctly."""

    def test_basic_arithmetic(self):
        """Simple arithmetic runs and produces no exception."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("x = 1 + 2")
        assert result.success
        assert sb.get_variable("x") == 3

    def test_print_captured(self):
        """print() output is captured in stdout."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("print(42)")
        assert result.success
        assert "42" in result.stdout

    def test_multiple_prints(self):
        """Multiple print() calls are accumulated."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("print('a')\nprint('b')")
        assert result.success
        assert "a" in result.stdout
        assert "b" in result.stdout

    def test_variables_persist(self):
        """Variables set in one execution are available in the next."""
        sb = RestrictedSandboxAdapter()
        sb.execute("x = 10")
        result = sb.execute("print(x + 5)")
        assert result.success
        assert "15" in result.stdout

    def test_list_comprehension(self):
        """List comprehensions work under RestrictedPython."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("squares = [i**2 for i in range(5)]\nprint(squares)")
        assert result.success
        assert "[0, 1, 4, 9, 16]" in result.stdout

    def test_string_methods(self):
        """Basic string methods are accessible."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("x = 'hello world'\nprint(x.upper())")
        assert result.success
        assert "HELLO WORLD" in result.stdout

    def test_dict_operations(self):
        """Dict creation and access work."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("d = {'a': 1, 'b': 2}\nprint(d['a'] + d['b'])")
        assert result.success
        assert "3" in result.stdout


# ---------------------------------------------------------------------------
# Safe imports
# ---------------------------------------------------------------------------


class TestRestrictedSandboxSafeImports:
    """Verify that allowed modules can be imported."""

    def test_import_json(self):
        """json module is importable and usable."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("import json\nprint(json.dumps({'k': 'v'}))")
        assert result.success
        assert '"k"' in result.stdout

    def test_import_re(self):
        """re module is importable and usable."""
        sb = RestrictedSandboxAdapter()
        # Use re.findall to avoid RestrictedPython guard issues with
        # calling methods on match objects returned by re.search.
        result = sb.execute("import re\nmatches = re.findall('[0-9]+', 'abc123')\nprint(matches[0])")
        assert result.success
        assert "123" in result.stdout

    def test_import_math(self):
        """math module is importable and usable."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("import math\nprint(int(math.pi * 100))")
        assert result.success
        assert "314" in result.stdout

    def test_import_collections(self):
        """collections module is importable."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute(
            "import collections\nd = collections.Counter('aab')\nprint(d['a'])"
        )
        assert result.success
        assert "2" in result.stdout


# ---------------------------------------------------------------------------
# Blocked imports
# ---------------------------------------------------------------------------


class TestRestrictedSandboxBlockedImports:
    """Verify that dangerous modules are blocked."""

    def test_blocks_os(self):
        """import os is blocked."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("import os")
        assert not result.success
        assert "SecurityViolationError" in result.exception
        assert "os" in result.exception

    def test_blocks_sys(self):
        """import sys is blocked."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("import sys")
        assert not result.success
        assert "SecurityViolationError" in result.exception

    def test_blocks_subprocess(self):
        """import subprocess is blocked."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("import subprocess")
        assert not result.success
        assert "SecurityViolationError" in result.exception

    def test_blocks_socket(self):
        """import socket is blocked."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("import socket")
        assert not result.success
        assert "SecurityViolationError" in result.exception

    def test_blocks_shutil(self):
        """import shutil is blocked."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("import shutil")
        assert not result.success
        assert "SecurityViolationError" in result.exception

    def test_blocks_pathlib(self):
        """import pathlib is blocked."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("import pathlib")
        assert not result.success
        assert "SecurityViolationError" in result.exception


# ---------------------------------------------------------------------------
# Blocked dunder attribute access (RestrictedPython compile-time check)
# ---------------------------------------------------------------------------


class TestRestrictedSandboxDunderBlocked:
    """Verify that access to dunder attributes is blocked at compile time."""

    def test_blocks_class_access(self):
        """Accessing __class__ is a SyntaxError."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("x = (1).__class__")
        assert not result.success
        assert "SyntaxError" in result.exception

    def test_blocks_bases_access(self):
        """Accessing __bases__ is a SyntaxError."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("x = int.__bases__")
        assert not result.success
        assert "SyntaxError" in result.exception

    def test_blocks_subclasses_access(self):
        """Accessing __subclasses__ is a SyntaxError."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("x = int.__subclasses__()")
        assert not result.success
        assert "SyntaxError" in result.exception

    def test_blocks_globals_access(self):
        """Accessing __globals__ is a SyntaxError."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("def f(): pass\nx = f.__globals__")
        assert not result.success
        assert "SyntaxError" in result.exception


# ---------------------------------------------------------------------------
# Blocked builtins
# ---------------------------------------------------------------------------


class TestRestrictedSandboxBlockedBuiltins:
    """Verify that dangerous builtins are not available."""

    def test_no_open(self):
        """open() is not available."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("open('/etc/passwd')")
        assert not result.success

    def test_no_eval(self):
        """eval() is not available."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("eval('1+1')")
        assert not result.success

    def test_no_exec(self):
        """exec() is not available."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("exec('x=1')")
        assert not result.success

    def test_no_compile(self):
        """compile() is not available."""
        sb = RestrictedSandboxAdapter()
        result = sb.execute("compile('x=1', '<str>', 'exec')")
        assert not result.success


# ---------------------------------------------------------------------------
# SandboxPort interface methods
# ---------------------------------------------------------------------------


class TestRestrictedSandboxInterface:
    """Verify set_variable, get_variable, reset, is_healthy."""

    def test_set_and_get_variable(self):
        """Variables can be injected and retrieved."""
        sb = RestrictedSandboxAdapter()
        sb.set_variable("my_data", [1, 2, 3])
        assert sb.get_variable("my_data") == [1, 2, 3]

    def test_get_variable_not_found(self):
        """get_variable returns None for unknown names."""
        sb = RestrictedSandboxAdapter()
        assert sb.get_variable("nonexistent") is None

    def test_reset_clears_namespace(self):
        """reset() removes all user-defined variables."""
        sb = RestrictedSandboxAdapter()
        sb.execute("x = 42")
        assert sb.get_variable("x") == 42

        sb.reset()
        assert sb.get_variable("x") is None

    def test_is_healthy(self):
        """is_healthy() returns True."""
        sb = RestrictedSandboxAdapter()
        assert sb.is_healthy() is True

    def test_custom_allowed_modules(self):
        """Custom allowed_modules restricts to exactly those modules."""
        sb = RestrictedSandboxAdapter(allowed_modules=frozenset({"json"}))

        # json allowed
        result = sb.execute("import json\nprint('ok')")
        assert result.success

        # math NOT allowed (not in the custom set)
        result = sb.execute("import math")
        assert not result.success
        assert "SecurityViolationError" in result.exception

    def test_stdout_truncation(self):
        """Output exceeding max_stdout_chars is truncated."""
        sb = RestrictedSandboxAdapter(max_stdout_chars=50)
        result = sb.execute("print('x' * 200)")
        assert result.truncated
        assert len(result.stdout) <= 70  # 50 chars + truncation message
