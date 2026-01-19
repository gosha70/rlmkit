# RLMKit Code Review Guide

Complete reference for understanding and debugging the codebase.

## 📚 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Components](#core-components)
3. [Key Classes & Functions](#key-classes--functions)
4. [Common Issues & Solutions](#common-issues--solutions)
5. [Testing Strategy](#testing-strategy)
6. [Debugging Tips](#debugging-tips)

---

## Architecture Overview

```
┌─────────────────────────────────────┐
│  User Code                          │
│  from rlmkit.envs import PyReplEnv  │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  PyReplEnv (Core Execution)         │
│  • Manages persistent Python state  │
│  • Coordinates all subsystems       │
└─────┬──────────┬──────────┬─────────┘
      │          │          │
      ▼          ▼          ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Sandbox  │ │ Timeout  │ │ Content  │
│ Security │ │ Limits   │ │ Storage  │
└──────────┘ └──────────┘ └──────────┘
```

---

## Core Components

### 1. **PyReplEnv** (`src/rlmkit/envs/pyrepl_env.py`)
**Purpose**: Main execution environment - the heart of RLMKit

**Key Responsibilities:**
- Execute Python code with persistent state
- Capture stdout/stderr
- Handle exceptions gracefully
- Manage large content (P variable)
- Coordinate security & timeouts

**Critical Code Sections:**

```python
def execute(self, code: str) -> Dict[str, Any]:
    """
    Main execution method - READ THIS FIRST
    
    Flow:
    1. Create timeout context
    2. Redirect stdout/stderr to buffers
    3. Execute code with exec() in self.env_globals
    4. Catch TimeoutError vs other exceptions
    5. Truncate output if needed
    6. Return structured result
    """
```

**Important Fields:**
- `self.env_globals` - Persistent Python namespace (dict)
- `self._content` - Large content stored as P variable
- `self.safe_mode` - Security toggle
- `self.max_exec_time_s` - Timeout limit
- `self.max_stdout_chars` - Output limit

**Common Issues:**
- ❌ **Lost state**: If you modify `env_globals` directly, call `reset()` to restore
- ❌ **Memory leak**: Large content in P should be replaced, not appended
- ✅ **Solution**: Always use `set_content()` to replace P

---

### 2. **Sandbox** (`src/rlmkit/envs/sandbox.py`)
**Purpose**: Security restrictions for safe code execution

**Key Components:**

#### RestrictedBuiltins
Filters Python's built-in functions:
```python
BLOCKED_BUILTINS = {
    'open', 'eval', 'exec', 'compile',  # Code execution
    '__import__', 'breakpoint',          # System access
    'exit', 'quit', 'input'             # User interaction
}

SAFE_BUILTINS = {
    'print', 'len', 'sum', 'range',     # Basic functions
    'list', 'dict', 'str', 'int',       # Data structures
    # ... see code for full list
}
```

#### SafeImporter
Controls module imports:
```python
BLOCKED_MODULES = {
    'os', 'sys', 'subprocess',          # System operations
    'socket', 'http', 'urllib',         # Network
    'pathlib', 'shutil', 'tempfile',    # File system
    # ... see code for full list
}

SAFE_MODULES = {
    'json', 're', 'math', 'datetime',   # Data processing
    'itertools', 'functools',           # Functional programming
    'collections', 'typing',            # Type utilities
    # ... see code for full list
}
```

**How It Works:**
1. `create_safe_globals()` builds restricted namespace
2. Replaces `__builtins__` with filtered dict
3. Injects custom `__import__` function
4. Custom `__import__` checks whitelist before allowing

**Critical Code:**
```python
def create_safe_globals(...) -> Dict[str, Any]:
    """
    Creates execution environment for safe mode.
    
    KEY: __import__ is added to safe_builtins dict,
    not to globals directly. Python looks in __builtins__
    for import statement handling.
    """
    safe_builtins = RestrictedBuiltins.get_restricted_builtins()
    importer = SafeImporter(allowed_imports)
    
    # CRITICAL: Add __import__ to builtins, not globals!
    safe_builtins['__import__'] = importer.safe_import
```

**Common Issues:**
- ❌ **Import not blocked**: Check if module in `BLOCKED_MODULES`
- ❌ **Safe import fails**: Verify module in `SAFE_MODULES` or custom whitelist
- ✅ **Solution**: Module names are case-sensitive, check exact spelling

---

### 3. **Timeout** (`src/rlmkit/envs/timeout.py`)
**Purpose**: Prevent infinite loops and long-running code

**Two Implementations:**

#### SignalTimeout (Unix only)
Fast, efficient using OS signals:
```python
class SignalTimeout:
    """
    Uses signal.alarm() - Unix/Linux/macOS only
    
    Pros: Fast, low overhead
    Cons: Unix-only, can't timeout C extensions
    """
    
    def __enter__(self):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.seconds)
```

#### ProcessTimeout (Cross-platform)
Reliable using multiprocessing:
```python
class ProcessTimeout:
    """
    Runs code in separate process
    
    Pros: Works everywhere, guaranteed kill
    Cons: Higher overhead, state isolation
    """
    
    def run(self, func, *args, **kwargs):
        process = mp.Process(target=func)
        process.start()
        process.join(timeout=self.seconds)
        if process.is_alive():
            process.terminate()
```

**Auto-Selection Logic:**
```python
def create_timeout(seconds, use_signal=True):
    """
    Automatically picks best timeout for platform:
    - Unix + use_signal=True → SignalTimeout
    - Windows or use_signal=False → ProcessTimeout
    """
    has_signal_alarm = hasattr(signal, 'SIGALRM')
    
    if use_signal and has_signal_alarm:
        return SignalTimeout(seconds)
    else:
        return ProcessTimeout(seconds)
```

**Common Issues:**
- ❌ **Timeout doesn't work**: Check if `SIGALRM` available (Unix only for signals)
- ❌ **State lost after timeout**: ProcessTimeout runs in separate process
- ✅ **Solution**: Use SignalTimeout on Unix for state preservation

---

### 4. **Error Hierarchy** (`src/rlmkit/core/errors.py`)
**Purpose**: Structured exception handling

```python
RLMError (Base)
├── BudgetExceeded      # Token/step limits (Week 3)
├── ExecutionError      # Code execution failures
│   └── TimeoutError    # Specific timeout case
├── ParseError          # Code parsing issues (Week 3)
└── SecurityError       # Sandbox violations
```

**Usage:**
```python
from rlmkit.core.errors import SecurityError

if not self.check_import(name):
    raise SecurityError(f"Import '{name}' not allowed")
```

---

## Key Classes & Functions

### PyReplEnv Methods

| Method | Purpose | Returns | Side Effects |
|--------|---------|---------|--------------|
| `__init__()` | Setup environment | - | Creates globals dict |
| `execute(code)` | Run Python code | Result dict | Modifies env_globals |
| `set_content(text)` | Store large content | - | Sets P variable |
| `get_var(name)` | Read variable | Any | None |
| `reset()` | Clear environment | - | Clears all variables |

### Result Dictionary Structure

```python
{
    'stdout': str,      # Captured print() output
    'stderr': str,      # Captured warnings/errors
    'exception': str,   # Exception traceback or None
    'timeout': bool,    # True if timed out
    'truncated': bool,  # True if output truncated
}
```

---

## Common Issues & Solutions

### Issue 1: ImportError in Safe Mode
```python
# ❌ Problem
env = PyReplEnv(safe_mode=True)
env.execute("import os")  # SecurityError

# ✅ Solution 1: Add to whitelist
env = PyReplEnv(safe_mode=True, allowed_imports=['os'])

# ✅ Solution 2: Use unsafe mode
env = PyReplEnv(safe_mode=False)
```

### Issue 2: Timeout on Normal Code
```python
# ❌ Problem
env = PyReplEnv(max_exec_time_s=0.1)  # Too short!
env.execute("sum(range(1000000))")     # Timeout

# ✅ Solution: Increase timeout
env = PyReplEnv(max_exec_time_s=5.0)
```

### Issue 3: Output Truncation
```python
# ❌ Problem
env = PyReplEnv(max_stdout_chars=100)
env.execute("print('x' * 10000)")     # Truncated

# ✅ Solution: Increase limit
env = PyReplEnv(max_stdout_chars=50000)
```

### Issue 4: Lost Variables After Reset
```python
# ❌ Problem
env.execute("x = 42")
env.reset()              # x is lost!
env.execute("print(x)")  # NameError

# ✅ Solution: Don't call reset, or re-set variables
env.execute("x = 42")
# Keep using env without reset
```

### Issue 5: Content P Not Available
```python
# ❌ Problem
env = PyReplEnv()
env.execute("print(P)")  # NameError: P not defined

# ✅ Solution: Set content first
env.set_content("Large document here")
env.execute("print(len(P))")  # Works!
```

---

## Testing Strategy

### Test Organization

```
tests/
├── test_env.py         # Basic execution (15 tests)
├── test_sandbox.py     # Security (22 tests)
└── test_timeout.py     # Timeouts (17 tests)
```

### Test Coverage by Feature

| Feature | Tests | Key Scenarios |
|---------|-------|---------------|
| Execution | 15 | Basic code, exceptions, state, content |
| Security | 22 | Blocked ops, safe ops, imports |
| Timeout | 17 | Infinite loops, sleep, state preservation |

### Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=rlmkit --cov-report=term-missing

# Specific test file
pytest tests/test_sandbox.py -v

# Specific test
pytest tests/test_env.py::TestBasicExecution::test_execute_simple_code -v

# Stop on first failure
pytest tests/ -x
```

---

## Debugging Tips

### 1. Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 2. Inspect Environment State
```python
env = PyReplEnv()
env.execute("x = 42")

# Check what's in environment
print(env.env_globals.keys())  # ['__builtins__', '__name__', 'x']
print(env.get_var('x'))        # 42
```

### 3. Test Timeout Behavior
```python
env = PyReplEnv(max_exec_time_s=1.0)

# This should timeout
result = env.execute("import time; time.sleep(10)")
print(f"Timeout: {result['timeout']}")
print(f"Exception: {result['exception']}")
```

### 4. Check Security Restrictions
```python
from rlmkit.envs.sandbox import SafeImporter

importer = SafeImporter()
print(importer.check_import('json'))  # True
print(importer.check_import('os'))    # False
```

### 5. Verify Builtins
```python
from rlmkit.envs.sandbox import RestrictedBuiltins

builtins = RestrictedBuiltins.get_restricted_builtins()
print('open' in builtins)     # False (blocked)
print('print' in builtins)    # True (safe)
```

### 6. Test Exception Handling
```python
env = PyReplEnv()
result = env.execute("1 / 0")

assert result['exception'] is not None
assert 'ZeroDivisionError' in result['exception']
print(result['exception'])  # Full traceback
```

---

## Quick Reference: File Locations

**Core Implementation:**
- `src/rlmkit/envs/pyrepl_env.py` - Main execution (55 lines)
- `src/rlmkit/envs/sandbox.py` - Security (44 lines)
- `src/rlmkit/envs/timeout.py` - Timeouts (65 lines)
- `src/rlmkit/core/errors.py` - Exceptions (5 lines)

**Tests:**
- `tests/test_env.py` - Execution tests
- `tests/test_sandbox.py` - Security tests
- `tests/test_timeout.py` - Timeout tests

**Documentation:**
- `README.md` - User guide
- `IMPLEMENTATION_PLAN.md` - Development roadmap
- `CODE_REVIEW_GUIDE.md` - This file

---

## Maintenance Checklist

When modifying code, check:

- [ ] Does it preserve state across executions?
- [ ] Are security restrictions maintained?
- [ ] Will it timeout properly on infinite loops?
- [ ] Are errors captured and returned gracefully?
- [ ] Is output truncated if too large?
- [ ] Do existing tests still pass?
- [ ] Should new tests be added?
- [ ] Is documentation updated?

---

## Next Steps (Week 2+)

**Week 2: Content Navigation**
- `peek()` - View content sections
- `grep()` - Search with context
- `chunk()` - Split content
- `select()` - Range operations

**Week 3: RLM Controller** ⭐
- Main control loop
- LLM integration
- Message management
- Budget tracking

**Week 4-6:**
- Recursion, Providers, Polish

---

*Last Updated: Week 1 Complete (54 tests passing, 77% coverage)*
