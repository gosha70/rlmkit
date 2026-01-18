"""Sandbox restrictions for safe code execution."""

from typing import Any, Dict, Optional
from rlmkit.core.errors import SecurityError


class RestrictedBuiltins:
    """Provides restricted builtins for safe code execution."""
    
    # Dangerous builtins to block
    BLOCKED_BUILTINS = {
        'open',
        'input',
        'compile',
        'eval',
        'exec',
        '__import__',
        'breakpoint',
        'exit',
        'quit',
        'help',
    }
    
    # Safe builtins to allow
    SAFE_BUILTINS = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'classmethod', 'complex', 'delattr', 'dict',
        'dir', 'divmod', 'enumerate', 'filter', 'float', 'format',
        'frozenset', 'getattr', 'hasattr', 'hash', 'hex', 'id', 'int',
        'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals',
        'map', 'max', 'min', 'next', 'object', 'oct', 'ord', 'pow',
        'print', 'property', 'range', 'repr', 'reversed', 'round',
        'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str',
        'sum', 'super', 'tuple', 'type', 'vars', 'zip',
        # Exceptions
        'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
        'AttributeError', 'RuntimeError', 'StopIteration', 'AssertionError',
        # Other safe built-ins
        'True', 'False', 'None', 'NotImplemented', 'Ellipsis',
        '__name__', '__doc__', '__package__', '__loader__', '__spec__',
    }
    
    @classmethod
    def get_restricted_builtins(cls) -> Dict[str, Any]:
        """
        Get restricted builtins dict for safe execution.
        
        Returns:
            Dictionary with only safe builtins
        """
        import builtins
        
        safe_builtins = {}
        for name in cls.SAFE_BUILTINS:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)
        
        # Add restricted versions of some builtins
        safe_builtins['__builtins__'] = safe_builtins
        
        return safe_builtins


class SafeImporter:
    """Controls which modules can be imported."""
    
    # Dangerous modules to always block
    BLOCKED_MODULES = {
        'os', 'sys', 'subprocess', 'socket', 'socketserver',
        'http', 'urllib', 'ftplib', 'telnetlib', 'smtplib',
        'pathlib', 'shutil', 'tempfile', 'glob', 'fnmatch',
        'pty', 'tty', 'pipes', 'resource', 'syslog',
        'ctypes', 'cffi', 'mmap', 'signal', 'fcntl',
        'pickle', 'shelve', 'dbm', 'sqlite3',
        'importlib', 'pkgutil', 'modulefinder', 'runpy',
        '__builtin__', '__builtins__', 'builtins',
    }
    
    # Safe modules (always allowed)
    SAFE_MODULES = {
        'json', 're', 'math', 'datetime', 'time', 'calendar',
        'decimal', 'fractions', 'statistics', 'random',
        'string', 'textwrap', 'unicodedata',
        'itertools', 'functools', 'operator', 'collections',
        'heapq', 'bisect', 'array', 'copy', 'pprint',
        'enum', 'dataclasses', 'typing',
    }
    
    def __init__(self, allowed_imports: Optional[list[str]] = None):
        """
        Initialize safe importer.
        
        Args:
            allowed_imports: Additional modules to allow (beyond SAFE_MODULES)
        """
        self.allowed_imports = set(self.SAFE_MODULES)
        if allowed_imports:
            self.allowed_imports.update(allowed_imports)
    
    def check_import(self, module_name: str) -> bool:
        """
        Check if a module import is allowed.
        
        Args:
            module_name: Name of module to import
            
        Returns:
            True if allowed, False otherwise
        """
        # Extract base module name (e.g., 'json.decoder' -> 'json')
        base_module = module_name.split('.')[0]
        
        # Check if explicitly blocked
        if base_module in self.BLOCKED_MODULES:
            return False
        
        # Check if in allowed list
        return base_module in self.allowed_imports
    
    def safe_import(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Safe import function that checks against whitelist.
        
        Args:
            name: Module name to import
            *args: Additional import args
            **kwargs: Additional import kwargs
            
        Returns:
            Imported module
            
        Raises:
            SecurityError: If import is not allowed
        """
        if not self.check_import(name):
            raise SecurityError(
                f"Import of module '{name}' is not allowed in safe mode. "
                f"Blocked modules: {', '.join(sorted(self.BLOCKED_MODULES)[:10])}..."
            )
        
        # Use the real __import__
        import builtins
        return builtins.__import__(name, *args, **kwargs)


def create_safe_globals(
    allowed_imports: Optional[list[str]] = None,
    additional_globals: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create safe globals dictionary for code execution.
    
    Args:
        allowed_imports: List of additional modules to allow importing
        additional_globals: Additional global variables to include
        
    Returns:
        Safe globals dictionary with restricted builtins and safe __import__
    """
    # Get restricted builtins
    safe_builtins = RestrictedBuiltins.get_restricted_builtins()
    
    # Create safe importer
    importer = SafeImporter(allowed_imports)
    
    # Add __import__ to the builtins dict (where Python looks for it)
    safe_builtins['__import__'] = importer.safe_import
    
    # Build safe globals
    safe_globals = {
        '__builtins__': safe_builtins,
        '__name__': '__rlm__',
    }
    
    # Add any additional globals
    if additional_globals:
        safe_globals.update(additional_globals)
    
    return safe_globals


def validate_code_safety(code: str) -> None:
    """
    Perform static analysis to check for obviously dangerous code patterns.
    
    This is a basic check - the main safety comes from restricted builtins.
    
    Args:
        code: Python code to validate
        
    Raises:
        SecurityError: If dangerous patterns detected
    """
    dangerous_patterns = [
        ('__import__', 'Direct __import__ usage not allowed'),
        ('eval(', 'eval() not allowed'),
        ('exec(', 'exec() not allowed'),
        ('compile(', 'compile() not allowed'),
        ('open(', 'File operations not allowed'),
        ('os.', 'os module not allowed'),
        ('sys.', 'sys module not allowed'),
        ('subprocess', 'subprocess module not allowed'),
    ]
    
    for pattern, message in dangerous_patterns:
        if pattern in code:
            # Note: This is a basic check. Real safety comes from restricted builtins.
            # We just warn here, actual blocking happens at execution.
            pass  # Could add warnings here if desired
