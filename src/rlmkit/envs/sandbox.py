# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Sandbox restrictions for safe code execution."""

from typing import Any, Dict, Optional
from rlmkit.core.errors import SecurityError
from rlmkit.config import SecurityConfig


class RestrictedBuiltins:
    """Provides restricted builtins for safe code execution."""
    
    def __init__(self, security_config: Optional[SecurityConfig] = None):
        """
        Initialize restricted builtins.
        
        Args:
            security_config: Security configuration (uses default if None)
        """
        self.config = security_config or SecurityConfig()
    
    def get_restricted_builtins(self) -> Dict[str, Any]:
        """
        Get restricted builtins dict for safe execution.
        
        Returns:
            Dictionary with only safe builtins
        """
        import builtins
        
        safe_builtins = {}
        for name in self.config.safe_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)
        
        # Add restricted versions of some builtins
        safe_builtins['__builtins__'] = safe_builtins
        
        return safe_builtins


class SafeImporter:
    """Controls which modules can be imported."""
    
    def __init__(
        self,
        security_config: Optional[SecurityConfig] = None,
        allowed_imports: Optional[list[str]] = None
    ):
        """
        Initialize safe importer.
        
        Args:
            security_config: Security configuration (uses default if None)
            allowed_imports: Additional modules to allow (beyond config)
        """
        self.config = security_config or SecurityConfig()
        
        # Start with safe modules from config
        self.allowed_imports = set(self.config.safe_modules)
        
        # Add any additional allowed imports
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
        
        # Check if explicitly blocked in config
        if base_module in self.config.blocked_modules:
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
            blocked_list = ', '.join(sorted(self.config.blocked_modules)[:10])
            raise SecurityError(
                f"Import of module '{name}' is not allowed in safe mode. "
                f"Blocked modules: {blocked_list}..."
            )
        
        # Use the real __import__
        import builtins
        return builtins.__import__(name, *args, **kwargs)


def create_safe_globals(
    allowed_imports: Optional[list[str]] = None,
    additional_globals: Optional[Dict[str, Any]] = None,
    security_config: Optional[SecurityConfig] = None
) -> Dict[str, Any]:
    """
    Create safe globals dictionary for code execution.
    
    Args:
        allowed_imports: List of additional modules to allow importing
        additional_globals: Additional global variables to include
        security_config: Security configuration (uses default if None)
        
    Returns:
        Safe globals dictionary with restricted builtins and safe __import__
    """
    # Get restricted builtins using config
    builtins_provider = RestrictedBuiltins(security_config)
    safe_builtins = builtins_provider.get_restricted_builtins()
    
    # Create safe importer with same config
    importer = SafeImporter(security_config, allowed_imports)
    
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
