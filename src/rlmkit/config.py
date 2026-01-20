"""Configuration management for RLMKit."""

import os
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import importlib.resources
from .llm.config import LLMConfig


@dataclass
class SecurityConfig:
    """Security configuration for sandbox execution."""
    
    # Module-level security
    blocked_modules: Set[str] = field(default_factory=lambda: {
        'os', 'sys', 'subprocess', 'socket', 'socketserver',
        'http', 'urllib', 'ftplib', 'telnetlib', 'smtplib',
        'pathlib', 'shutil', 'tempfile', 'glob', 'fnmatch',
        'pty', 'tty', 'pipes', 'resource', 'syslog',
        'ctypes', 'cffi', 'mmap', 'signal', 'fcntl',
        'pickle', 'shelve', 'dbm', 'sqlite3',
        'importlib', 'pkgutil', 'modulefinder', 'runpy',
        '__builtin__', '__builtins__', 'builtins',
    })
    
    safe_modules: Set[str] = field(default_factory=lambda: {
        'json', 're', 'math', 'datetime', 'time', 'calendar',
        'decimal', 'fractions', 'statistics', 'random',
        'string', 'textwrap', 'unicodedata',
        'itertools', 'functools', 'operator', 'collections',
        'heapq', 'bisect', 'array', 'copy', 'pprint',
        'enum', 'dataclasses', 'typing',
    })
    
    # Builtin-level security
    blocked_builtins: Set[str] = field(default_factory=lambda: {
        'open', 'input', 'compile', 'eval', 'exec',
        '__import__', 'breakpoint', 'exit', 'quit', 'help',
    })
    
    safe_builtins: Set[str] = field(default_factory=lambda: {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytearray', 'bytes',
        'callable', 'chr', 'classmethod', 'complex', 'delattr', 'dict',
        'dir', 'divmod', 'enumerate', 'filter', 'float', 'format',
        'frozenset', 'getattr', 'hasattr', 'hash', 'hex', 'id', 'int',
        'isinstance', 'issubclass', 'iter', 'len', 'list', 'locals',
        'map', 'max', 'min', 'next', 'object', 'oct', 'ord', 'pow',
        'print', 'property', 'range', 'repr', 'reversed', 'round',
        'set', 'setattr', 'slice', 'sorted', 'staticmethod', 'str',
        'sum', 'super', 'tuple', 'type', 'vars', 'zip',
        'Exception', 'ValueError', 'TypeError', 'KeyError', 'IndexError',
        'AttributeError', 'RuntimeError', 'StopIteration', 'AssertionError',
        'True', 'False', 'None', 'NotImplemented', 'Ellipsis',
        '__name__', '__doc__', '__package__', '__loader__', '__spec__',
    })
    
    def add_safe_module(self, module: str) -> None:
        """Add a module to the safe list."""
        self.safe_modules.add(module)
        self.blocked_modules.discard(module)  # Remove from blocked if present
    
    def remove_safe_module(self, module: str) -> None:
        """Remove a module from the safe list."""
        self.safe_modules.discard(module)
    
    def add_blocked_module(self, module: str) -> None:
        """Add a module to the blocked list."""
        self.blocked_modules.add(module)
        self.safe_modules.discard(module)  # Remove from safe if present
    
    def remove_blocked_module(self, module: str) -> None:
        """Remove a module from the blocked list."""
        self.blocked_modules.discard(module)
    
    def is_module_allowed(self, module: str) -> bool:
        """Check if a module is allowed."""
        base_module = module.split('.')[0]
        if base_module in self.blocked_modules:
            return False
        return base_module in self.safe_modules
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'blocked_modules': sorted(self.blocked_modules),
            'safe_modules': sorted(self.safe_modules),
            'blocked_builtins': sorted(self.blocked_builtins),
            'safe_builtins': sorted(self.safe_builtins),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecurityConfig':
        """Create config from dictionary."""
        return cls(
            blocked_modules=set(data.get('blocked_modules', [])),
            safe_modules=set(data.get('safe_modules', [])),
            blocked_builtins=set(data.get('blocked_builtins', [])),
            safe_builtins=set(data.get('safe_builtins', [])),
        )


@dataclass
class ExecutionConfig:
    """Execution environment configuration."""
    
    default_timeout: float = 5.0
    max_output_chars: int = 10000
    default_safe_mode: bool = False
    max_steps: int = 16  # Maximum execution steps for RLM loop
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'default_timeout': self.default_timeout,
            'max_output_chars': self.max_output_chars,
            'default_safe_mode': self.default_safe_mode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionConfig':
        """Create config from dictionary."""
        return cls(
            default_timeout=data.get('default_timeout', 5.0),
            max_output_chars=data.get('max_output_chars', 10000),
            default_safe_mode=data.get('default_safe_mode', False),
        )


@dataclass
class MonitoringConfig:
    """Monitoring and telemetry configuration."""
    
    enable_telemetry: bool = False
    log_level: str = 'INFO'
    export_format: str = 'json'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'enable_telemetry': self.enable_telemetry,
            'log_level': self.log_level,
            'export_format': self.export_format,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonitoringConfig':
        """Create config from dictionary."""
        return cls(
            enable_telemetry=data.get('enable_telemetry', False),
            log_level=data.get('log_level', 'INFO'),
            export_format=data.get('export_format', 'json'),
        )


class RLMConfig:
    """Main RLMKit configuration."""
    
    # Default config file locations (in priority order)
    CONFIG_SEARCH_PATHS = [
        './rlmkit_config.yaml',
        './rlmkit_config.json',
        '~/.rlmkit/config.yaml',
        '~/.rlmkit/config.json',
        '/etc/rlmkit/config.yaml',
        '/etc/rlmkit/config.json',
    ]
    
    def __init__(
        self,
        security: Optional[SecurityConfig] = None,
        execution: Optional[ExecutionConfig] = None,
        monitoring: Optional[MonitoringConfig] = None,
        llm: Optional[LLMConfig] = None,
    ):
        """
        Initialize RLMKit configuration.
        
        Args:
            security: Security configuration
            execution: Execution configuration
            monitoring: Monitoring configuration
            llm: LLM provider configuration
        """
        self.security = security or SecurityConfig()
        self.execution = execution or ExecutionConfig()
        self.monitoring = monitoring or MonitoringConfig()
        self.llm = llm or LLMConfig()
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'RLMConfig':
        """
        Load configuration from file.
        
        Search order:
        1. Explicit config_path parameter
        2. RLMKIT_CONFIG_PATH environment variable
        3. Default search paths (project, user home, system)
        
        Args:
            config_path: Optional explicit path to config file
            
        Returns:
            RLMConfig instance
        """
        # Try explicit path first
        if config_path:
            return cls._load_from_file(config_path)
        
        # Try environment variable
        env_path = os.getenv('RLMKIT_CONFIG_PATH')
        if env_path and Path(env_path).exists():
            return cls._load_from_file(env_path)
        
        # Try default search paths
        for path_str in cls.CONFIG_SEARCH_PATHS:
            path = Path(path_str).expanduser()
            if path.exists():
                return cls._load_from_file(str(path))
        
        # No config file found, use defaults
        return cls()
    
    @classmethod
    def _load_from_file(cls, filepath: str) -> 'RLMConfig':
        """Load configuration from a specific file."""
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        # Read file content
        with open(path, 'r') as f:
            content = f.read()
        
        # Parse based on extension
        if path.suffix in ['.yaml', '.yml']:
            data = yaml.safe_load(content)
        elif path.suffix == '.json':
            data = json.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        # Create config from data
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RLMConfig':
        """Create config from dictionary."""
        security = SecurityConfig.from_dict(data.get('security', {}))
        execution = ExecutionConfig.from_dict(data.get('execution', {}))
        monitoring = MonitoringConfig.from_dict(data.get('monitoring', {}))
        llm = LLMConfig.from_dict(data.get('llm', {}))
        
        return cls(
            security=security,
            execution=execution,
            monitoring=monitoring,
            llm=llm
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'security': self.security.to_dict(),
            'execution': self.execution.to_dict(),
            'monitoring': self.monitoring.to_dict(),
            'llm': self.llm.to_dict(),
        }
    
    def save(self, filepath: str) -> None:
        """
        Save configuration to file.
        
        Args:
            filepath: Path to save config file
        """
        path = Path(filepath)
        data = self.to_dict()
        
        # Save based on extension
        if path.suffix in ['.yaml', '.yml']:
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


# Global default configuration instance
_default_config: Optional[RLMConfig] = None


def get_default_config() -> RLMConfig:
    """
    Get the default configuration instance.
    
    Lazily loads configuration on first access.
    
    Returns:
        RLMConfig instance
    """
    global _default_config
    if _default_config is None:
        _default_config = RLMConfig.load()
    return _default_config


def set_default_config(config: RLMConfig) -> None:
    """
    Set the default configuration instance.
    
    Args:
        config: RLMConfig to use as default
    """
    global _default_config
    _default_config = config
