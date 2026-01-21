# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Prompt templates for RLM system prompts.

This module provides templated system prompts that can be customized
and versioned. Prompts are loaded from YAML files for easy editing and
version control.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import importlib.resources


# Cache for loaded templates
_template_cache: Dict[str, str] = {}


def get_default_system_prompt(version: str = "1.0") -> str:
    """
    Get the default system prompt template from versioned template file.
    
    Templates are stored as discoverable files: system_prompt_v{major}_{minor}.txt
    
    Args:
        version: Version of the prompt template to use (e.g., "1.0", "2.0")
        
    Returns:
        System prompt template string
        
    Raises:
        ValueError: If version is not supported
        FileNotFoundError: If template file doesn't exist
        
    Example:
        >>> template = get_default_system_prompt("1.0")
        >>> # Loads from system_prompt_v1_0.txt
    """
    # Check cache first
    cache_key = f"system_prompt_v{version}"
    if cache_key in _template_cache:
        return _template_cache[cache_key]
    
    # Convert version to filename format: "1.0" -> "v1_0"
    version_str = version.replace(".", "_")
    filename = f"system_prompt_v{version_str}.txt"
    
    # Try to load from package resources
    try:
        import importlib.resources as pkg_resources
        try:
            # Python 3.9+
            files = pkg_resources.files('rlmkit.prompts')
            template_file = files / filename
            template = template_file.read_text()
        except AttributeError:
            # Python 3.7-3.8 fallback
            template = pkg_resources.read_text('rlmkit.prompts', filename)
        
        # Cache it
        _template_cache[cache_key] = template
        return template
        
    except (FileNotFoundError, ModuleNotFoundError):
        raise ValueError(
            f"Unsupported system prompt version: {version}. "
            f"Template file not found: {filename}"
        )


def format_system_prompt(
    template: Optional[str] = None,
    prompt_length: int = 0,
    version: str = "1.0",
    **kwargs: Any
) -> str:
    """
    Format a system prompt template with given parameters.
    
    Args:
        template: Custom template string. If None, uses default for version
        prompt_length: Length of the content being analyzed
        version: Version of default template to use if template is None
        **kwargs: Additional template variables
        
    Returns:
        Formatted system prompt string
        
    Example:
        >>> prompt = format_system_prompt(prompt_length=12345)
        >>> assert "12,345" in prompt
        
        >>> custom = "Content length: {prompt_length}"
        >>> prompt = format_system_prompt(template=custom, prompt_length=1000)
        >>> assert "1000" in prompt
    """
    if template is None:
        template = get_default_system_prompt(version=version)
    
    # Format with provided kwargs
    format_vars = {"prompt_length": prompt_length, **kwargs}
    
    try:
        return template.format(**format_vars)
    except KeyError as e:
        raise ValueError(f"Missing template variable: {e}")


def load_prompt_from_file(filepath: Path) -> str:
    """
    Load a prompt template from a file.
    
    Supports:
    - .txt files: Plain text templates
    - .md files: Markdown templates
    - .yaml/.yml files: YAML with 'template' key
    
    Args:
        filepath: Path to template file
        
    Returns:
        Template string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Template file not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    if suffix in ['.txt', '.md']:
        return filepath.read_text()
    elif suffix in ['.yaml', '.yml']:
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        if 'template' not in data:
            raise ValueError(f"YAML file must have 'template' key: {filepath}")
        return data['template']
    else:
        raise ValueError(f"Unsupported template file format: {suffix}")


def save_prompt_to_file(template: str, filepath: Path) -> None:
    """
    Save a prompt template to a file.
    
    Args:
        template: Template string to save
        filepath: Path where to save the template
    """
    suffix = filepath.suffix.lower()
    
    if suffix in ['.txt', '.md']:
        filepath.write_text(template)
    elif suffix in ['.yaml', '.yml']:
        import yaml
        data = {'template': template, 'version': '1.0'}
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    else:
        raise ValueError(f"Unsupported template file format: {suffix}")
