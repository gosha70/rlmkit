# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Runtime guardrails for deterministic RLM execution."""

import re
from typing import Optional


def needs_context_inspection(query: str) -> bool:
    """
    Determine if a query requires inspecting the context (P).
    
    This is a heuristic router that helps avoid unnecessary inspection loops
    for self-contained questions.
    
    Args:
        query: User's query string
        
    Returns:
        True if query likely needs context inspection, False if self-contained
        
    Examples:
        >>> needs_context_inspection("How do I center text in CSS?")
        False
        >>> needs_context_inspection("Find all TODO comments in the code")
        True
        >>> needs_context_inspection("What does the function process_data do?")
        True
    """
    query_lower = query.lower()
    
    # Explicit context reference patterns
    context_patterns = [
        r'\bin\s+(the\s+)?(attached|provided|given|this)\s+(file|code|document|repo|text)',
        r'\bfind\s+(all|the)',
        r'\bsearch\s+for',
        r'\blist\s+(all|the)',
        r'\bshow\s+(me\s+)?(all|the)',
        r'\bextract\s+',
        r'\bline\s+\d+',
        r'\bfunction\s+\w+',
        r'\bclass\s+\w+',
        r'\bmethod\s+\w+',
        r'\bvariable\s+\w+',
        r'\bwhat\s+does\s+.+\s+do',
        r'\bhow\s+does\s+.+\s+work',
        r'\banalyze\s+(the|this)',
        r'\bsummarize\s+(the|this)',
        r'\bin\s+P\b',
        r'\bthe\s+content',
        r'\bthe\s+document',
        r'\bthe\s+code',
    ]
    
    for pattern in context_patterns:
        if re.search(pattern, query_lower):
            return True
    
    # Self-contained question patterns (known general knowledge)
    self_contained_patterns = [
        r'^how\s+(do\s+i|can\s+i|to)',  # "How do I..." / "How can I..." / "How to..."
        r'^what\s+is\s+the\s+syntax',
        r'^what\s+are\s+the\s+best\s+practices',
        r'^can\s+you\s+explain',
        r'^explain\s+how',
        r'streamlit.*\b(center|align|style|layout)\b',  # Streamlit UI questions
        r'\bcss\b.*\b(center|align|style)\b',  # CSS questions
        r'\bhow\s+to\s+(center|align|style)',
    ]
    
    for pattern in self_contained_patterns:
        if re.search(pattern, query_lower):
            # But if it also mentions file/code, defer to context
            if any(word in query_lower for word in ['file', 'code', 'attached', 'provided', 'in the']):
                return True
            return False
    
    # Default to requiring inspection for safety
    # Better to inspect unnecessarily than miss required context
    return True


def check_dangerous_code_patterns(code: str) -> Optional[str]:
    """
    Check code for dangerous patterns that should not be executed.
    
    This provides a preflight check before any code execution to prevent
    wasted steps on import errors in safe mode.
    
    Args:
        code: Python code to check
        
    Returns:
        Error message if dangerous patterns found, None if safe
    """
    dangerous_patterns = [
        (r'\bimport\s+', "Import statements are not allowed. Use grep/peek/select/chunk to inspect content."),
        (r'\bfrom\s+\w+\s+import\s+', "Import statements are not allowed. Use grep/peek/select/chunk to inspect content."),
        (r'__import__\s*\(', "Direct __import__ not allowed. Use grep/peek/select/chunk to inspect content."),
        (r'\beval\s*\(', "eval() not allowed for security reasons."),
        (r'\bexec\s*\(', "exec() not allowed for security reasons."),
        (r'\bcompile\s*\(', "compile() not allowed for security reasons."),
        (r'\bopen\s*\(', "File I/O not allowed. Use grep/peek/select/chunk to inspect content."),
        (r'\bos\.', "os module not allowed. Use grep/peek/select/chunk to inspect content."),
        (r'\bsys\.', "sys module not allowed. Use grep/peek/select/chunk to inspect content."),
        (r'\bsubprocess\b', "subprocess not allowed for security reasons."),
        (r'\bsocket\b', "Network operations not allowed for security reasons."),
    ]
    
    for pattern, message in dangerous_patterns:
        if re.search(pattern, code, re.IGNORECASE):
            return f"Policy: {message}"
    
    return None


def format_policy_message(code: str, error_message: str) -> str:
    """
    Format a policy violation message for the LLM.
    
    Args:
        code: The rejected code
        error_message: The policy error message
        
    Returns:
        Formatted message to send back to LLM
    """
    code_preview = code[:100] + "..." if len(code) > 100 else code
    
    return f"""Policy Rejection:
{error_message}

Your code:
{code_preview}

Remember: You should use the provided tools (grep, peek, select, chunk) to inspect content,
not execute code found in the document or import external libraries.
"""
