# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""Content navigation tools for exploring large text."""

import re
from typing import List, Optional, Tuple, Union


def peek(content: str, start: int = 0, end: Optional[int] = None, max_chars: int = 10000) -> str:
    """
    View a section of content by character positions.
    
    Args:
        content: The content to peek into
        start: Starting character position (0-indexed)
        end: Ending character position (exclusive). If None, goes to content end
        max_chars: Maximum characters to return (safety limit)
        
    Returns:
        Substring from start to end (limited by max_chars)
        
    Examples:
        >>> peek(P, 0, 100)  # First 100 chars
        >>> peek(P, 1000, 2000)  # Chars 1000-2000
        >>> peek(P, -100)  # Last 100 chars (Python slice semantics)
    """
    # Handle None end
    if end is None:
        end = len(content)
    
    # Bounds checking
    content_len = len(content)
    if start < 0:
        start = max(0, content_len + start)  # Handle negative indexing
    if end < 0:
        end = max(0, content_len + end)
    
    # Clamp to content bounds
    start = max(0, min(start, content_len))
    end = max(start, min(end, content_len))
    
    # Extract substring
    result = content[start:end]
    
    # Apply max_chars limit
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n... (truncated, showing {max_chars}/{len(content[start:end])} chars)"
    
    return result


def grep(
    content: str,
    pattern: str,
    context_lines: int = 2,
    max_matches: int = 100,
    ignore_case: bool = False,
    use_regex: bool = False
) -> str:
    """
    Search content for pattern and return matches with context.
    
    Args:
        content: The content to search
        pattern: Search pattern (literal string or regex)
        context_lines: Number of lines to show before/after each match
        max_matches: Maximum number of matches to return
        ignore_case: Case-insensitive search
        use_regex: Treat pattern as regex (default: literal string)
        
    Returns:
        Formatted string with matches and line numbers
        
    Examples:
        >>> grep(P, "error")  # Find "error" with 2 lines context
        >>> grep(P, r"\\d{4}-\\d{2}-\\d{2}", use_regex=True)  # Find dates
        >>> grep(P, "warning", context_lines=5, ignore_case=True)
    """
    lines = content.splitlines()
    
    # Prepare pattern
    if not use_regex:
        pattern = re.escape(pattern)  # Escape special chars for literal match
    
    flags = re.IGNORECASE if ignore_case else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"
    
    # Find matching lines
    matches: List[Tuple[int, str]] = []
    for line_no, line in enumerate(lines, start=1):
        if regex.search(line):
            matches.append((line_no, line))
            if len(matches) >= max_matches:
                break
    
    if not matches:
        return f"No matches found for pattern: {pattern}"
    
    # Build output with context
    result_lines: List[str] = []
    result_lines.append(f"Found {len(matches)} match(es):\n")
    
    for match_line_no, match_line in matches:
        # Calculate context range
        start_line = max(1, match_line_no - context_lines)
        end_line = min(len(lines), match_line_no + context_lines)
        
        # Add separator
        result_lines.append(f"\n{'='*60}")
        result_lines.append(f"Match at line {match_line_no}:")
        result_lines.append(f"{'='*60}")
        
        # Add context and match
        for line_no in range(start_line, end_line + 1):
            line = lines[line_no - 1]  # Convert to 0-indexed
            prefix = ">>> " if line_no == match_line_no else "    "
            result_lines.append(f"{prefix}{line_no:4d}: {line}")
    
    return "\n".join(result_lines)


def chunk(
    content: str,
    size: int = 1000,
    overlap: int = 0,
    by: str = 'chars',
    max_chunks: int = 100
) -> List[str]:
    """
    Split content into chunks for systematic exploration.
    
    Args:
        content: The content to chunk
        size: Chunk size (chars or lines depending on 'by')
        overlap: Overlap between chunks (0 = no overlap)
        by: 'chars' or 'lines' - chunking strategy
        max_chunks: Maximum number of chunks to return
        
    Returns:
        List of content chunks
        
    Examples:
        >>> chunks = chunk(P, size=1000, by='chars')  # 1000-char chunks
        >>> chunks = chunk(P, size=50, by='lines')    # 50-line chunks
        >>> chunks = chunk(P, size=500, overlap=50)   # With 50-char overlap
    """
    if by not in ('chars', 'lines'):
        raise ValueError(f"Invalid 'by' parameter: {by}. Must be 'chars' or 'lines'")
    
    if size <= 0:
        raise ValueError(f"Size must be positive, got: {size}")
    
    if overlap < 0 or overlap >= size:
        raise ValueError(f"Overlap must be 0 <= overlap < size, got: {overlap}")
    
    chunks: List[str] = []
    
    if by == 'chars':
        # Character-based chunking
        step = size - overlap
        for i in range(0, len(content), step):
            chunk_text = content[i:i + size]
            chunks.append(chunk_text)
            
            if len(chunks) >= max_chunks:
                break
                
    else:  # by == 'lines'
        # Line-based chunking
        lines = content.splitlines(keepends=True)
        step = size - overlap
        
        for i in range(0, len(lines), step):
            chunk_lines = lines[i:i + size]
            chunk_text = ''.join(chunk_lines)
            chunks.append(chunk_text)
            
            if len(chunks) >= max_chunks:
                break
    
    return chunks


def select(content: str, ranges: List[Tuple[int, int]], max_chars: int = 10000) -> str:
    """
    Extract and combine multiple ranges from content.
    
    Args:
        content: The content to select from
        ranges: List of (start, end) tuples (character positions)
        max_chars: Maximum total characters to return
        
    Returns:
        Combined content from all ranges, separated by markers
        
    Examples:
        >>> select(P, [(0, 100), (500, 600)])  # First and middle sections
        >>> select(P, [(0, 50), (-50, None)])  # Beginning and end
    """
    if not ranges:
        return ""
    
    parts: List[str] = []
    total_chars = 0
    
    for i, (start, end) in enumerate(ranges):
        # Use peek for each range (handles bounds checking)
        part = peek(content, start, end, max_chars=max_chars - total_chars)
        
        # Check if truncated
        if total_chars + len(part) > max_chars:
            parts.append(f"... (output limit reached at range {i+1}/{len(ranges)})")
            break
        
        # Add range marker
        if i > 0:
            parts.append(f"\n{'='*60}")
            parts.append(f"Range {i+1}: chars {start} to {end}")
            parts.append(f"{'='*60}\n")
        
        parts.append(part)
        total_chars += len(part)
    
    return "".join(parts)
