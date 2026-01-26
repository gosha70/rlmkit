# Copyright (c) EGOGE - All Rights Reserved.
# This software may be used and distributed according to the terms of the MIT license.

"""JSON action protocol for deterministic RLM execution."""

import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Literal
from enum import Enum


class ActionType(str, Enum):
    """Valid action types in the protocol."""
    INSPECT = "inspect"
    FINAL = "final"
    SUBCALL = "subcall"


class ToolType(str, Enum):
    """Valid tool types for inspect actions."""
    GREP = "grep"
    PEEK = "peek"
    SELECT = "select"
    CHUNK = "chunk"


@dataclass
class InspectAction:
    """Action to inspect content using tools."""
    tool: str  # grep, peek, select, chunk
    args: Dict[str, Any]
    note: Optional[str] = None
    
    def validate(self) -> None:
        """Validate the inspect action."""
        if self.tool not in ["grep", "peek", "select", "chunk"]:
            raise ValueError(f"Invalid tool: {self.tool}. Must be one of: grep, peek, select, chunk")
        
        if not isinstance(self.args, dict):
            raise ValueError(f"args must be a dictionary, got: {type(self.args)}")


@dataclass
class FinalAction:
    """Action to finalize with an answer."""
    answer: str
    confidence: Optional[float] = None
    citations: Optional[List[str]] = None
    
    def validate(self) -> None:
        """Validate the final action."""
        if not self.answer or not isinstance(self.answer, str):
            raise ValueError("answer must be a non-empty string")
        
        if self.answer.strip() == "":
            raise ValueError("answer cannot be empty or whitespace only")
        
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)):
                raise ValueError(f"confidence must be a number, got: {type(self.confidence)}")
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(f"confidence must be between 0.0 and 1.0, got: {self.confidence}")


@dataclass
class SubcallAction:
    """Action to make a recursive subcall."""
    prompt: str
    query: str
    note: Optional[str] = None
    
    def validate(self) -> None:
        """Validate the subcall action."""
        if not self.prompt or not isinstance(self.prompt, str):
            raise ValueError("prompt must be a non-empty string")
        if not self.query or not isinstance(self.query, str):
            raise ValueError("query must be a non-empty string")


class ParseError(Exception):
    """Error parsing action from LLM response."""
    pass


def extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the first JSON object from text.
    
    Looks for {...} and attempts to parse it as JSON.
    Rejects if there's non-whitespace content after the JSON.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON dict or None if not found
        
    Raises:
        ParseError: If JSON is found but invalid, or has trailing content
    """
    text = text.strip()
    
    if not text:
        return None
    
    # Find first '{' and matching '}'
    start = text.find('{')
    if start == -1:
        return None
    
    # Find matching closing brace
    brace_count = 0
    end = start
    
    for i in range(start, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    
    if brace_count != 0:
        raise ParseError("Unclosed JSON object - missing closing brace")
    
    # Extract JSON string
    json_str = text[start:end]
    
    # Check for trailing non-whitespace content
    trailing = text[end:].strip()
    if trailing:
        raise ParseError(f"Found trailing content after JSON: {trailing[:50]}...")
    
    # Parse JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ParseError(f"Invalid JSON: {e}")


def parse_action(text: str) -> tuple[Literal["inspect", "final", "subcall"], Any]:
    """
    Parse action from LLM response.
    
    Expects a single JSON object with a "type" field.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Tuple of (action_type, action_object)
        
    Raises:
        ParseError: If parsing fails or validation fails
        
    Examples:
        >>> parse_action('{"type": "final", "answer": "42"}')
        ('final', FinalAction(answer='42', confidence=None, citations=None))
        
        >>> parse_action('{"type": "inspect", "tool": "grep", "args": {"pattern": "test"}}')
        ('inspect', InspectAction(tool='grep', args={'pattern': 'test'}, note=None))
    """
    # Extract JSON
    try:
        json_obj = extract_first_json_object(text)
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"Failed to extract JSON: {e}")
    
    if json_obj is None:
        raise ParseError("No JSON object found in response")
    
    # Validate has 'type' field
    if "type" not in json_obj:
        raise ParseError("JSON object missing required 'type' field")
    
    action_type = json_obj["type"]
    
    # Parse based on type
    if action_type == "inspect":
        return _parse_inspect_action(json_obj)
    elif action_type == "final":
        return _parse_final_action(json_obj)
    elif action_type == "subcall":
        return _parse_subcall_action(json_obj)
    else:
        raise ParseError(f"Unknown action type: {action_type}. Must be: inspect, final, or subcall")


def _parse_inspect_action(json_obj: Dict[str, Any]) -> tuple[Literal["inspect"], InspectAction]:
    """Parse inspect action from JSON."""
    # Required fields
    if "tool" not in json_obj:
        raise ParseError("inspect action missing required 'tool' field")
    if "args" not in json_obj:
        raise ParseError("inspect action missing required 'args' field")
    
    # Optional fields
    note = json_obj.get("note")
    
    # Create and validate
    action = InspectAction(
        tool=json_obj["tool"],
        args=json_obj["args"],
        note=note
    )
    
    try:
        action.validate()
    except ValueError as e:
        raise ParseError(f"Invalid inspect action: {e}")
    
    return ("inspect", action)


def _parse_final_action(json_obj: Dict[str, Any]) -> tuple[Literal["final"], FinalAction]:
    """Parse final action from JSON."""
    # Required fields
    if "answer" not in json_obj:
        raise ParseError("final action missing required 'answer' field")
    
    # Optional fields
    confidence = json_obj.get("confidence")
    citations = json_obj.get("citations")
    
    # Create and validate
    action = FinalAction(
        answer=json_obj["answer"],
        confidence=confidence,
        citations=citations
    )
    
    try:
        action.validate()
    except ValueError as e:
        raise ParseError(f"Invalid final action: {e}")
    
    return ("final", action)


def _parse_subcall_action(json_obj: Dict[str, Any]) -> tuple[Literal["subcall"], SubcallAction]:
    """Parse subcall action from JSON."""
    # Required fields
    if "prompt" not in json_obj:
        raise ParseError("subcall action missing required 'prompt' field")
    if "query" not in json_obj:
        raise ParseError("subcall action missing required 'query' field")
    
    # Optional fields
    note = json_obj.get("note")
    
    # Create and validate
    action = SubcallAction(
        prompt=json_obj["prompt"],
        query=json_obj["query"],
        note=note
    )
    
    try:
        action.validate()
    except ValueError as e:
        raise ParseError(f"Invalid subcall action: {e}")
    
    return ("subcall", action)


def validate_final_answer_quality(answer: str, query: str) -> Optional[str]:
    """
    Validate the quality of a final answer.
    
    Rejects answers that:
    - Claim impossibility without workarounds
    - Assert APIs don't exist for known frameworks
    
    Args:
        answer: The final answer text
        query: The original user query
        
    Returns:
        Error message if validation fails, None if passes
    """
    answer_lower = answer.lower()
    query_lower = query.lower()
    
    # Check for impossibility claims
    impossibility_patterns = [
        "not possible",
        "no way",
        "cannot be done",
        "doesn't exist",
        "does not exist",
        "impossible to",
        "can't be done",
        "unable to"
    ]
    
    has_impossibility = any(pattern in answer_lower for pattern in impossibility_patterns)
    
    if has_impossibility:
        # Check for workaround keywords
        workaround_patterns = [
            "workaround",
            "alternative",
            "instead",
            "however",
            "you can",
            "try",
            "consider",
            "css",
            "column",
            "container",
            "layout",
            "style"
        ]
        
        has_workaround = any(pattern in answer_lower for pattern in workaround_patterns)
        
        if not has_workaround:
            return (
                "Quality gate: Answer contains impossibility claim without workaround. "
                "Please provide at least one alternative approach or solution."
            )
    
    # Check for API existence claims when query mentions known frameworks
    if "st." in query_lower or "streamlit" in query_lower:
        if "doesn't exist" in answer_lower or "does not exist" in answer_lower:
            if "api" in answer_lower or "function" in answer_lower or "method" in answer_lower:
                return (
                    "Quality gate: Claiming API doesn't exist for known framework. "
                    "Please verify or provide workaround solutions."
                )
    
    return None
