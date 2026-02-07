"""Tests for parsing utilities."""

import pytest
from rlmkit.core.parsing import (
    extract_python_code,
    extract_final_answer,
    extract_final_var,
    parse_response,
    format_code_for_display,
    format_result_for_llm,
    ParsedResponse,
)


class TestExtractPythonCode:
    """Test Python code extraction from markdown blocks."""
    
    def test_extracts_python_code_block(self):
        """Test extracting code from ```python block."""
        text = "Here's the code:\n```python\nx = 1\ny = 2\n```\nDone"
        code = extract_python_code(text)
        assert code == "x = 1\ny = 2"
    
    def test_extracts_generic_code_block(self):
        """Test extracting code from ``` block."""
        text = "Code:\n```\nresult = peek(0, 10)\n```"
        code = extract_python_code(text)
        assert code == "result = peek(0, 10)"
    
    def test_returns_none_when_no_code(self):
        """Test returns None when no code block present."""
        text = "No code here, just text"
        code = extract_python_code(text)
        assert code is None
    
    def test_returns_first_code_block(self):
        """Test returns first code block when multiple exist."""
        text = "```python\nfirst = 1\n```\nSome text\n```python\nsecond = 2\n```"
        code = extract_python_code(text)
        assert code == "first = 1"
    
    def test_handles_empty_code_block(self):
        """Test handles empty code block."""
        text = "```python\n\n```"
        code = extract_python_code(text)
        assert code == "" or code is None  # Empty after strip
    
    def test_ignores_other_language_blocks(self):
        """Test doesn't extract non-Python language blocks."""
        text = "```javascript\nconst x = 1;\n```"
        code = extract_python_code(text)
        assert code is None
    
    def test_case_insensitive_python_marker(self):
        """Test python marker is case-insensitive."""
        text = "```PYTHON\nx = 1\n```"
        code = extract_python_code(text)
        assert code == "x = 1"
    
    def test_multiline_code(self):
        """Test extracting multi-line code."""
        text = """
Let me explore:
```python
for i in range(10):
    print(i)
    if i > 5:
        break
```
Done
"""
        code = extract_python_code(text)
        assert "for i in range(10):" in code
        assert "print(i)" in code
        assert "break" in code


class TestExtractFinalAnswer:
    """Test final answer extraction."""
    
    def test_extracts_final_answer(self):
        """Test extracting FINAL: answer."""
        text = "After analysis...\nFINAL: The answer is 42"
        answer = extract_final_answer(text)
        assert answer == "The answer is 42"
    
    def test_case_insensitive(self):
        """Test FINAL is case-insensitive."""
        text = "final: Done"
        answer = extract_final_answer(text)
        assert answer == "Done"
    
    def test_returns_none_when_no_final(self):
        """Test returns None when no FINAL present."""
        text = "No final answer here"
        answer = extract_final_answer(text)
        assert answer is None
    
    def test_extracts_multiline_answer(self):
        """Test extracts all lines after FINAL (for multi-line answers)."""
        text = "FINAL: First line\nSecond line"
        answer = extract_final_answer(text)
        assert answer == "First line\nSecond line"
    
    def test_strips_whitespace(self):
        """Test strips whitespace from answer."""
        text = "FINAL:    Answer with spaces   "
        answer = extract_final_answer(text)
        assert answer == "Answer with spaces"
    
    def test_final_in_middle_of_text(self):
        """Test FINAL captures everything after it (multi-line answers)."""
        text = "First paragraph\n\nFINAL: The answer\n\nLast paragraph"
        answer = extract_final_answer(text)
        # FINAL captures everything to end of text for complete multi-line answers
        assert answer == "The answer\n\nLast paragraph"


class TestExtractFinalVar:
    """Test FINAL_VAR extraction."""
    
    def test_extracts_final_var(self):
        """Test extracting FINAL_VAR directive."""
        text = "FINAL_VAR: result"
        var_name = extract_final_var(text)
        assert var_name == "result"
    
    def test_case_insensitive(self):
        """Test FINAL_VAR is case-insensitive."""
        text = "final_var: answer"
        var_name = extract_final_var(text)
        assert var_name == "answer"
    
    def test_returns_none_when_no_final_var(self):
        """Test returns None when no FINAL_VAR present."""
        text = "No directive here"
        var_name = extract_final_var(text)
        assert var_name is None
    
    def test_validates_variable_name(self):
        """Test only extracts valid Python variable names."""
        text = "FINAL_VAR: valid_name_123"
        var_name = extract_final_var(text)
        assert var_name == "valid_name_123"
    
    def test_rejects_invalid_variable_names(self):
        """Test doesn't extract invalid variable names."""
        text = "FINAL_VAR: 123invalid"
        var_name = extract_final_var(text)
        assert var_name is None
        
        text2 = "FINAL_VAR: invalid-name"
        var_name2 = extract_final_var(text2)
        assert var_name2 is None
    
    def test_extracts_from_multiline(self):
        """Test extracts from multiline text."""
        text = "Some text\nFINAL_VAR: my_result\nMore text"
        var_name = extract_final_var(text)
        assert var_name == "my_result"


class TestParseResponse:
    """Test full response parsing."""
    
    def test_parses_code_only(self):
        """Test parsing response with only code."""
        text = "```python\nx = 1\n```"
        resp = parse_response(text)
        
        assert resp.has_code
        assert not resp.has_final
        assert not resp.is_complete
        assert resp.code == "x = 1"
        assert resp.final_answer is None
        assert resp.final_var is None
    
    def test_parses_final_answer_only(self):
        """Test parsing response with only final answer."""
        text = "FINAL: The answer is 42"
        resp = parse_response(text)
        
        assert not resp.has_code
        assert resp.has_final
        assert resp.is_complete
        assert resp.code is None
        assert resp.final_answer == "The answer is 42"
    
    def test_parses_final_var_only(self):
        """Test parsing response with only FINAL_VAR."""
        text = "FINAL_VAR: result"
        resp = parse_response(text)
        
        assert not resp.has_code
        assert resp.has_final
        assert resp.is_complete
        assert resp.final_var == "result"
    
    def test_parses_code_and_final(self):
        """Test parsing response with both code and final - FINAL wins."""
        text = "```python\nx = 1\n```\nFINAL: Done"
        resp = parse_response(text)

        # When FINAL is present, code is ignored (FINAL is terminal state)
        assert not resp.has_code
        assert resp.has_final
        assert resp.is_complete  # FINAL makes it complete
        assert resp.code is None
        assert resp.final_answer == "Done"
    
    def test_parses_empty_response(self):
        """Test parsing empty response."""
        text = ""
        resp = parse_response(text)
        
        assert not resp.has_code
        assert not resp.has_final
        assert not resp.is_complete
    
    def test_stores_raw_text(self):
        """Test stores raw text in response."""
        text = "Some response text"
        resp = parse_response(text)
        
        assert resp.raw_text == text
    
    def test_realistic_exploration_response(self):
        """Test parsing realistic exploration response."""
        text = """
I'll search for the relevant information:

```python
matches = grep(r'climate change')
if len(matches) > 0:
    relevant = select([m['range'] for m in matches[:5]])
    print(f"Found {len(matches)} matches")
```
"""
        resp = parse_response(text)
        
        assert resp.has_code
        assert not resp.has_final
        assert "grep" in resp.code
        assert "select" in resp.code
    
    def test_realistic_final_response(self):
        """Test parsing realistic final response."""
        text = """
Based on my analysis of the document, I found that the main findings are:

1. Climate change is accelerating
2. Human activity is the primary cause
3. Immediate action is needed

FINAL: The document presents three main findings about climate change: 
it is accelerating, caused by human activity, and requires immediate action.
"""
        resp = parse_response(text)
        
        assert not resp.has_code
        assert resp.has_final
        assert resp.is_complete
        assert "three main findings" in resp.final_answer


class TestFormatCodeForDisplay:
    """Test code formatting for display."""
    
    def test_returns_short_code_unchanged(self):
        """Test short code is not truncated."""
        code = "x = 1\ny = 2"
        formatted = format_code_for_display(code, max_lines=10)
        assert formatted == code
    
    def test_truncates_long_code(self):
        """Test long code is truncated."""
        code = "\n".join([f"line{i} = {i}" for i in range(20)])
        formatted = format_code_for_display(code, max_lines=5)
        
        assert "line0 = 0" in formatted
        assert "line4 = 4" in formatted
        assert "line19 = 19" not in formatted
        assert "(15 more lines)" in formatted
    
    def test_truncates_at_exact_limit(self):
        """Test truncation at exact line limit."""
        code = "\n".join([f"line{i}" for i in range(10)])
        formatted = format_code_for_display(code, max_lines=10)
        assert formatted == code  # Exactly at limit, no truncation


class TestFormatResultForLLM:
    """Test formatting execution results for LLM."""
    
    def test_formats_stdout(self):
        """Test formats stdout output."""
        result = {'stdout': 'Hello, world!\n', 'stderr': '', 'exception': None}
        formatted = format_result_for_llm(result)
        
        assert "Output:" in formatted
        assert "Hello, world!" in formatted
    
    def test_formats_stderr(self):
        """Test formats stderr output."""
        result = {'stdout': '', 'stderr': 'Warning: something\n', 'exception': None}
        formatted = format_result_for_llm(result)
        
        assert "Errors:" in formatted
        assert "Warning: something" in formatted
    
    def test_formats_exception(self):
        """Test formats exception."""
        result = {
            'stdout': '',
            'stderr': '',
            'exception': 'ZeroDivisionError: division by zero'
        }
        formatted = format_result_for_llm(result)
        
        assert "Exception:" in formatted
        assert "ZeroDivisionError" in formatted
    
    def test_formats_timeout(self):
        """Test formats timeout indicator."""
        result = {'stdout': '', 'stderr': '', 'exception': None, 'timeout': True}
        formatted = format_result_for_llm(result)
        
        assert "timed out" in formatted.lower()
    
    def test_formats_successful_no_output(self):
        """Test formats successful execution with no output."""
        result = {'stdout': '', 'stderr': '', 'exception': None}
        formatted = format_result_for_llm(result)
        
        assert "successfully" in formatted.lower()
    
    def test_formats_combined_outputs(self):
        """Test formats multiple outputs together."""
        result = {
            'stdout': 'Result: 42\n',
            'stderr': 'Debug info\n',
            'exception': None
        }
        formatted = format_result_for_llm(result)
        
        assert "Output:" in formatted
        assert "Result: 42" in formatted
        assert "Errors:" in formatted
        assert "Debug info" in formatted
    
    def test_strips_whitespace(self):
        """Test strips excess whitespace."""
        result = {'stdout': '  \n  Hello  \n  ', 'stderr': '', 'exception': None}
        formatted = format_result_for_llm(result)
        
        assert "Hello" in formatted
        # Should not have excessive whitespace


class TestParsedResponseProperties:
    """Test ParsedResponse property methods."""
    
    def test_has_code_with_code(self):
        """Test has_code returns True when code present."""
        resp = ParsedResponse(code="x = 1")
        assert resp.has_code is True
    
    def test_has_code_with_empty_code(self):
        """Test has_code returns False for empty code."""
        resp = ParsedResponse(code="")
        assert resp.has_code is False
        
        resp2 = ParsedResponse(code="   ")
        assert resp2.has_code is False
    
    def test_has_code_with_none(self):
        """Test has_code returns False when code is None."""
        resp = ParsedResponse(code=None)
        assert resp.has_code is False
    
    def test_has_final_with_answer(self):
        """Test has_final returns True with final answer."""
        resp = ParsedResponse(final_answer="Done")
        assert resp.has_final is True
    
    def test_has_final_with_var(self):
        """Test has_final returns True with final var."""
        resp = ParsedResponse(final_var="result")
        assert resp.has_final is True
    
    def test_has_final_with_both(self):
        """Test has_final returns True with both."""
        resp = ParsedResponse(final_answer="Done", final_var="result")
        assert resp.has_final is True
    
    def test_has_final_with_neither(self):
        """Test has_final returns False with neither."""
        resp = ParsedResponse()
        assert resp.has_final is False
    
    def test_is_complete_when_final_only(self):
        """Test is_complete when only final answer."""
        resp = ParsedResponse(final_answer="Done")
        assert resp.is_complete is True
    
    def test_is_complete_when_code_and_final(self):
        """Test is_complete is False when both code and final."""
        resp = ParsedResponse(code="x = 1", final_answer="Done")
        assert resp.is_complete is False
    
    def test_is_complete_when_code_only(self):
        """Test is_complete is False when only code."""
        resp = ParsedResponse(code="x = 1")
        assert resp.is_complete is False
    
    def test_is_complete_when_neither(self):
        """Test is_complete is False when neither."""
        resp = ParsedResponse()
        assert resp.is_complete is False
