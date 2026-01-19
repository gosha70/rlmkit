"""Tests for content navigation tools."""

import pytest
from rlmkit.envs import PyReplEnv


# Sample content for testing
SAMPLE_TEXT = """Line 1: Introduction to the document
Line 2: This is a test file
Line 3: It contains multiple lines
Line 4: With various patterns
Line 5: ERROR: Something went wrong
Line 6: WARNING: Check this
Line 7: INFO: All systems operational
Line 8: ERROR: Another error occurred
Line 9: Final line of the document
Line 10: End of file"""

LONG_TEXT = "x" * 50000  # 50K chars for truncation tests


class TestPeekTool:
    """Test peek() content viewing tool."""
    
    def test_peek_basic(self):
        """Test basic peek functionality."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("print(peek(P, 0, 10))")
        assert result['exception'] is None
        assert "Line 1:" in result['stdout']
    
    def test_peek_middle_section(self):
        """Test peeking middle section."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("text = peek(P, 20, 50); print(len(text))")
        assert result['exception'] is None
        assert '30' in result['stdout']  # 50-20 = 30 chars
    
    def test_peek_to_end(self):
        """Test peek to end when end=None."""
        env = PyReplEnv()
        env.set_content("12345")
        
        result = env.execute("print(peek(P, 2))")  # From char 2 to end
        assert result['exception'] is None
        assert '345' in result['stdout']
    
    def test_peek_negative_index(self):
        """Test peek with negative indexing."""
        env = PyReplEnv()
        env.set_content("0123456789")
        
        result = env.execute("print(peek(P, -5))")  # Last 5 chars
        assert result['exception'] is None
        assert '56789' in result['stdout']
    
    def test_peek_truncation(self):
        """Test peek truncates large output."""
        env = PyReplEnv()
        env.set_content(LONG_TEXT)
        
        result = env.execute("text = peek(P, 0, 50000, max_chars=100); print('done')")
        assert result['exception'] is None
        assert 'done' in result['stdout']


class TestGrepTool:
    """Test grep() search tool."""
    
    def test_grep_basic(self):
        """Test basic grep functionality."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("print(grep(P, 'ERROR'))")
        assert result['exception'] is None
        assert 'Found 2 match(es)' in result['stdout']
        assert 'Line 5' in result['stdout']
        assert 'Line 8' in result['stdout']
    
    def test_grep_no_matches(self):
        """Test grep when pattern not found."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("print(grep(P, 'NOTFOUND'))")
        assert result['exception'] is None
        assert 'No matches found' in result['stdout']
    
    def test_grep_with_context(self):
        """Test grep with context lines."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("print(grep(P, 'ERROR', context_lines=1))")
        assert result['exception'] is None
        # Should show line before and after
        assert 'Line 4' in result['stdout']  # Before Line 5
        assert 'Line 6' in result['stdout']  # After Line 5
    
    def test_grep_case_insensitive(self):
        """Test case-insensitive grep."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("print(grep(P, 'error', ignore_case=True))")
        assert result['exception'] is None
        assert 'Found 2 match(es)' in result['stdout']
    
    def test_grep_with_regex(self):
        """Test grep with regex pattern."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        # Find lines with "Line" followed by digits
        result = env.execute("print(grep(P, r'Line \\d+', use_regex=True))")
        assert result['exception'] is None
        assert 'Found' in result['stdout']
    
    def test_grep_max_matches(self):
        """Test grep max_matches limit."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("print(grep(P, 'Line', max_matches=3))")
        assert result['exception'] is None
        assert 'Found 3 match(es)' in result['stdout']


class TestChunkTool:
    """Test chunk() splitting tool."""
    
    def test_chunk_by_chars(self):
        """Test chunking by characters."""
        env = PyReplEnv()
        env.set_content("0123456789")
        
        result = env.execute("chunks = chunk(P, size=3, by='chars'); print(len(chunks))")
        assert result['exception'] is None
        assert '4' in result['stdout']  # 10 chars / 3 = 4 chunks (rounded up)
    
    def test_chunk_by_lines(self):
        """Test chunking by lines."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("chunks = chunk(P, size=3, by='lines'); print(len(chunks))")
        assert result['exception'] is None
        # SAMPLE_TEXT has 10 lines, so 3-line chunks = 4 chunks
        assert '4' in result['stdout']
    
    def test_chunk_with_overlap(self):
        """Test chunking with overlap."""
        env = PyReplEnv()
        env.set_content("0123456789")
        
        result = env.execute("""
chunks = chunk(P, size=5, overlap=2, by='chars')
print(f'chunks: {len(chunks)}')
print(f'first: {chunks[0]}')
print(f'second: {chunks[1]}')
""")
        assert result['exception'] is None
        assert '01234' in result['stdout']  # First chunk
        assert '34567' in result['stdout']  # Second chunk (overlap of 2)
    
    def test_chunk_max_chunks(self):
        """Test chunk max_chunks limit."""
        env = PyReplEnv()
        env.set_content("x" * 1000)
        
        result = env.execute("chunks = chunk(P, size=10, max_chunks=5); print(len(chunks))")
        assert result['exception'] is None
        assert '5' in result['stdout']
    
    def test_chunk_invalid_params(self):
        """Test chunk with invalid parameters."""
        env = PyReplEnv()
        env.set_content("test")
        
        # Invalid 'by' parameter
        result = env.execute("chunks = chunk(P, by='invalid')")
        assert result['exception'] is not None
        assert 'ValueError' in result['exception']


class TestSelectTool:
    """Test select() range extraction tool."""
    
    def test_select_single_range(self):
        """Test selecting a single range."""
        env = PyReplEnv()
        env.set_content("0123456789")
        
        result = env.execute("print(select(P, [(2, 5)]))")
        assert result['exception'] is None
        assert '234' in result['stdout']
    
    def test_select_multiple_ranges(self):
        """Test selecting multiple ranges."""
        env = PyReplEnv()
        env.set_content("0123456789")
        
        result = env.execute("print(select(P, [(0, 3), (7, 10)]))")
        assert result['exception'] is None
        assert '012' in result['stdout']
        assert '789' in result['stdout']
        assert 'Range 2' in result['stdout']  # Range marker
    
    def test_select_empty_ranges(self):
        """Test select with empty ranges list."""
        env = PyReplEnv()
        env.set_content("test")
        
        result = env.execute("text = select(P, []); print(f'len={len(text)}')")
        assert result['exception'] is None
        assert 'len=0' in result['stdout']
    
    def test_select_with_negative_index(self):
        """Test select with negative indices."""
        env = PyReplEnv()
        env.set_content("0123456789")
        
        result = env.execute("print(select(P, [(-5, None)]))")  # Last 5 chars
        assert result['exception'] is None
        assert '56789' in result['stdout']
    
    def test_select_max_chars_limit(self):
        """Test select respects max_chars limit."""
        env = PyReplEnv()
        env.set_content("x" * 1000)
        
        result = env.execute("text = select(P, [(0, 1000)], max_chars=100); print('done')")
        assert result['exception'] is None
        assert 'done' in result['stdout']


class TestToolsIntegration:
    """Test tools working together with P variable."""
    
    def test_tools_available_in_env(self):
        """Test that all tools are available in environment."""
        env = PyReplEnv()
        
        result = env.execute("""
print(callable(peek))
print(callable(grep))
print(callable(chunk))
print(callable(select))
""")
        assert result['exception'] is None
        assert result['stdout'].count('True') == 4
    
    def test_tools_with_p_variable(self):
        """Test tools work with P variable."""
        env = PyReplEnv()
        env.set_content("Test content for P variable")
        
        result = env.execute("print(peek(P, 0, 4))")
        assert result['exception'] is None
        assert 'Test' in result['stdout']
    
    def test_tools_persist_after_reset(self):
        """Test tools available after reset."""
        env = PyReplEnv()
        env.execute("x = 1")
        env.reset()
        
        result = env.execute("print(callable(peek))")
        assert result['exception'] is None
        assert 'True' in result['stdout']
    
    def test_combined_tool_usage(self):
        """Test using multiple tools in sequence."""
        env = PyReplEnv()
        env.set_content(SAMPLE_TEXT)
        
        result = env.execute("""
# Find ERROR lines
errors = grep(P, 'ERROR')
print('Found errors')

# Chunk the content
chunks = chunk(P, size=100, by='chars')
print(f'Created {len(chunks)} chunks')

# Peek at beginning
beginning = peek(P, 0, 50)
print(f'Beginning length: {len(beginning)}')
""")
        assert result['exception'] is None
        assert 'Found errors' in result['stdout']
        assert 'chunks' in result['stdout']
        assert 'Beginning length' in result['stdout']


class TestToolsWithSafeMode:
    """Test tools work in safe mode."""
    
    def test_tools_in_safe_mode(self):
        """Test that tools work with safe mode enabled."""
        env = PyReplEnv(safe_mode=True)
        env.set_content("Safe mode test")
        
        result = env.execute("print(peek(P, 0, 4))")
        assert result['exception'] is None
        assert 'Safe' in result['stdout']
    
    def test_tools_with_imports_in_safe_mode(self):
        """Test tools with safe imports."""
        env = PyReplEnv(safe_mode=True)
        env.set_content("test")
        
        result = env.execute("""
import re
# grep uses re internally, should work
result = grep(P, 'test')
print('grep works')
""")
        assert result['exception'] is None
        assert 'grep works' in result['stdout']
