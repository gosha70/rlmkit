"""User interface components for RLMKit.

This package provides:
- Streamlit web interface for interactive testing
- File upload and processing utilities
- Chart and visualization components
- Results display and export functionality
"""

from .file_processor import FileProcessor, process_file

__all__ = ['FileProcessor', 'process_file']
