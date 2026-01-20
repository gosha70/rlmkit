"""File processing utilities for handling various file formats.

Supports:
- PDF files (text extraction)
- DOCX files (Microsoft Word)
- Plain text files (TXT, MD, etc.)
- JSON files
- Code files (PY, JS, etc.)
"""

import io
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass


@dataclass
class FileInfo:
    """Information about a processed file."""
    
    filename: str
    """Original filename"""
    
    content: str
    """Extracted text content"""
    
    file_type: str
    """File type/extension"""
    
    size_bytes: int
    """File size in bytes"""
    
    char_count: int
    """Number of characters in content"""
    
    estimated_tokens: int
    """Estimated token count (using 4 chars per token heuristic)"""
    
    success: bool = True
    """Whether processing was successful"""
    
    error: Optional[str] = None
    """Error message if processing failed"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'filename': self.filename,
            'file_type': self.file_type,
            'size_bytes': self.size_bytes,
            'char_count': self.char_count,
            'estimated_tokens': self.estimated_tokens,
            'success': self.success,
            'error': self.error,
        }


class FileProcessor:
    """Process various file formats into text."""
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        # Text formats
        '.txt', '.md', '.markdown', '.rst',
        # Code formats
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs',
        '.html', '.css', '.json', '.xml', '.yaml', '.yml',
        # Document formats (require extra libraries)
        '.pdf', '.docx',
    }
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count from text.
        
        Uses simple heuristic: ~4 characters per token.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        return max(1, len(text) // 4)
    
    @classmethod
    def process_file(
        cls,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        filename: Optional[str] = None
    ) -> FileInfo:
        """
        Process a file and extract text content.
        
        Args:
            file_path: Path to file on disk
            file_bytes: Raw file bytes (for uploaded files)
            filename: Original filename (required if using file_bytes)
            
        Returns:
            FileInfo with extracted content and metadata
        """
        # Determine filename and get file extension
        if file_path:
            path = Path(file_path)
            filename = path.name
            file_ext = path.suffix.lower()
            
            # Read file bytes
            with open(path, 'rb') as f:
                file_bytes = f.read()
        elif file_bytes and filename:
            file_ext = Path(filename).suffix.lower()
        else:
            return FileInfo(
                filename=filename or "unknown",
                content="",
                file_type="unknown",
                size_bytes=0,
                char_count=0,
                estimated_tokens=0,
                success=False,
                error="Must provide either file_path or both file_bytes and filename"
            )
        
        size_bytes = len(file_bytes)
        
        # Check if supported
        if file_ext not in cls.SUPPORTED_EXTENSIONS:
            return FileInfo(
                filename=filename,
                content="",
                file_type=file_ext,
                size_bytes=size_bytes,
                char_count=0,
                estimated_tokens=0,
                success=False,
                error=f"Unsupported file type: {file_ext}"
            )
        
        # Process based on file type
        try:
            if file_ext == '.pdf':
                content = cls._process_pdf(file_bytes)
            elif file_ext == '.docx':
                content = cls._process_docx(file_bytes)
            elif file_ext in ['.json']:
                content = file_bytes.decode('utf-8')
            else:
                # Try to decode as text
                content = cls._process_text(file_bytes)
            
            char_count = len(content)
            estimated_tokens = cls.estimate_tokens(content)
            
            return FileInfo(
                filename=filename,
                content=content,
                file_type=file_ext,
                size_bytes=size_bytes,
                char_count=char_count,
                estimated_tokens=estimated_tokens,
                success=True
            )
            
        except Exception as e:
            return FileInfo(
                filename=filename,
                content="",
                file_type=file_ext,
                size_bytes=size_bytes,
                char_count=0,
                estimated_tokens=0,
                success=False,
                error=f"Error processing file: {str(e)}"
            )
    
    @staticmethod
    def _process_text(file_bytes: bytes) -> str:
        """Process plain text file."""
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        
        # Fallback: decode with errors ignored
        return file_bytes.decode('utf-8', errors='ignore')
    
    @staticmethod
    def _process_pdf(file_bytes: bytes) -> str:
        """
        Process PDF file and extract text.
        
        Requires: PyPDF2 or pypdf
        """
        try:
            import PyPDF2
            pdf_file = io.BytesIO(file_bytes)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text_parts = []
            for page in reader.pages:
                text_parts.append(page.extract_text())
            
            return '\n\n'.join(text_parts)
        except ImportError:
            try:
                import pypdf
                pdf_file = io.BytesIO(file_bytes)
                reader = pypdf.PdfReader(pdf_file)
                
                text_parts = []
                for page in reader.pages:
                    text_parts.append(page.extract_text())
                
                return '\n\n'.join(text_parts)
            except ImportError:
                raise ImportError(
                    "PDF support requires 'PyPDF2' or 'pypdf'. "
                    "Install with: pip install PyPDF2"
                )
    
    @staticmethod
    def _process_docx(file_bytes: bytes) -> str:
        """
        Process DOCX file and extract text.
        
        Requires: python-docx
        """
        try:
            from docx import Document
            docx_file = io.BytesIO(file_bytes)
            doc = Document(docx_file)
            
            text_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text)
            
            return '\n\n'.join(text_parts)
        except ImportError:
            raise ImportError(
                "DOCX support requires 'python-docx'. "
                "Install with: pip install python-docx"
            )


def process_file(
    file_path: Optional[str] = None,
    file_bytes: Optional[bytes] = None,
    filename: Optional[str] = None
) -> FileInfo:
    """
    Convenience function to process a file.
    
    Args:
        file_path: Path to file on disk
        file_bytes: Raw file bytes (for uploaded files)
        filename: Original filename (required if using file_bytes)
        
    Returns:
        FileInfo with extracted content and metadata
    """
    return FileProcessor.process_file(file_path, file_bytes, filename)
