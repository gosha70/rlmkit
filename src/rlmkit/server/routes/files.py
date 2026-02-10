"""File upload endpoints."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from rlmkit.server.dependencies import AppState, FileRecord, get_state
from rlmkit.server.models import ErrorDetail, ErrorResponse, FileUploadResponse

router = APIRouter()

_ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".py", ".json", ".csv"}
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/api/files/upload", status_code=201)
async def upload_file(
    file: UploadFile,
    state: AppState = Depends(get_state),
) -> FileUploadResponse:
    """Upload a document for analysis."""
    # Validate file type
    name = file.filename or "unnamed"
    ext = ""
    if "." in name:
        ext = "." + name.rsplit(".", 1)[-1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(_ALLOWED_EXTENSIONS))}",
        )

    # Read content
    raw = await file.read()
    if len(raw) > _MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File exceeds 50 MB limit")

    # Extract text
    text = _extract_text(raw, ext)
    token_estimate = max(1, len(text) // 4)

    file_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    record = FileRecord(
        id=file_id,
        name=name,
        size_bytes=len(raw),
        content_type=file.content_type or "application/octet-stream",
        text_content=text,
        token_count=token_estimate,
        created_at=now,
    )
    state.files[file_id] = record

    return FileUploadResponse(
        id=file_id,
        name=name,
        size_bytes=len(raw),
        type=record.content_type,
        token_count=token_estimate,
        created_at=now,
    )


@router.get("/api/files/{file_id}")
async def get_file(
    file_id: str,
    state: AppState = Depends(get_state),
) -> FileUploadResponse:
    """Get info about an uploaded file."""
    record = state.files.get(file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="File not found")

    return FileUploadResponse(
        id=record.id,
        name=record.name,
        size_bytes=record.size_bytes,
        type=record.content_type,
        token_count=record.token_count,
        created_at=record.created_at,
    )


def _extract_text(raw: bytes, ext: str) -> str:
    """Extract text content from raw file bytes."""
    if ext in (".txt", ".md", ".py", ".json", ".csv"):
        return raw.decode("utf-8", errors="replace")
    if ext == ".pdf":
        try:
            from PyPDF2 import PdfReader
            import io

            reader = PdfReader(io.BytesIO(raw))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return raw.decode("utf-8", errors="replace")
    if ext == ".docx":
        try:
            import docx
            import io

            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return raw.decode("utf-8", errors="replace")
    return raw.decode("utf-8", errors="replace")
