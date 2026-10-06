"""File Loader Module for RepoLens.

Responsible for:
- Reading supported source and documentation files from folders, ZIP archives, and file uploads.
- Filtering out unsupported files, empty files, and secret/ignored paths.
- Normalizing contents and attaching standardized metadata (source, file_path, file_type).
"""

import os
import io
import zipfile
from typing import List, Optional, Dict, Any, Union
from langchain_core.documents import Document

# Mapping of supported file extensions to human-readable language types
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript_react",
    ".ts": "typescript",
    ".tsx": "typescript_react",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".txt": "text",
    ".json": "json",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".sql": "sql",
}

# Directories and files that must always be ignored for security and performance
IGNORED_PATTERNS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".env",
    ".DS_Store",
    ".pytest_cache",
    "__MACOSX",
}


def is_ignored_path(file_path: str) -> bool:
    """Check if the given path contains any ignored directory or file name."""
    normalized = file_path.replace("\\", "/")
    parts = normalized.strip("/").split("/")
    for part in parts:
        if part in IGNORED_PATTERNS or part.startswith(".env") or part.startswith("._"):
            return True
    return False


def is_supported_file(file_name: str) -> bool:
    """Check if the file has an allowed extension and is not an ignored file."""
    _, ext = os.path.splitext(file_name.lower())
    if ext not in SUPPORTED_EXTENSIONS:
        return False
    base_name = os.path.basename(file_name)
    if base_name in IGNORED_PATTERNS or base_name.startswith(".env") or base_name.startswith("._"):
        return False
    return True


def get_file_type(file_name: str) -> str:
    """Return the language/file type based on file extension."""
    _, ext = os.path.splitext(file_name.lower())
    return SUPPORTED_EXTENSIONS.get(ext, "unknown")


def create_document_from_text(
    content: str,
    file_path: str,
) -> Optional[Document]:
    """Create a LangChain Document with required metadata from text content.

    Returns None if content is empty or file is ignored/unsupported.
    """
    if is_ignored_path(file_path) or not is_supported_file(file_path):
        return None

    cleaned_content = content.strip()
    if not cleaned_content:
        # Skip empty files safely
        return None

    norm_path = file_path.replace("\\", "/").lstrip("./")
    metadata = {
        "source": norm_path,
        "file_path": norm_path,
        "file_type": get_file_type(file_path),
    }

    return Document(page_content=cleaned_content, metadata=metadata)


def load_from_uploaded_files(uploaded_files: List) -> List[Document]:
    """Process a list of Streamlit UploadedFile objects.

    Extracts content, skips binary/unsupported/empty files, and attaches metadata.
    """
    documents: List[Document] = []

    for file_obj in uploaded_files:
        filename = getattr(file_obj, "name", str(file_obj))
        if not is_supported_file(filename) or is_ignored_path(filename):
            continue

        try:
            # Read bytes and decode
            file_bytes = file_obj.getvalue() if hasattr(file_obj, "getvalue") else file_obj.read()
            text_content = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            continue

        doc = create_document_from_text(text_content, filename)
        if doc:
            documents.append(doc)

    return documents


def load_from_directory(directory_path: str) -> List[Document]:
    """Recursively load all supported files from a local directory path.

    Traverses the directory tree and preserves relative file paths (e.g. backend/auth.js).
    """
    documents: List[Document] = []
    base_path = os.path.abspath(directory_path)

    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        return documents

    for root, dirs, files in os.walk(base_path):
        # Prune ignored directories in-place to prevent traversing them
        dirs[:] = [d for d in dirs if d not in IGNORED_PATTERNS and not is_ignored_path(d)]

        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_path)

            if is_ignored_path(rel_path) or not is_supported_file(file):
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                doc = create_document_from_text(content, rel_path)
                if doc:
                    documents.append(doc)
            except Exception:
                continue

    return documents


def load_from_zip(zip_source: Union[bytes, io.BytesIO, str, Any]) -> List[Document]:
    """Extract and load supported files from a ZIP archive (bytes, UploadedFile, or file path).

    Maintains nested folder hierarchy, ignoring hidden metadata and secrets.
    """
    documents: List[Document] = []

    if isinstance(zip_source, str):
        zf = zipfile.ZipFile(zip_source, "r")
    elif hasattr(zip_source, "getvalue"):
        zf = zipfile.ZipFile(io.BytesIO(zip_source.getvalue()), "r")
    elif hasattr(zip_source, "read"):
        zf = zipfile.ZipFile(io.BytesIO(zip_source.read()), "r")
    elif isinstance(zip_source, bytes):
        zf = zipfile.ZipFile(io.BytesIO(zip_source), "r")
    else:
        return documents

    with zf:
        namelist = zf.namelist()

        # Check if archive has a common top-level root directory (e.g. "repo-main/")
        common_prefix = ""
        non_empty_names = [n for n in namelist if not n.endswith("/")]
        if non_empty_names:
            first_parts = [n.split("/")[0] for n in non_empty_names if "/" in n]
            if first_parts and all(p == first_parts[0] for p in first_parts):
                common_prefix = first_parts[0] + "/"

        for member in zf.infolist():
            if member.is_dir():
                continue

            raw_path = member.filename
            if is_ignored_path(raw_path):
                continue

            # Strip common root folder prefix if present
            clean_path = raw_path
            if common_prefix and clean_path.startswith(common_prefix):
                clean_path = clean_path[len(common_prefix):]

            if not is_supported_file(clean_path) or is_ignored_path(clean_path):
                continue

            try:
                with zf.open(member) as f:
                    file_bytes = f.read()
                text = file_bytes.decode("utf-8", errors="replace")
                doc = create_document_from_text(text, clean_path)
                if doc:
                    documents.append(doc)
            except Exception:
                continue

    return documents


def scan_directory_summary(directory_path: str) -> Dict[str, Any]:
    """Scan a directory and return a summary of supported files, extensions, and validity."""
    base_path = os.path.abspath(directory_path)
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        return {"exists": False, "supported_files": [], "file_count": 0}

    supported_files = []
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_PATTERNS and not is_ignored_path(d)]
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_path).replace("\\", "/")
            if not is_ignored_path(rel_path) and is_supported_file(file):
                supported_files.append(rel_path)

    return {
        "exists": True,
        "supported_files": sorted(supported_files),
        "file_count": len(supported_files),
        "directory_path": base_path,
    }
