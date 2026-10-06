"""Tests for File Loader module (Phase 2)."""

import io
import zipfile
from src.loader import (
    is_supported_file,
    is_ignored_path,
    get_file_type,
    create_document_from_text,
    load_from_uploaded_files,
    load_from_directory,
    load_from_zip,
    scan_directory_summary,
)


class MockUploadedFile:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_is_supported_file():
    assert is_supported_file("app.py") is True
    assert is_supported_file("auth.js") is True
    assert is_supported_file("README.md") is True
    assert is_supported_file("schema.sql") is True
    assert is_supported_file("image.png") is False
    assert is_supported_file("binary.exe") is False
    assert is_supported_file(".env") is False


def test_is_ignored_path():
    assert is_ignored_path("node_modules/express/index.js") is True
    assert is_ignored_path(".git/config") is True
    assert is_ignored_path(".venv/lib/python3.13/site-packages") is True
    assert is_ignored_path(".env") is True
    assert is_ignored_path("backend/auth.js") is False
    assert is_ignored_path("README.md") is False


def test_get_file_type():
    assert get_file_type("main.py") == "python"
    assert get_file_type("server.js") == "javascript"
    assert get_file_type("component.tsx") == "typescript_react"
    assert get_file_type("README.md") == "markdown"
    assert get_file_type("unknown.xyz") == "unknown"


def test_create_document_metadata_and_empty():
    # Valid document
    doc = create_document_from_text("console.log('hello');", "backend/auth.js")
    assert doc is not None
    assert doc.page_content == "console.log('hello');"
    assert doc.metadata["source"] == "backend/auth.js"
    assert doc.metadata["file_path"] == "backend/auth.js"
    assert doc.metadata["file_type"] == "javascript"

    # Empty document should be skipped
    empty_doc = create_document_from_text("   \n\t  ", "backend/empty.js")
    assert empty_doc is None

    # Ignored path should be skipped
    ignored_doc = create_document_from_text("secret", ".env")
    assert ignored_doc is None


def test_load_from_uploaded_files():
    mock_files = [
        MockUploadedFile("auth.js", b"function login() { return true; }"),
        MockUploadedFile("README.md", b"# My Project"),
        MockUploadedFile("empty.py", b"   "),
        MockUploadedFile("unsupported.bin", b"\x00\x01\x02"),
    ]
    docs = load_from_uploaded_files(mock_files)
    assert len(docs) == 2
    sources = [d.metadata["source"] for d in docs]
    assert "auth.js" in sources
    assert "README.md" in sources


def test_load_from_sample_repo_directory():
    docs = load_from_directory("sample_repo")
    assert len(docs) >= 5
    sources = [d.metadata["source"] for d in docs]
    assert any("README.md" in s for s in sources)
    assert any("auth.js" in s for s in sources)
    assert any("middleware.js" in s for s in sources)
    assert any("database.js" in s for s in sources)
    assert any("server.js" in s for s in sources)


def test_scan_directory_summary():
    summary = scan_directory_summary("sample_repo")
    assert summary["exists"] is True
    assert summary["file_count"] >= 5
    assert any("backend/auth.js" in f for f in summary["supported_files"])

    bad_summary = scan_directory_summary("non_existent_folder_xyz")
    assert bad_summary["exists"] is False
    assert bad_summary["file_count"] == 0


def test_load_from_zip():
    # Create an in-memory zip file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("my-repo-main/backend/auth.js", "const jwt = require('jsonwebtoken');")
        zf.writestr("my-repo-main/backend/database.js", "const pool = new Pool();")
        zf.writestr("my-repo-main/README.md", "# Test Zip Repo")
        zf.writestr("my-repo-main/.git/config", "ignored git config")
        zf.writestr("my-repo-main/node_modules/pkg/index.js", "ignored node_modules")
        zf.writestr("my-repo-main/image.png", b"\x89PNG\r\n\x1a\n")

    docs = load_from_zip(zip_buffer.getvalue())
    assert len(docs) == 3
    sources = [d.metadata["source"] for d in docs]
    assert "backend/auth.js" in sources
    assert "backend/database.js" in sources
    assert "README.md" in sources
    # Verify node_modules, .git, and png were ignored
    assert not any("node_modules" in s for s in sources)
    assert not any(".git" in s for s in sources)

