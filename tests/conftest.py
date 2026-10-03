"""
Pytest configuration and test environment isolation fixtures.
Guarantees that test executions never write to or pollute production
sessions/ or logs/ directories, and ensures test corpora are isolated.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Generator

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant import DocumentAssistant
from src.retrieval import DocumentRepository, DEFAULT_CORPUS_DIR


@pytest.fixture
def temp_test_env(tmp_path: Path) -> Generator[dict[str, Path], None, None]:
    """
    Creates an isolated environment for test runs with temporary
    directories for sessions, logs, and a clean copy of the corpus.
    """
    sessions_dir = tmp_path / "sessions"
    logs_dir = tmp_path / "logs"
    corpus_dir = tmp_path / "corpus"

    sessions_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Seed temporary corpus with files from default corpus
    if DEFAULT_CORPUS_DIR.exists():
        for file in DEFAULT_CORPUS_DIR.iterdir():
            if file.is_file():
                shutil.copy2(file, corpus_dir / file.name)

    yield {
        "sessions_dir": sessions_dir,
        "logs_dir": logs_dir,
        "corpus_dir": corpus_dir,
    }


@pytest.fixture
def isolated_assistant(temp_test_env: dict[str, Path]) -> DocumentAssistant:
    """
    Returns a DocumentAssistant instance completely isolated from
    production sessions, logs, and corpus directories.
    """
    repo = DocumentRepository(corpus_dir=temp_test_env["corpus_dir"])
    return DocumentAssistant(
        sessions_dir=temp_test_env["sessions_dir"],
        logs_dir=temp_test_env["logs_dir"],
        repository=repo
    )
