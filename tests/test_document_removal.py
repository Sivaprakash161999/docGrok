"""
Automated unit and integration test suite for document removal and lifecycle management:
Tests soft memory unloading, permanent disk deletion, benchmark seed protection,
and automatic pinned scope cleanup in an isolated test environment.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant import DocumentAssistant
from src.retrieval import DocumentRepository, DEFAULT_CORPUS_DIR, SEED_PROTECTED_DOCS


class TestDocumentRemoval(unittest.TestCase):
    """Verifies document lifecycle management, permanent removal, and seed protection."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_removal_")
        self.corpus_dir = Path(self.temp_dir) / "corpus"
        self.sessions_dir = Path(self.temp_dir) / "sessions"
        self.logs_dir = Path(self.temp_dir) / "logs"

        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Seed temporary corpus from DEFAULT_CORPUS_DIR
        if DEFAULT_CORPUS_DIR.exists():
            for f in DEFAULT_CORPUS_DIR.iterdir():
                if f.is_file():
                    shutil.copy2(f, self.corpus_dir / f.name)

        self.repo = DocumentRepository(corpus_dir=self.corpus_dir)
        self.assistant = DocumentAssistant(
            sessions_dir=self.sessions_dir,
            logs_dir=self.logs_dir,
            repository=self.repo
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_soft_removal(self):
        """Tests unloading a document from memory while preserving disk files."""
        doc_id = "TEMP-TEST-SOFT"
        self.repo.add_document(
            doc_id=doc_id,
            title="Temporary Soft Doc",
            doc_type="test",
            content="This is a temporary memory document.",
            persist=False
        )
        self.assertIsNotNone(self.repo.get_document_by_id(doc_id))

        success, msg = self.repo.remove_document(doc_id, permanent=False)
        self.assertTrue(success)
        self.assertIn("Unloaded", msg)
        self.assertIsNone(self.repo.get_document_by_id(doc_id))

    def test_permanent_removal(self):
        """Tests permanently deleting a document from memory and from disk."""
        doc_id = "TEMP-TEST-PERM"
        self.repo.add_document(
            doc_id=doc_id,
            title="Temporary Permanent Doc",
            doc_type="test",
            content="This document will be deleted from disk.",
            persist=True
        )
        disk_file = self.corpus_dir / f"{doc_id}.json"
        self.assertTrue(disk_file.is_file())
        self.assertIsNotNone(self.repo.get_document_by_id(doc_id))

        success, msg = self.repo.remove_document(doc_id, permanent=True)
        self.assertTrue(success)
        self.assertIn("Permanently removed", msg)
        self.assertFalse(disk_file.is_file())
        self.assertIsNone(self.repo.get_document_by_id(doc_id))

    def test_seed_protection(self):
        """Tests that base seed benchmark documents are protected from accidental deletion."""
        seed_id = "FIN-2024-Q1"
        seed_file = self.corpus_dir / f"{seed_id}.json"
        self.assertTrue(seed_file.is_file())

        # Attempt deletion without force
        success, msg = self.repo.remove_document(seed_id, permanent=True, force=False)
        self.assertFalse(success)
        self.assertIn("protected benchmark seed document", msg)
        self.assertTrue(seed_file.is_file())
        self.assertIsNotNone(self.repo.get_document_by_id(seed_id))

    def test_assistant_active_pin_cleanup(self):
        """Tests that removing a document automatically clears active pinned document scope in assistant."""
        doc_id = "TEMP-PIN-TEST"
        self.assistant.repo.add_document(
            doc_id=doc_id,
            title="Pinned Test Doc",
            doc_type="test",
            content="Content to pin.",
            persist=False
        )
        pinned = self.assistant.pin_document(doc_id)
        self.assertTrue(pinned)
        self.assertEqual(self.assistant.pinned_doc_id, doc_id)

        # Remove the pinned document
        success, msg = self.assistant.remove_document(doc_id, permanent=False)
        self.assertTrue(success)
        self.assertIsNone(self.assistant.pinned_doc_id)

    def test_nonexistent_document_handling(self):
        """Tests graceful handling when attempting to remove a document that does not exist."""
        success, msg = self.repo.remove_document("DOES-NOT-EXIST-9999", permanent=True)
        self.assertFalse(success)
        self.assertIn("was not found", msg)


if __name__ == "__main__":
    unittest.main()
