"""
Automated unit and integration test suite for session document isolation:
Verifies that ephemeral (non-persisted) documents are strictly isolated to their
active session thread, do not leak into fresh threads, and are properly restored
when resuming previous sessions.
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
from src.retrieval import DocumentRepository, DEFAULT_CORPUS_DIR


class TestSessionDocumentIsolation(unittest.TestCase):
    """Validates session boundary isolation and ephemeral document lifecycle."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_isolation_")
        cls.sessions_dir = Path(cls.temp_dir) / "sessions"
        cls.logs_dir = Path(cls.temp_dir) / "logs"
        cls.corpus_dir = Path(cls.temp_dir) / "corpus"

        cls.sessions_dir.mkdir(parents=True, exist_ok=True)
        cls.logs_dir.mkdir(parents=True, exist_ok=True)
        cls.corpus_dir.mkdir(parents=True, exist_ok=True)

        if DEFAULT_CORPUS_DIR.exists():
            for f in DEFAULT_CORPUS_DIR.iterdir():
                if f.is_file():
                    shutil.copy2(f, cls.corpus_dir / f.name)

        cls.repo = DocumentRepository(corpus_dir=cls.corpus_dir)
        cls.assistant = DocumentAssistant(
            sessions_dir=cls.sessions_dir,
            logs_dir=cls.logs_dir,
            repository=cls.repo
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        self.assistant.reset_session()
        self.repo.clear_ephemeral_documents()

    def tearDown(self):
        self.repo.clear_ephemeral_documents()

    def test_repository_ephemeral_document_tracking(self):
        """Verifies that DocumentRepository correctly differentiates persisted vs ephemeral docs."""
        initial_corpus_count = len(self.repo.get_corpus_documents())
        self.assertGreater(initial_corpus_count, 0)

        # Ingest an ephemeral document
        self.repo.add_document(
            doc_id="TEMP-EPHEMERAL-001",
            title="Temporary Ephemeral Document",
            doc_type="text_document",
            content="Ephemeral contents for session 1.",
            persist=False
        )
        self.assertFalse(self.repo.is_persisted("TEMP-EPHEMERAL-001"))
        self.assertEqual(len(self.repo.get_corpus_documents()), initial_corpus_count)
        self.assertEqual(len(self.repo.get_all_documents()), initial_corpus_count + 1)

        # Clear ephemeral documents
        cleared_count = self.repo.clear_ephemeral_documents()
        self.assertEqual(cleared_count, 1)
        self.assertIsNone(self.repo.get_document_by_id("TEMP-EPHEMERAL-001"))
        self.assertEqual(len(self.repo.get_all_documents()), initial_corpus_count)

    def test_reset_session_purges_ephemeral_scope_creep(self):
        """Tests that creating a new thread (reset_session) completely purges non-persisted files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f:
            f.write("Secret notes for Session A only.")
            tmp_path = f.name

        try:
            chunk = self.assistant.load_external_document(
                file_path=tmp_path,
                persist=False,
                doc_id="EXT-SESSION-A-DOC"
            )
            self.assertEqual(chunk.doc_id, "EXT-SESSION-A-DOC")
            self.assistant.pin_document("EXT-SESSION-A-DOC")

            all_doc_ids_a = [d.doc_id for d in self.assistant.repo.get_all_documents()]
            self.assertIn("EXT-SESSION-A-DOC", all_doc_ids_a)
            self.assertEqual(self.assistant.pinned_doc_ids, ["EXT-SESSION-A-DOC"])

            # User starts a fresh thread ("➕ New Thread")
            self.assistant.reset_session()

            # Verify: New session has NO scope creep from Session A
            all_doc_ids_new = [d.doc_id for d in self.assistant.repo.get_all_documents()]
            corpus_doc_ids_new = [d.doc_id for d in self.assistant.repo.get_corpus_documents()]

            self.assertNotIn("EXT-SESSION-A-DOC", all_doc_ids_new)
            self.assertNotIn("EXT-SESSION-A-DOC", corpus_doc_ids_new)
            self.assertEqual(self.assistant.pinned_doc_ids, [])
            self.assertEqual(self.assistant.active_documents, [])
            self.assertEqual(len(all_doc_ids_new), len(corpus_doc_ids_new))

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_multi_session_ephemeral_isolation_and_resumption(self):
        """Verifies switching between two sessions isolates their respective ephemeral documents."""
        sid_1 = self.assistant.session_id
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f1:
            f1.write("Session 1 confidential content.")
            tmp1_path = f1.name

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", encoding="utf-8") as f2:
            f2.write("Session 2 confidential content.")
            tmp2_path = f2.name

        try:
            self.assistant.load_external_document(file_path=tmp1_path, persist=False, doc_id="EXT-DOC-ONE")
            self.assistant.pin_document("EXT-DOC-ONE")

            self.assistant.session_manager.record_turn(
                session_id=sid_1,
                user_input="Analyze doc 1",
                result_state={"actions_taken": [], "active_documents": ["EXT-DOC-ONE"]},
                pinned_doc_id=["EXT-DOC-ONE"],
                ephemeral_documents={
                    "EXT-DOC-ONE": {
                        "doc_id": "EXT-DOC-ONE",
                        "title": "Doc One",
                        "doc_type": "text_document",
                        "content": "Session 1 confidential content."
                    }
                }
            )

            # Switch to Session 2 and load Doc 2
            sid_2 = self.assistant.reset_session()
            self.assertNotIn("EXT-DOC-ONE", [d.doc_id for d in self.assistant.repo.get_all_documents()])

            self.assistant.load_external_document(file_path=tmp2_path, persist=False, doc_id="EXT-DOC-TWO")
            self.assistant.pin_document("EXT-DOC-TWO")

            self.assistant.session_manager.record_turn(
                session_id=sid_2,
                user_input="Analyze doc 2",
                result_state={"actions_taken": [], "active_documents": ["EXT-DOC-TWO"]},
                pinned_doc_id=["EXT-DOC-TWO"],
                ephemeral_documents={
                    "EXT-DOC-TWO": {
                        "doc_id": "EXT-DOC-TWO",
                        "title": "Doc Two",
                        "doc_type": "text_document",
                        "content": "Session 2 confidential content."
                    }
                }
            )

            s2_docs = [d.doc_id for d in self.assistant.repo.get_all_documents()]
            self.assertIn("EXT-DOC-TWO", s2_docs)
            self.assertNotIn("EXT-DOC-ONE", s2_docs)

            # Resume Session 1
            resumed_1 = self.assistant.resume_session(sid_1)
            self.assertTrue(resumed_1)

            s1_docs = [d.doc_id for d in self.assistant.repo.get_all_documents()]
            self.assertIn("EXT-DOC-ONE", s1_docs)
            self.assertNotIn("EXT-DOC-TWO", s1_docs)
            self.assertEqual(self.assistant.pinned_doc_ids, ["EXT-DOC-ONE"])

            # Resume Session 2
            resumed_2 = self.assistant.resume_session(sid_2)
            self.assertTrue(resumed_2)

            s2_docs_again = [d.doc_id for d in self.assistant.repo.get_all_documents()]
            self.assertIn("EXT-DOC-TWO", s2_docs_again)
            self.assertNotIn("EXT-DOC-ONE", s2_docs_again)
            self.assertEqual(self.assistant.pinned_doc_ids, ["EXT-DOC-TWO"])

        finally:
            if os.path.exists(tmp1_path):
                os.unlink(tmp1_path)
            if os.path.exists(tmp2_path):
                os.unlink(tmp2_path)


if __name__ == "__main__":
    unittest.main()
