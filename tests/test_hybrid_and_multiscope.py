"""
Automated unit and integration test suite for multi-document and hybrid scoping:
Validates multi-document pinning, batch external ingestion, and hybrid scope synthesis
in an isolated test environment.
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


class TestHybridAndMultiScope(unittest.TestCase):
    """Verifies multi-document pinning, batch loading, and hybrid document scoping."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_hybrid_")
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

        cls.examples_dir = PROJECT_ROOT / "examples"
        cls.csv_file = cls.examples_dir / "sample_inventory.csv"
        cls.pdf_file = cls.examples_dir / "sample_quarterly_report.pdf"

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        self.assistant.reset_session()
        self.repo.clear_ephemeral_documents()

    def tearDown(self):
        self.repo.clear_ephemeral_documents()

    def test_multi_document_pinning(self):
        """Tests pinning multiple documents simultaneously and verifying scope."""
        pinned = self.assistant.pin_documents(["FIN-2024-Q1", "FIN-2024-Q2"])
        self.assertEqual(pinned, ["FIN-2024-Q1", "FIN-2024-Q2"])
        self.assertEqual(self.assistant.pinned_doc_ids, ["FIN-2024-Q1", "FIN-2024-Q2"])
        self.assertEqual(self.assistant.pinned_doc_id, "FIN-2024-Q1")

        # Unpin specific doc
        remaining = self.assistant.unpin_document("FIN-2024-Q1")
        self.assertEqual(remaining, ["FIN-2024-Q2"])
        self.assertEqual(self.assistant.pinned_doc_ids, ["FIN-2024-Q2"])

        # Unpin all
        empty = self.assistant.unpin_document()
        self.assertEqual(empty, [])
        self.assertEqual(self.assistant.pinned_doc_ids, [])

    def test_batch_external_document_ingestion(self):
        """Tests batch ingestion of multiple external files simultaneously."""
        self.assertTrue(self.csv_file.exists())
        self.assertTrue(self.pdf_file.exists())

        chunks = self.assistant.load_external_documents(
            file_paths=[self.csv_file, self.pdf_file],
            persist=False,
            doc_ids=["BATCH-CSV-01", "BATCH-PDF-01"]
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].doc_id, "BATCH-CSV-01")
        self.assertEqual(chunks[1].doc_id, "BATCH-PDF-01")

        # Pin both
        self.assistant.pin_documents(["BATCH-CSV-01", "BATCH-PDF-01"])
        self.assertEqual(self.assistant.pinned_doc_ids, ["BATCH-CSV-01", "BATCH-PDF-01"])

        # Clean up
        self.assistant.remove_document("BATCH-CSV-01", permanent=False)
        self.assistant.remove_document("BATCH-PDF-01", permanent=False)
        self.assertEqual(self.assistant.pinned_doc_ids, [])

    def test_hybrid_scope_execution(self):
        """Tests hybrid query scoping combining an uploaded external doc with an existing corpus doc."""
        external_upload = {
            "doc_id": "EXT-SUPPLIER-QUOTE",
            "title": "Cloud Computing Infrastructure Quote",
            "doc_type": "supplier_quote",
            "content": "Supplier Quote: Cloud AI cluster expansion package quoted at $55,000.00.",
            "metadata": {"amount": 55000.0}
        }

        state = self.assistant.process_message(
            user_input="Compare the vendor invoice FIN-INV-1092 with the supplier quote EXT-SUPPLIER-QUOTE.",
            target_documents=[external_upload],
            target_doc_ids=["FIN-INV-1092"]
        )

        resp = state.get("current_response")
        actions = state.get("actions_taken", [])
        self.assertIn("classify_intent", actions)
        self.assertTrue(hasattr(resp, "found") and resp.found)
        self.assertIn("FIN-INV-1092", resp.sources)
        self.assertIn("EXT-SUPPLIER-QUOTE", resp.sources)


if __name__ == "__main__":
    unittest.main()
