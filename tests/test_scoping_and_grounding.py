"""
Automated unit and integration test suite for document scoping and anti-hallucination grounding:
Validates strict document focus, missing information handling, dynamic corpus lookup,
and external document ingestion in a completely isolated environment.
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
from src.schemas import AnswerResponse, CalculationResponse


class TestScopingAndGrounding(unittest.TestCase):
    """Verifies strict scoping boundaries, anti-hallucination, and missing information guards."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_scoping_")
        cls.sessions_dir = Path(cls.temp_dir) / "sessions"
        cls.logs_dir = Path(cls.temp_dir) / "logs"
        cls.corpus_dir = Path(cls.temp_dir) / "corpus"

        cls.sessions_dir.mkdir(parents=True, exist_ok=True)
        cls.logs_dir.mkdir(parents=True, exist_ok=True)
        cls.corpus_dir.mkdir(parents=True, exist_ok=True)

        # Seed temporary corpus from DEFAULT_CORPUS_DIR
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

    def test_01_external_doc_ingestion_and_isolated_persistence(self):
        """Validates external document ingestion and persistence into the isolated test corpus."""
        external_doc = {
            "doc_id": "TEST-ISOLATED-AUDIT",
            "title": "Isolated Security Audit",
            "doc_type": "security_audit",
            "content": "Audit Report: Penetration testing expenditure was $42,000.00. ISO 27001 verified.",
            "metadata": {"amount": 42000.0, "department": "Security"}
        }

        state = self.assistant.process_message(
            user_input="How much was spent on penetration testing according to the audit?",
            target_document=external_doc,
            persist_target_document=True
        )

        resp = state.get("current_response")
        self.assertIsInstance(resp, AnswerResponse)
        self.assertTrue(resp.found)
        self.assertIn("TEST-ISOLATED-AUDIT", resp.sources)
        # Check that file was persisted in isolated corpus, NOT production corpus
        self.assertTrue((self.corpus_dir / "TEST-ISOLATED-AUDIT.json").is_file())
        self.assertFalse((DEFAULT_CORPUS_DIR / "TEST-ISOLATED-AUDIT.json").exists())

    def test_02_targeted_scope_restriction(self):
        """Validates restricting queries to a specific targeted document ID."""
        state = self.assistant.process_message(
            user_input="Who is the vendor named on this invoice?",
            target_doc_id="FIN-INV-1092"
        )

        resp = state.get("current_response")
        self.assertIsInstance(resp, AnswerResponse)
        self.assertTrue(resp.found)
        self.assertIn("FIN-INV-1092", resp.sources)
        self.assertIn("horizon", resp.answer.lower())

    def test_03_missing_data_and_anti_hallucination(self):
        """Validates strict grounding and anti-hallucination when information is absent."""
        state = self.assistant.process_message(
            user_input="What is the employee turnover rate and HR satisfaction score?",
            target_doc_id="FIN-INV-1092"
        )

        resp = state.get("current_response")
        self.assertIsInstance(resp, AnswerResponse)
        self.assertFalse(resp.found, "Response should report found=False for missing data")
        refusal_phrases = [
            "not found", "could not be found", "not available",
            "no information", "do not contain", "does not contain", "cannot find"
        ]
        self.assertTrue(
            any(phrase in resp.answer.lower() for phrase in refusal_phrases),
            f"Expected refusal phrase in answer, got: {resp.answer}"
        )

    def test_04_dynamic_corpus_search(self):
        """Validates automatic dynamic keyword retrieval when no target document is specified."""
        self.assistant.unpin_document()
        state = self.assistant.process_message(
            user_input="What is the corporate budget allocation for IT modernization?"
        )

        resp = state.get("current_response")
        self.assertIsInstance(resp, AnswerResponse)
        self.assertTrue(resp.found)
        self.assertIn("FIN-BUD-003", resp.sources)
        self.assertIn("350,000", resp.answer)

    def test_05_graceful_not_found_on_corpus_miss(self):
        """Validates graceful refusal when a query matches nothing in the corpus."""
        self.assistant.unpin_document()
        state = self.assistant.process_message(
            user_input="What were the Q3 2019 aerospace propulsion widget metrics?"
        )

        resp = state.get("current_response")
        self.assertIsInstance(resp, AnswerResponse)
        self.assertFalse(resp.found)
        refusal_phrases = [
            "not found", "could not be found", "not available",
            "no information", "do not contain", "does not contain", "cannot find"
        ]
        self.assertTrue(
            any(phrase in resp.answer.lower() for phrase in refusal_phrases),
            f"Expected refusal phrase in answer, got: {resp.answer}"
        )


if __name__ == "__main__":
    unittest.main()
