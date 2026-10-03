"""
Automated unit and integration test suite for Session Persistence & Resumption:
Tests session creation, serialization, multi-turn history, and resumption
across separate DocumentAssistant instances in an isolated environment.
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
from src.session_manager import SessionManager
from src.schemas import CalculationResponse


class TestSessionPersistence(unittest.TestCase):
    """Verifies SessionManager storage, resumption, and multi-turn state continuity."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_sess_")
        self.sessions_dir = Path(self.temp_dir) / "sessions"
        self.logs_dir = Path(self.temp_dir) / "logs"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_session_manager_crud(self):
        """Tests basic CRUD operations on SessionManager."""
        manager = SessionManager(sessions_dir=self.sessions_dir)
        test_sid = "test_crud_01"

        # 1. Create
        session_data = manager.create_session(session_id=test_sid, user_id="tester")
        self.assertEqual(session_data["session_id"], test_sid)
        self.assertTrue(manager.session_exists(test_sid))

        # 2. File exists in isolated directory
        session_file = self.sessions_dir / f"{test_sid}.json"
        self.assertTrue(session_file.is_file())

        # 3. Load
        loaded = manager.load_session(test_sid)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["user_id"], "tester")
        self.assertEqual(loaded["turns"], [])

        # 4. List
        sessions_list = manager.list_sessions()
        self.assertEqual(len(sessions_list), 1)
        self.assertEqual(sessions_list[0]["session_id"], test_sid)

        # 5. Delete
        self.assertTrue(manager.delete_session(test_sid))
        self.assertFalse(manager.session_exists(test_sid))
        self.assertEqual(len(manager.list_sessions()), 0)

    def test_session_persistence_and_resumption_e2e(self):
        """Tests end-to-end multi-turn conversation persistence across separate assistant instances."""
        session_id = "test_e2e_isolation"

        # Instance A starts session
        assistant_a = DocumentAssistant(
            sessions_dir=self.sessions_dir,
            logs_dir=self.logs_dir,
            session_id=session_id
        )
        assistant_a.pin_document("FIN-2024-Q1")

        state_1 = assistant_a.process_message("What was the total revenue in Q1 2024?")
        self.assertTrue(state_1["current_response"].found)

        saved_file = self.sessions_dir / f"{session_id}.json"
        self.assertTrue(saved_file.is_file(), "Session must be persisted to isolated sessions dir")

        # Instance B resumes the session
        assistant_b = DocumentAssistant(
            sessions_dir=self.sessions_dir,
            logs_dir=self.logs_dir,
            session_id=session_id
        )
        self.assertEqual(assistant_b.pinned_doc_ids, ["FIN-2024-Q1"])
        self.assertIn("FIN-2024-Q1", assistant_b.active_documents)
        self.assertGreater(len(assistant_b.conversation_summary), 10)

        # Instance B asks follow-up calculation
        state_2 = assistant_b.process_message("What is the difference between total revenue and R&D expenses?")
        resp_2 = state_2.get("current_response")
        self.assertIsInstance(resp_2, CalculationResponse)
        self.assertTrue(resp_2.found)
        # Revenue 4,250,000 - 850,000 = 3,400,000
        self.assertAlmostEqual(resp_2.result, 3400000.0, delta=1.0)

        # Verify saved history has 2 turns
        session_data = assistant_b.get_session_history()
        self.assertEqual(len(session_data.get("turns", [])), 2)


if __name__ == "__main__":
    unittest.main()
