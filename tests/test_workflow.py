"""
Automated unit and integration test suite for LangGraph Document Assistant workflow:
Validates intent classification, specialist node execution, state accumulation,
and persistent memory across conversational turns in an isolated test environment.
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
from src.schemas import AnswerResponse, CalculationResponse, SummaryResponse


class TestWorkflowExecution(unittest.TestCase):
    """Verifies end-to-end routing and specialist agent execution."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="test_workflow_")
        cls.sessions_dir = Path(cls.temp_dir) / "sessions"
        cls.logs_dir = Path(cls.temp_dir) / "logs"
        cls.sessions_dir.mkdir(parents=True, exist_ok=True)
        cls.logs_dir.mkdir(parents=True, exist_ok=True)

        cls.assistant = DocumentAssistant(
            sessions_dir=cls.sessions_dir,
            logs_dir=cls.logs_dir
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_factual_qa_route(self):
        """Validates factual Q&A route, intent classification, and source citation."""
        query = "What medical procedure was performed according to claim MED-CLM-7701?"
        state = self.assistant.process_message(query)

        intent = state.get("intent")
        resp = state.get("current_response")
        actions = state.get("actions_taken", [])

        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "qa")
        self.assertIn("classify_intent", actions)
        self.assertIn("qa_agent", actions)
        self.assertIn("update_memory", actions)
        self.assertIsInstance(resp, AnswerResponse)
        self.assertIn("MED-CLM-7701", resp.sources)
        self.assertTrue(resp.found)

    def test_02_calculation_route_and_tool(self):
        """Validates calculation intent and safe AST calculator tool execution."""
        query = "What was the total combined revenue of Q1 2024 and Q2 2024?"
        state = self.assistant.process_message(query)

        intent = state.get("intent")
        resp = state.get("current_response")
        actions = state.get("actions_taken", [])

        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "calculation")
        self.assertIn("calculation_agent", actions)
        self.assertIsInstance(resp, CalculationResponse)
        # Q1 (4,250,000) + Q2 (4,800,000) = 9,050,000
        self.assertAlmostEqual(resp.result, 9050000.0, delta=1.0)

    def test_03_summarization_route_and_memory(self):
        """Validates summarization route and progressive conversation summary accumulation."""
        query = "Can you summarize the inpatient knee surgery hospital claim MED-CLM-7701?"
        state = self.assistant.process_message(query)

        intent = state.get("intent")
        resp = state.get("current_response")
        actions = state.get("actions_taken", [])
        memory_summary = state.get("conversation_summary", "")

        self.assertIsNotNone(intent)
        self.assertEqual(intent.intent_type, "summarization")
        self.assertIn("summarization_agent", actions)
        self.assertIsInstance(resp, SummaryResponse)
        self.assertGreater(len(resp.key_points), 0)
        self.assertGreater(len(memory_summary), 10)


if __name__ == "__main__":
    unittest.main()
