"""
Automated unit and integration test suite for logging infrastructure:
Verifies formatted stream logging and rotating file persistence in an isolated test environment.
"""

from __future__ import annotations

import logging
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
from src.logger import get_logger, setup_logging


class TestLoggingInfrastructure(unittest.TestCase):
    """Verifies file-based logging, formatting schemas, and namespace hierarchy."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_log_suite_")
        self.logs_dir = Path(self.temp_dir) / "logs"
        self.sessions_dir = Path(self.temp_dir) / "sessions"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Remove handlers on docgrok and doc_assistant to release file locks on Windows
        for name in ("docgrok", "doc_assistant"):
            root_logger = logging.getLogger(name)
            for handler in list(root_logger.handlers):
                try:
                    handler.close()
                except Exception:
                    pass
                root_logger.removeHandler(handler)

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_setup_logging_unit(self):
        """Tests that setup_logging writes formatted log records to disk in an isolated directory."""
        log_file = self.logs_dir / "test_run.log"
        logger = setup_logging(
            log_dir=self.logs_dir,
            log_filename="test_run.log",
            file_level=logging.DEBUG,
            force_reconfigure=True
        )

        test_msg_info = "UNIT_TEST_ISOLATED_INFO_123"
        test_msg_warn = "UNIT_TEST_ISOLATED_WARN_456"

        logger.info(test_msg_info)
        child_logger = get_logger("agent")
        child_logger.warning(test_msg_warn)

        # Flush handlers
        for h in logger.handlers:
            h.flush()

        self.assertTrue(log_file.is_file())
        content = log_file.read_text(encoding="utf-8")
        self.assertIn(test_msg_info, content)
        self.assertIn(test_msg_warn, content)
        self.assertIn("docgrok.agent", content)
        self.assertIn("INFO", content)
        self.assertIn("WARNING", content)

    def test_assistant_logging_integration_isolated(self):
        """Tests that DocumentAssistant logs graph actions into an isolated logs directory."""
        assistant = DocumentAssistant(
            sessions_dir=self.sessions_dir,
            logs_dir=self.logs_dir
        )
        assistant.process_message("What is document FIN-INV-1092?")

        log_file = self.logs_dir / "assistant.log"
        self.assertTrue(log_file.is_file())
        content = log_file.read_text(encoding="utf-8")
        self.assertGreater(len(content), 100)
        self.assertIn("docgrok", content)


if __name__ == "__main__":
    unittest.main()
