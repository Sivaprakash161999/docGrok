"""
DocGrok: Intelligent Document Assistant Package.
Orchestrated multi-agent document analysis using LangGraph, LangChain, and Google Gemini.
"""

from src.assistant import DocumentAssistant, DocGrokAssistant, DocGrok
from src.logger import get_logger, setup_logging

__version__ = "0.1.0"
__all__ = ["DocGrok", "DocGrokAssistant", "DocumentAssistant", "setup_logging", "get_logger"]
