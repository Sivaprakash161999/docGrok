"""
Custom agent tools for document retrieval and secure mathematical computation.
Includes AST-validated mathematical evaluator and LangChain @tool wrappers
with execution logging.
"""

from __future__ import annotations

import ast
import logging
import operator
from typing import Any, Callable, Dict, List, Optional
from langchain_core.tools import tool

from src.retrieval import DocumentRepository, default_repository

# Setup structured logger for tool invocations
logger = logging.getLogger("doc_assistant.tools")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[TOOL LOG - %(asctime)s] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# 1. AST-Validated Safe Calculator Engine
# ============================================================================

# Whitelist allowed AST binary and unary operators
ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Whitelist safe built-in mathematical functions
ALLOWED_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
}


def _safe_eval_ast(node: ast.AST) -> float:
    """Recursively evaluates an AST node strictly restricting execution to safe arithmetic."""
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)

    elif isinstance(node, ast.Constant):  # Python 3.8+ numeric/literal
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            operand = _safe_eval_ast(node.operand)
            return float(ALLOWED_OPERATORS[op_type](operand))
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in ALLOWED_OPERATORS:
            left = _safe_eval_ast(node.left)
            right = _safe_eval_ast(node.right)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division or modulo by zero is not permitted.")
            return float(ALLOWED_OPERATORS[op_type](left, right))
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCTIONS:
            func = ALLOWED_FUNCTIONS[node.func.id]
            args = [_safe_eval_ast(arg) for arg in node.args]
            return float(func(*args))
        raise ValueError(f"Function call not permitted: {ast.dump(node.func)}")

    else:
        raise ValueError(f"Disallowed expression element: {type(node).__name__}")


def evaluate_math_expression(expression: str) -> float:
    """
    Safely parses and computes a mathematical string using AST parsing.
    Prevents any code injection, file access, or unsafe execution.
    """
    cleaned = expression.strip().replace("$", "").replace(",", "")
    if not cleaned:
        raise ValueError("Empty mathematical expression provided.")

    try:
        parsed_tree = ast.parse(cleaned, mode="eval")
        return _safe_eval_ast(parsed_tree)
    except SyntaxError as e:
        raise ValueError(f"Malformed arithmetic expression '{expression}': {str(e)}")


# ============================================================================
# 2. Tool Factories & LangChain @tool Definitions
# ============================================================================

def create_calculator_tool():
    """
    Creates and returns the LangChain calculator tool for deterministic arithmetic.
    """
    @tool
    def calculator(expression: str) -> str:
        """
        Safely evaluates mathematical expressions (e.g. '4250000 + 4800000', '1020000 / 4250000 * 100').
        Always use this tool for all arithmetic, percentages, totals, and differences.
        Input should be a standard math expression using +, -, *, /, **, %, round, abs, min, max.
        """
        logger.info(f"Invoking calculator tool with expression: '{expression}'")
        try:
            val = evaluate_math_expression(expression)
            logger.info(f"Calculator result: {val}")
            return str(val)
        except Exception as e:
            err_msg = f"Calculator error evaluating '{expression}': {str(e)}"
            logger.error(err_msg)
            return err_msg

    return calculator


def create_document_search_tool(repo: Optional[DocumentRepository] = None):
    """
    Creates and returns the LangChain tool for searching documents by query keywords.
    """
    active_repo = repo or default_repository

    @tool
    def search_documents(query: str) -> str:
        """
        Searches the document repository for relevant financial reports, invoices,
        budgets, and healthcare records based on keyword query.
        Returns a list of matching document summaries with doc_id and preview.
        """
        logger.info(f"Searching documents with query: '{query}'")
        matches = active_repo.search_by_keyword(query, limit=4)
        if not matches:
            return "No matching documents found in repository for the given query."

        results = []
        for i, chunk in enumerate(matches, start=1):
            results.append(
                f"[{i}] Doc ID: {chunk.doc_id}\n"
                f"    Title: {chunk.title} (Type: {chunk.doc_type})\n"
                f"    Relevance: {chunk.relevance_score}\n"
                f"    Excerpt: {chunk.content[:250]}..."
            )
        return "\n\n".join(results)

    return search_documents


def create_document_reader_tool(repo: Optional[DocumentRepository] = None):
    """
    Creates and returns the LangChain tool for reading full document contents by ID.
    """
    active_repo = repo or default_repository

    @tool
    def read_document(doc_id: str) -> str:
        """
        Retrieves the complete content, details, and metadata of a specific document by its doc_id.
        Use this when you know the doc_id (e.g. 'FIN-2024-Q1', 'MED-CLM-7701') and need exact figures.
        """
        logger.info(f"Reading document ID: '{doc_id}'")
        doc = active_repo.get_document_by_id(doc_id)
        if not doc:
            return f"Error: Document with ID '{doc_id}' was not found in the repository."

        meta_lines = [f"{k}: {v}" for k, v in doc.metadata.items()]
        return (
            f"Document ID: {doc.doc_id}\n"
            f"Title: {doc.title}\n"
            f"Type: {doc.doc_type}\n"
            f"Metadata:\n  " + "\n  ".join(meta_lines) + "\n\n"
            f"Full Content:\n{doc.content}"
        )

    return read_document


def create_document_statistics_tool(repo: Optional[DocumentRepository] = None):
    """
    Creates and returns the LangChain tool for inspecting overall corpus statistics.
    """
    active_repo = repo or default_repository

    @tool
    def get_document_statistics() -> str:
        """
        Provides statistical metrics of all documents in the repository,
        including total document count, categories, and dollar volume.
        """
        logger.info("Retrieving corpus statistics.")
        stats = active_repo.get_statistics()
        return (
            f"Repository Statistics:\n"
            f"- Total Documents: {stats['total_documents']}\n"
            f"- Documents with Amounts: {stats['documents_with_amounts']}\n"
            f"- Total Dollar Volume: ${stats['total_dollar_volume']:,.2f}\n"
            f"- Average Amount: ${stats['average_amount']:,.2f}\n"
            f"- Categories: {stats['document_types']}"
        )

    return get_document_statistics


def get_default_tools(repo: Optional[DocumentRepository] = None) -> List[Any]:
    """Returns the full suite of default agent tools."""
    return [
        create_calculator_tool(),
        create_document_search_tool(repo),
        create_document_reader_tool(repo),
        create_document_statistics_tool(repo)
    ]
