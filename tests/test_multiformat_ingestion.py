"""
Automated unit and integration test suite for multi-format document ingestion:
PDF (.pdf), Excel (.xlsx, .xls), CSV (.csv), and Infographic Images (.png, .jpg).
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

from src.parsers import parse_csv, parse_excel, parse_pdf, parse_image_infographic, parse_file
from src.retrieval import DocumentRepository


class TestMultiFormatIngestion(unittest.TestCase):
    """Verifies parsing and repository registration across diverse formats."""

    @classmethod
    def setUpClass(cls):
        cls.examples_dir = PROJECT_ROOT / "examples"
        cls.csv_file = cls.examples_dir / "sample_inventory.csv"
        cls.excel_file = cls.examples_dir / "sample_financial_model.xlsx"
        cls.pdf_file = cls.examples_dir / "sample_quarterly_report.pdf"
        cls.image_file = cls.examples_dir / "sample_kpi_infographic.png"

        cls.temp_dir = tempfile.mkdtemp(prefix="test_ingest_")
        cls.corpus_dir = Path(cls.temp_dir) / "corpus"
        cls.corpus_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_csv_parsing(self):
        """Validates CSV parsing, row counts, markdown formatting, and numeric summaries."""
        self.assertTrue(self.csv_file.exists(), "sample_inventory.csv must exist in examples/")
        content, meta = parse_csv(self.csv_file)

        self.assertEqual(meta["format"], "csv")
        self.assertEqual(meta["row_count"], 5)
        self.assertEqual(meta["column_count"], 6)
        self.assertIn("Item_ID", meta["columns"])
        self.assertIn("Total_Value", meta["numeric_summaries"])
        self.assertAlmostEqual(meta["numeric_summaries"]["Total_Value"]["total_sum"], 75500.0)
        self.assertIn("| Item_ID |", content)
        self.assertIn("SKU-1001", content)

    def test_excel_parsing(self):
        """Validates Excel parsing across multiple worksheets with custom statistics."""
        self.assertTrue(self.excel_file.exists(), "sample_financial_model.xlsx must exist in examples/")
        content, meta = parse_excel(self.excel_file)

        self.assertEqual(meta["format"], "excel")
        self.assertEqual(meta["total_worksheets"], 2)
        self.assertIn("Revenue_Streams", meta["sheet_names"])
        self.assertIn("Operating_Expenses", meta["sheet_names"])
        self.assertIn("## Sheet: Revenue_Streams", content)
        self.assertIn("## Sheet: Operating_Expenses", content)
        self.assertIn("SaaS_Subscriptions", content)

    def test_pdf_parsing(self):
        """Validates PDF page-by-page text extraction and metadata inspection."""
        self.assertTrue(self.pdf_file.exists(), "sample_quarterly_report.pdf must exist in examples/")
        content, meta = parse_pdf(self.pdf_file)

        self.assertEqual(meta["format"], "pdf")
        self.assertEqual(meta["total_pages"], 2)
        self.assertGreater(meta["char_count"], 100)
        self.assertIn("--- [Page 1 of 2] ---", content)
        self.assertIn("--- [Page 2 of 2] ---", content)
        self.assertIn("Executive Operations Briefing", content)

    def test_image_infographic_parsing(self):
        """Validates image properties, MIME detection, and multimodal transcription."""
        self.assertTrue(self.image_file.exists(), "sample_kpi_infographic.png must exist in examples/")
        content, meta = parse_image_infographic(self.image_file)

        self.assertEqual(meta["format"], "image")
        self.assertEqual(meta["image_format"], "PNG")
        self.assertEqual(meta["dimensions"], "800x600")
        self.assertEqual(meta["mime_type"], "image/png")
        self.assertIn("# Infographic Analysis", content)

    def test_unified_dispatcher(self):
        """Tests unified parse_file dispatcher returning appropriate doc_type."""
        _, dtype_csv, _ = parse_file(self.csv_file)
        self.assertEqual(dtype_csv, "csv_table")

        _, dtype_excel, _ = parse_file(self.excel_file)
        self.assertEqual(dtype_excel, "excel_spreadsheet")

        _, dtype_pdf, _ = parse_file(self.pdf_file)
        self.assertEqual(dtype_pdf, "pdf_document")

        _, dtype_img, _ = parse_file(self.image_file)
        self.assertEqual(dtype_img, "infographic_image")

    def test_repository_external_file_ingestion(self):
        """Verifies DocumentRepository seamlessly loads all multi-format files into an isolated repo."""
        repo = DocumentRepository(corpus_dir=self.corpus_dir)

        # Ingest CSV
        chunk_csv = repo.load_external_file(self.csv_file, doc_id="TEST-CSV")
        self.assertEqual(chunk_csv.doc_id, "TEST-CSV")
        self.assertEqual(chunk_csv.doc_type, "csv_table")
        self.assertIn("TEST-CSV", repo._docs)

        # Ingest Excel
        chunk_excel = repo.load_external_file(self.excel_file, doc_id="TEST-XLSX")
        self.assertEqual(chunk_excel.doc_id, "TEST-XLSX")
        self.assertEqual(chunk_excel.doc_type, "excel_spreadsheet")
        self.assertIn("TEST-XLSX", repo._docs)

        # Ingest PDF
        chunk_pdf = repo.load_external_file(self.pdf_file, doc_id="TEST-PDF")
        self.assertEqual(chunk_pdf.doc_id, "TEST-PDF")
        self.assertEqual(chunk_pdf.doc_type, "pdf_document")
        self.assertIn("TEST-PDF", repo._docs)

        # Ingest Infographic Image
        chunk_img = repo.load_external_file(self.image_file, doc_id="TEST-IMG")
        self.assertEqual(chunk_img.doc_id, "TEST-IMG")
        self.assertEqual(chunk_img.doc_type, "infographic_image")
        self.assertIn("TEST-IMG", repo._docs)

        # Test search finds terms from the newly ingested documents
        results_csv = repo.search_by_keyword("Gigabit")
        self.assertTrue(any(r.doc_id == "TEST-CSV" for r in results_csv))

        results_pdf = repo.search_by_keyword("ISO-27001")
        self.assertTrue(any(r.doc_id == "TEST-PDF" for r in results_pdf))


if __name__ == "__main__":
    unittest.main()
