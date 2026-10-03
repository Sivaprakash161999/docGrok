# DocGrok: Intelligent Agentic Document Assistant

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/framework-LangChain-green.svg)](https://github.com/langchain-ai/langchain)
[![Google Gemini](https://img.shields.io/badge/llm-Google%20Gemini-blueviolet.svg)](https://ai.google.dev/)
[![Tests](https://img.shields.io/badge/tests-29%20passed%20%7C%20100%25-success.svg)](tests/)

**DocGrok** is an autonomous, multi-agent document intelligence and reporting platform orchestrated using **LangGraph** and **LangChain**, powered by **Google Gemini**.

DocGrok analyzes complex corporate filings, invoices, budget reports, and healthcare claim records. It dynamically routes inquiries to specialized agents, executes deterministic arithmetic via an AST-sandboxed calculator tool, preserves multi-turn session states, and supports both a modern **Streamlit Web UI** and an interactive **Rich CLI**.

---

## 🏛️ System Architecture

```
                      [ User Input / Attachments ]
                                   │
                                   ▼
                         ┌───────────────────┐
                         │  classify_intent  │
                         └─────────┬─────────┘
                                   │ (Conditional Routing via UserIntent)
                ┌──────────────────┼──────────────────┐
                │ (qa)             │ (summarization)  │ (calculation)
                ▼                  ▼                  ▼
         ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐
         │   qa_agent   │   │ summarization│   │ calculation_agent │
         │              │   │   _agent     │   │ (AST calc tool)   │
         └──────┬───────┘   └──────┬───────┘   └─────────┬─────────┘
                │                  │                     │
                └──────────────────┼─────────────────────┘
                                   ▼
                         ┌───────────────────┐
                         │   update_memory   │
                         └─────────┬─────────┘
                                   │ (InMemorySaver + Session Serialization)
                                   ▼
                                [ END ]
```

---

## ✨ Key Capabilities

1. **Dual Frontends**:
   - **Streamlit Web Application (`app.py`)**: Modern chat interface with multi-file attachment uploader (`+`), real-time agent execution lineage pills, visual metric cards, document removal safeguards, and session persistence.
   - **Rich Interactive CLI (`main.py`)**: Terminal UI featuring colored panels, document management tables, and session switching.

2. **Specialist Multi-Agent Routing**:
   - **Intent Classifier (`classify_intent`)**: Dynamically categorizes inquiries into typed schemas (`qa`, `summarization`, `calculation`, `unknown`).
   - **Q&A Agent (`qa_agent`)**: Factual responses with explicit document citations (`sources`) and confidence scoring.
   - **Summarization Agent (`summarization_agent`)**: Generates structured executive summaries with key takeaway bullet points.
   - **Calculation Agent (`calculation_agent`)**: Sandboxed, deterministic AST evaluation preventing arithmetic hallucination.

3. **Multi-Format Ingestion Engine (`src/parsers.py`)**:
   - **PDF (`.pdf`)**: Page-by-page extraction, page boundary markers, author and page count metadata via `pypdf`.
   - **Excel (`.xlsx`, `.xls`)**: Multi-sheet extraction, tabular markdown, and descriptive statistics (sums, averages, min/max) via `openpyxl` & `pandas`.
   - **CSV (`.csv`)**: Tabular parsing, Markdown rendering without external dependencies, and automated column totals.
   - **Infographic Images (`.png`, `.jpg`, `.jpeg`, `.webp`)**: Multimodal visual transcription via Google Gemini Vision (`ChatGoogleGenerativeAI`), extracting chart data, axes, percentages, and KPIs.

4. **Multi-Document & Hybrid Scoping**:
   - **Multi-Document Pinning (`pinned_doc_ids`)**: Constrain queries across multiple documents simultaneously.
   - **Hybrid Scope**: Dynamically analyze uploaded external files together with existing corpus files in a single unified prompt context.
   - **Ephemeral Session Boundary**: Uploaded files can remain ephemeral (scoped strictly to the active session) or persisted permanently into `document_corpus/`.

5. **Strict Grounding & Anti-Hallucination**:
   - Answers are grounded strictly in the provided document context.
   - If data or documents are missing, the system gracefully reports `found=False` with a clear explanation rather than fabricating figures.

---

## 📂 Project Structure

```
doc_assistant_project/
├── PROJECT_SPECIFICATION.md  # Detailed project blueprint & requirements
├── README.md                 # Project guide & documentation (this file)
├── requirements.txt          # Python dependencies
├── app.py                    # Streamlit Web UI (DocGrok Assistant)
├── main.py                   # Interactive Rich CLI application
├── tests/                    # Fully isolated automated test suite (29 tests)
│   ├── __init__.py           # Test package marker
│   ├── conftest.py           # Pytest fixtures for temporary environment isolation
│   ├── test_document_removal.py      # Verification of document removal & seed protection
│   ├── test_hybrid_and_multiscope.py # Verification of multi-document & hybrid scoping
│   ├── test_logging.py               # Verification of file-based logging under isolated logs/
│   ├── test_multiformat_ingestion.py # Verification of PDF, Excel, CSV, and Image parsing
│   ├── test_scoping_and_grounding.py # Verification of strict scoping & missing data handling
│   ├── test_session_isolation.py     # Verification of ephemeral boundaries & session reset
│   ├── test_session_persistence.py   # Verification of sessions/ storage and resumption
│   └── test_workflow.py              # End-to-end multi-agent LangGraph workflow execution
├── sessions/                 # Persisted JSON conversation sessions
├── logs/                     # Persistent rotating log files (assistant.log)
├── examples/                 # Sample multi-format files for testing
│   ├── sample_vendor_invoice.json
│   ├── sample_clinical_memo.md
│   ├── sample_inventory.csv
│   ├── sample_financial_model.xlsx
│   ├── sample_quarterly_report.pdf
│   └── sample_kpi_infographic.png
├── document_corpus/          # Primary repository for indexed enterprise documents
└── src/
    ├── __init__.py           # Package exports (DocGrok, DocumentAssistant)
    ├── logger.py             # DocGrok centralized logging configuration
    ├── schemas.py            # Pydantic data schemas & AgentState contracts
    ├── parsers.py            # Multi-format document parser engine
    ├── retrieval.py          # DocumentRepository & search indexing
    ├── tools.py              # LangChain @tool definitions & AST calculator
    ├── prompts.py            # Contextual system prompts & persona definitions
    ├── agent.py              # LangGraph StateGraph, nodes, edges & compilation
    ├── session_manager.py    # Session serialization, storage, and resumption
    └── assistant.py          # DocumentAssistant / DocGrok orchestrator
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.12+ (or 3.10+)
- Google Gemini API Key ([Get one here](https://aistudio.google.com/app/apikey))

### 2. Activate Virtual Environment
```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-flash-lite-latest
```

---

## 💻 Running DocGrok

### Option A: Streamlit Web Interface (Recommended)
Launch the modern chat application in your browser:
```bash
streamlit run app.py
```
- Multi-file attachment drag-and-drop directly in the chat box (`+`).
- Corpus document multi-selector and hybrid scope selector.
- Visual execution lineage pills and formatted calculation cards.
- Persistent session switcher and document removal management.

### Option B: Interactive Rich CLI
Launch the terminal application:
```bash
python main.py
```

#### CLI Commands:
- `docs` / `list` – Display all indexed documents and key financial metrics.
- `use <doc_id>` – Restrict scope strictly to a specific document.
- `unpin` – Clear pinned scope and return to dynamic corpus search.
- `load <path> [--persist]` – Ingest any external file (PDF, Excel, CSV, PNG, JSON, Markdown).
- `remove <id> [--permanent]` – Remove a document from the corpus.
- `sessions` – Display saved conversation sessions.
- `resume <session_id>` – Resume a prior conversation thread from disk.
- `new` / `reset` – Start a clean session.
- `stats` – Show repository statistics and dollar volume.
- `exit` / `quit` – Terminate the application.

---

## 🧪 Automated Testing

DocGrok includes a comprehensive suite of **29 automated tests** covering all rubric requirements. All test runs are executed in isolated temporary sandboxes (`tempfile.mkdtemp`) to prevent pollution of production `sessions/`, `logs/assistant.log`, or `document_corpus/`.

### Run with Pytest:
```bash
pytest tests
```

### Run with Python Unittest:
```bash
python -m unittest discover tests
```
