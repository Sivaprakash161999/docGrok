# Project Specification: DocGrok (Agentic Document Assistant)

## 1. Executive Overview
**DocGrok** is an intelligent, multi-agent document analysis and report-building system orchestrated with **LangGraph** and **LangChain**. It processes complex business, financial, and healthcare documents by dynamically routing user queries to specialized agents, tracking execution lineage, invoking custom tools, and maintaining conversational context across multi-turn interactions.

---

## 2. Core Functional Requirements & Rubric Alignment

### 2.1 Intent Routing & Classification
- **Dynamic Intent Classifier**: Analyzes user input alongside conversation history and determines the appropriate processing path:
  - `qa`: Specific factual questions referencing document details.
  - `summarization`: Synthesis, key takeaway extraction, or executive summaries.
  - `calculation`: Quantitative, mathematical, and comparative operations requiring precision.
  - `unknown`: Fallback handled gracefully (defaults to standard Q&A with conversational guidance).
- **Structured Output**: Uses strict Pydantic data schemas (`UserIntent`) enforcing typed reasoning and confidence scoring.

### 2.2 Specialist Agent Nodes
1. **Q&A Agent (`qa_agent`)**:
   - Searches and reads relevant documents.
   - Synthesizes factual answers with explicit document citations (`sources`) and confidence metrics.
2. **Summarization Agent (`summarization_agent`)**:
   - Compiles structured overviews, executive bullet points, and document summaries.
   - Tracks referenced documents and extracts high-level insights.
3. **Calculation Agent (`calculation_agent`)**:
   - Gathers exact numerical data points from documents.
   - Translates business questions into mathematical expressions.
   - Enforces the use of a sandboxed, deterministic calculator tool for all arithmetic rather than LLM guesswork.

### 2.3 Memory, Session Persistence & Graph Checkpointing
- **State Reducers**: Uses `operator.add` for `actions_taken: Annotated[list[str], operator.add]` to track execution provenance across nodes within each turn.
- **Context Summarization & Active Document Tracking (`update_memory`)**:
  - Distills conversation turns into a progressive rolling summary.
  - Keeps track of all document IDs actively discussed.
- **File-Backed Session Persistence (`sessions/` Directory)**:
  - Persists full session states (`sessions/<session_id>.json`) including turn history, parsed intents, specialist responses, execution lineages, rolling summaries, active documents, and pinned document contexts.
  - Enables users to resume previous sessions across CLI launches (`--session <id>` or `resume <id>`) and continue multi-turn threads seamlessly.
- **In-Memory Graph Checkpointing (`InMemorySaver`)**:
  - Uses LangGraph's checkpointer to persist thread state across invocations using `thread_id`.

### 2.4 Secure Tools & Document Engine
- **Safe Calculator Tool**: Evaluates mathematical expressions using an AST-based arithmetic parser (avoiding raw, unsafe `eval`).
- **Domain Document Engine**: In-memory retrieval engine with structured metadata for financial filings (invoices, revenue, balance sheets) and healthcare records (patient charges, insurance claims, clinical reports).
- **Document Search & Reader Tools**: LangChain `@tool` functions for keyword search, ID lookup, and range/attribute filtering.
- **Document Lifecycle Management (Add & Remove)**:
  - Supports soft memory unloading (`unload <id>`) and permanent deletion (`remove <id> --permanent`).
  - Implements benchmark seed protection (`SEED_PROTECTED_DOCS`) preventing accidental deletion of test documents without explicit `--force`.
  - Automatically unpins and clears active documents when a document is removed.

### 2.5 Multi-Format Document & Media Ingestion Engine (`src/parsers.py`)
- **PDF Documents (`.pdf`)**: Page-by-page text extraction, page demarcations (`--- [Page X of Y] ---`), author/subject metadata extraction via `pypdf`.
- **Excel Spreadsheets (`.xlsx`, `.xls`)**: Multi-worksheet extraction, sheet schema parsing, table markdown formatting, and statistical metric computation (sums, means, min/max) via `pandas` and `openpyxl`.
- **CSV Data Tables (`.csv`)**: Tabular parsing, markdown rendering without third-party `tabulate`, and automatic column-level numeric totals and summaries.
- **Infographic Images (`.png`, `.jpg`, `.jpeg`, `.webp`)**: Multimodal visual transcription via Google Gemini Vision (`ChatGoogleGenerativeAI`), extracting chart breakdowns, axes, percentages, financial metrics, callouts, and structured tables alongside Pillow image dimension metadata.
- **Unified Dispatcher (`parse_file`)**: Dynamically routes any supported file extension into typed `DocumentChunk` records with optional persistence into `document_corpus/`.

### 2.6 Multi-Document & Hybrid Scoping Engine
- **Multi-Document Pinning (`pinned_doc_ids`)**: Constrains queries strictly across multiple corpus documents simultaneously (e.g. cross-quarter financial comparisons).
- **Batch Document Ingestion**: Uploads and registers multiple files simultaneously in the Streamlit UI or CLI without overwriting previous uploads.
- **Hybrid Scoping Architecture**: Dynamically aggregates one or more uploaded external files alongside one or more existing corpus documents into a unified, strictly-grounded prompt context (`=== RESTRICTED TARGET DOCUMENTS SCOPE (HYBRID / MULTI-DOCUMENT SCOPE) ===`).
- **Interactive Multi-Select UI**: Replaces single-select dropdowns with `st.multiselect` in both sidebar and in-chat scopes, coupled with `st.chat_input(accept_file="multiple")` for drag-and-drop batch file attachments.

### 2.7 Session Document Boundary & Ephemeral Scope Isolation
- **Persistent Corpus Boundary (`_persisted_doc_ids`)**: `DocumentRepository` tracks permanently stored corpus documents loaded from `document_corpus/` or saved with `persist=True`.
- **Ephemeral Session Isolation (`_session_ephemeral_docs`)**: Any external document loaded without persistence (`persist=False`) is treated as session-ephemeral and bound strictly to the active `session_id`.
- **Scope Creep Prevention on Session Reset**: When starting a fresh session thread (`reset_session()`), `repo.clear_ephemeral_documents()` is automatically invoked, evicting all ephemeral documents from in-memory index so that fresh threads only see persistent corpus documents.
- **Session Re-hydration without Bleed**: Resuming a previous thread (`resume_session(session_id)`) clears foreign ephemeral files, then safely re-hydrates only that specific session's uploaded documents (serialized in `sessions/<session_id>.json` under `ephemeral_documents`).
- **Filtered Corpus Selector (`get_corpus_documents`)**: Ensures chat-level and corpus-level document dropdowns strictly present permanent corpus documents, preventing transient uploads from cluttering global search menus.

### 2.8 Automated Testing & Strict Execution Isolation (`tests/`)
- **Centralized Test Directory (`tests/`)**: All 29 unit and integration tests are organized under `tests/` with modular test suites covering document removal, hybrid scoping, multi-format parsing, anti-hallucination grounding, session isolation, session persistence, logging, and LangGraph workflow orchestration.
- **Dual Framework Support**: Fully compatible with both `pytest tests` and Python's standard `python -m unittest discover tests`.
- **Zero Production Pollution**: Test suites utilize temporary directories (`tempfile.mkdtemp`) and fixtures (`tests/conftest.py`) to isolate `sessions/`, `logs/assistant.log`, and `document_corpus/`. Test runs create zero mock artifacts in production directories and clean up file handles to prevent Windows file lock conflicts.

---

## 3. Technology Stack & Backend Customization

- **Orchestration**: `langgraph>=0.2.0`, `langchain>=0.2.0`, `langchain-core>=0.2.0`
- **LLM Provider**: **Google Gemini** via `langchain-google-genai` (`ChatGoogleGenerativeAI`, e.g., `gemini-1.5-flash` or `gemini-1.5-pro`) with native structured output (`with_structured_output`) and tool calling.
- **Validation**: `pydantic>=2.0.0`
- **Environment**: Python 3.12+ (isolated virtual environment)
- **Configuration**: `.env` managing `GEMINI_API_KEY` and `GEMINI_MODEL`.

---

## 4. Graph Workflow Architecture

```
                  ┌────────────────────────┐
                  │      User Request      │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │     classify_intent    │
                  └───────────┬────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │ [qa]            │ [summarization] │ [calculation]
            ▼                 ▼                 ▼
     ┌─────────────┐   ┌─────────────┐   ┌──────────────────┐
     │  qa_agent   │   │summarization│   │calculation_agent │
     │             │   │   _agent    │   │ (uses calc tool) │
     └──────┬──────┘   └──────┬──────┘   └────────┬─────────┘
            │                 │                   │
            └─────────────────┼───────────────────┘
                              ▼
                  ┌────────────────────────┐
                  │     update_memory      │
                  └───────────┬────────────┘
                              │
                              ▼
                           [ END ]
```

---

## 5. File & Module Structure

```
doc_assistant_project/
├── PROJECT_SPECIFICATION.md  # Detailed project blueprint & requirements (this document)
├── .env.example              # Template for API keys
├── .gitignore                # Python & virtual environment ignore rules
├── requirements.txt          # Python dependencies (langchain, langgraph, pypdf, openpyxl, pillow, pandas, reportlab)
├── main.py                   # Interactive CLI runner (sessions, doc pinning, rich output, multi-format loading)
├── app.py                    # Streamlit Web UI (conversational chat, visual metrics, in-chat attachments)
├── tests/                    # Fully isolated unit and integration test suite (29 tests)
│   ├── __init__.py           # Package marker
│   ├── conftest.py           # Pytest fixtures for temporary environment isolation
│   ├── test_document_removal.py      # Verification of document removal, safeguards & pin cleanup
│   ├── test_hybrid_and_multiscope.py # Verification of multi-document scope, batch loading, and hybrid scopes
│   ├── test_logging.py               # Verification of file-based logging under isolated logs/
│   ├── test_multiformat_ingestion.py # Verification of PDF, Excel, CSV, and Infographic Image parsing
│   ├── test_scoping_and_grounding.py # Verification of strict document scoping & missing data handling
│   ├── test_session_isolation.py     # Verification of ephemeral document boundaries and session reset isolation
│   ├── test_session_persistence.py   # Verification of sessions/ storage and session resumption
│   └── test_workflow.py              # End-to-end multi-agent LangGraph workflow execution
├── sessions/                 # Persisted JSON conversation sessions
├── logs/                     # Persistent log files (assistant.log with log rotation)
├── examples/                 # Sample external documents for loading/unloading tests
│   ├── sample_vendor_invoice.json
│   ├── sample_clinical_memo.md
│   ├── sample_inventory.csv
│   ├── sample_financial_model.xlsx
│   ├── sample_quarterly_report.pdf
│   └── sample_kpi_infographic.png
├── document_corpus/          # Dynamic filesystem repository for documents (.json, .md, .txt, .pdf, .xlsx, .csv)
│   ├── FIN-2024-Q1.json
│   ├── FIN-2024-Q2.json
│   ├── FIN-INV-1092.json
│   ├── FIN-BUD-003.json
│   ├── MED-CLM-7701.json
│   ├── MED-CLM-8842.json
│   ├── MED-REP-301.json
│   └── MED-EQP-502.json
└── src/
    ├── __init__.py           # Package initializer
    ├── logger.py             # Dual stream & rotating file handler configuration under logs/
    ├── schemas.py            # Pydantic data contracts (Intent, Responses, State)
    ├── parsers.py            # Multi-format parsers (PDF, Excel, CSV, Gemini Multimodal Vision Infographics)
    ├── retrieval.py          # Dynamic document repository, search engine & lifecycle manager
    ├── tools.py              # LangChain @tool definitions (safe calculator, doc tools)
    ├── prompts.py            # System prompts with dynamic document injection
    ├── agent.py              # LangGraph StateGraph, nodes, edges & compilation
    ├── session_manager.py    # Session serialization, storage, and resumption under sessions/
    └── assistant.py          # Assistant orchestrator, session handling & checkpointer setup
```

---
