"""
Assistant Orchestration Layer.
Manages session state, thread_id persistence, LLM model configuration,
document pinning/ad-hoc ingestion, and message execution through the compiled LangGraph workflow.
"""

from __future__ import annotations

import os
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

from src.agent import create_workflow
from src.retrieval import DocumentChunk, Document, DocumentRepository, default_repository
from src.schemas import (
    AnswerResponse,
    CalculationResponse,
    SummaryResponse,
    UserIntent,
)
from src.logger import setup_logging
from src.session_manager import SessionManager
from src.tools import get_default_tools

logger = logging.getLogger("docgrok.assistant")


class DocumentAssistant:
    """
    High-level conversational DocGrok Assistant orchestrator.
    Manages session threads, invokes the LangGraph state machine,
    and formats responses for presentation.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        repository: Optional[DocumentRepository] = None,
        temperature: float = 0.0,
        sessions_dir: Union[str, Path] = "sessions",
        logs_dir: Union[str, Path] = "logs",
        session_id: Optional[str] = None
    ):
        # Initialize file and console logging
        setup_logging(log_dir=logs_dir)

        # Load environment variables safely
        load_dotenv()

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in your .env file or pass it to DocumentAssistant."
            )

        env_model = os.getenv("GEMINI_MODEL")
        if not env_model or "1.5" in env_model or "2.5" in env_model or "3.8" in env_model:
            env_model = "gemini-flash-lite-latest"
        self.model_name = model_name or env_model
        self.temperature = temperature
        self.repo = repository or default_repository
        self.tools = get_default_tools(self.repo)

        # Initialize Google Gemini Chat Model with temperature=0.0 for strict factual accuracy
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=self.temperature,
        )

        # Compile LangGraph Workflow with persistent memory
        self.workflow = create_workflow()

        # Session tracking & persistence under sessions/ folder
        self.session_manager = SessionManager(sessions_dir=sessions_dir)
        self.user_id: str = os.getenv("DEFAULT_USER_ID", "user_default")
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self.pinned_doc_ids: List[str] = []
        self.conversation_summary: str = ""
        self.active_documents: List[str] = []
        self._session_ephemeral_docs: Dict[str, Dict[str, Document]] = {}

        # If existing session ID provided, resume its state
        if session_id and self.session_manager.session_exists(session_id):
            self.resume_session(session_id)
        else:
            # Initialize session record on disk
            self.session_manager.create_session(session_id=self.session_id, user_id=self.user_id)

    @property
    def pinned_doc_id(self) -> Optional[str]:
        """Backward-compatible access returning the primary pinned document ID."""
        return self.pinned_doc_ids[0] if self.pinned_doc_ids else None

    @pinned_doc_id.setter
    def pinned_doc_id(self, val: Optional[str]):
        """Backward-compatible setter for pinning a single document ID."""
        if val:
            clean = val.strip().upper()
            self.pinned_doc_ids = [clean]
        else:
            self.pinned_doc_ids = []

    def resume_session(self, session_id: str) -> bool:
        """
        Resumes a previous conversation session from sessions/<session_id>.json.
        Clears ephemeral documents from other sessions and restores thread context,
        pinned documents, conversation summary, and this session's ephemeral documents.
        """
        session_data = self.session_manager.load_session(session_id)
        if not session_data:
            logger.warning(f"Could not find session {session_id} to resume.")
            return False

        # 1. Clear any ephemeral documents belonging to previous/other sessions
        self.repo.clear_ephemeral_documents()

        self.session_id = session_id
        pinned_ids = session_data.get("pinned_doc_ids")
        if pinned_ids is not None:
            self.pinned_doc_ids = list(pinned_ids)
        elif session_data.get("pinned_doc_id"):
            self.pinned_doc_ids = [session_data["pinned_doc_id"]]
        else:
            self.pinned_doc_ids = []

        self.conversation_summary = session_data.get("conversation_summary", "")
        self.active_documents = list(session_data.get("active_documents", []))

        # 2. Re-hydrate ephemeral documents for this session
        if "ephemeral_documents" in session_data and session_data["ephemeral_documents"]:
            for edoc in session_data["ephemeral_documents"].values():
                doc = Document(
                    doc_id=edoc["doc_id"],
                    title=edoc.get("title", edoc["doc_id"]),
                    doc_type=edoc.get("doc_type", "general"),
                    content=edoc.get("content", ""),
                    metadata=edoc.get("metadata", {})
                )
                self._session_ephemeral_docs.setdefault(session_id, {})[doc.doc_id] = doc
                self.repo.add_document(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    doc_type=doc.doc_type,
                    content=doc.content,
                    metadata=doc.metadata,
                    persist=False
                )
        elif session_id in self._session_ephemeral_docs:
            for doc in self._session_ephemeral_docs[session_id].values():
                self.repo.add_document(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    doc_type=doc.doc_type,
                    content=doc.content,
                    metadata=doc.metadata,
                    persist=False
                )

        logger.info(
            f"Resumed session '{session_id}': {len(session_data.get('turns', []))} turns, "
            f"pinned_docs={self.pinned_doc_ids}, active_docs={self.active_documents}"
        )
        return True

    def reset_session(self, new_session_id: Optional[str] = None) -> str:
        """
        Resets the conversation thread with a fresh session_id and clears in-memory context.
        Evicts all ephemeral documents from the repository to prevent scope creep.
        """
        # Prune all ephemeral documents from repository memory
        self.repo.clear_ephemeral_documents()

        self.session_id = new_session_id or str(uuid.uuid4())[:8]
        self.pinned_doc_ids = []
        self.conversation_summary = ""
        self.active_documents = []
        self.session_manager.create_session(session_id=self.session_id, user_id=self.user_id)
        logger.info(f"Reset session to fresh thread_id: {self.session_id} (cleared ephemeral repository documents)")
        return self.session_id

    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """Lists all persisted sessions available under the sessions/ directory."""
        return self.session_manager.list_sessions()

    def get_session_history(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieves the full persisted history and turns for a given session."""
        target_sid = session_id or self.session_id
        return self.session_manager.load_session(target_sid)

    def pin_document(self, doc_id: str) -> bool:
        """
        Pins an existing document by ID. Appends to active pinned set.
        """
        clean_id = doc_id.strip().upper()
        if self.repo.get_document_by_id(clean_id):
            if clean_id not in self.pinned_doc_ids:
                self.pinned_doc_ids.append(clean_id)
            logger.info(f"Pinned document scope to: {self.pinned_doc_ids}")
            return True
        return False

    def pin_documents(self, doc_ids: List[str]) -> List[str]:
        """
        Pins multiple documents by ID, replacing the active pinned set.
        """
        valid_ids: List[str] = []
        for d_id in doc_ids:
            clean_id = d_id.strip().upper()
            if self.repo.get_document_by_id(clean_id) and clean_id not in valid_ids:
                valid_ids.append(clean_id)
        self.pinned_doc_ids = valid_ids
        logger.info(f"Pinned multi-document scope to: {self.pinned_doc_ids}")
        return self.pinned_doc_ids

    def unpin_document(self, doc_id: Optional[str] = None) -> List[str]:
        """
        Unpins a specific document ID or clears all pinned documents if None.
        Returns the remaining pinned document IDs.
        """
        if doc_id:
            clean_id = doc_id.strip().upper()
            self.pinned_doc_ids = [d for d in self.pinned_doc_ids if d != clean_id]
        else:
            self.pinned_doc_ids = []
        logger.info(f"Updated pinned document scope: {self.pinned_doc_ids}")
        return self.pinned_doc_ids

    def load_external_document(
        self,
        file_path: str | Path,
        persist: bool = False,
        doc_id: Optional[str] = None
    ) -> DocumentChunk:
        """
        Loads an external file from disk (.pdf, .xlsx, .xls, .csv, .png, .jpg, .json, .md, .txt)
        and registers it in the active repository.
        Optionally persists it permanently to the document_corpus directory.
        """
        chunk = self.repo.load_external_file(
            file_path=file_path,
            doc_id=doc_id,
            persist=persist,
            llm=self.llm
        )
        if not persist:
            doc = self.repo._docs.get(chunk.doc_id)
            if doc:
                self._session_ephemeral_docs.setdefault(self.session_id, {})[chunk.doc_id] = doc
        return chunk

    def load_external_documents(
        self,
        file_paths: List[Union[str, Path]],
        persist: bool = False,
        doc_ids: Optional[List[str]] = None
    ) -> List[DocumentChunk]:
        """
        Loads multiple external files from disk in a single batch.
        Optionally persists them permanently to document_corpus/.
        """
        chunks: List[DocumentChunk] = []
        for i, fp in enumerate(file_paths):
            cid = doc_ids[i] if (doc_ids and i < len(doc_ids)) else None
            chunk = self.load_external_document(file_path=fp, persist=persist, doc_id=cid)
            chunks.append(chunk)
        return chunks

    def remove_document(
        self,
        doc_id: str,
        permanent: bool = False,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Removes a document from repository and unpins it if currently active.
        """
        clean_id = doc_id.strip().upper()
        # Automatically unpin if this document is currently pinned
        if clean_id in self.pinned_doc_ids:
            self.pinned_doc_ids = [d for d in self.pinned_doc_ids if d != clean_id]
            logger.info(f"Automatically unpinned document '{clean_id}' due to removal.")

        if clean_id in self.active_documents:
            self.active_documents = [d for d in self.active_documents if d != clean_id]

        # Clean from session ephemeral tracker if present
        if self.session_id in self._session_ephemeral_docs:
            self._session_ephemeral_docs[self.session_id].pop(clean_id, None)

        success, message = self.repo.remove_document(doc_id=clean_id, permanent=permanent, force=force)
        return success, message

    def process_message(
        self,
        user_input: str,
        target_document: Optional[Union[Dict[str, Any], DocumentChunk, Document, List[Any]]] = None,
        target_doc_id: Optional[Union[str, List[str]]] = None,
        target_documents: Optional[List[Dict[str, Any]]] = None,
        target_doc_ids: Optional[List[str]] = None,
        persist_target_document: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes a single user message through the LangGraph workflow.
        
        Document Scoping Behavior:
        - If `target_document(s)` provided, model includes those uploaded documents.
        - If `target_doc_id(s)` (or active self.pinned_doc_ids) provided, model includes those corpus documents.
        - If BOTH are provided, executes in a HYBRID scope strictly restricted to the combined document set.
        - If no document is provided, dynamically retrieves relevant documents from the available corpus.
        """
        active_thread_id = session_id or self.session_id

        # Normalize external target documents to a list of dicts
        all_target_docs: List[Dict[str, Any]] = []
        if target_documents:
            for td in target_documents:
                if isinstance(td, dict):
                    all_target_docs.append(td)
                elif hasattr(td, "to_dict"):
                    all_target_docs.append(td.to_dict())

        if target_document:
            td_list = target_document if isinstance(target_document, list) else [target_document]
            for td in td_list:
                if isinstance(td, dict):
                    all_target_docs.append(td)
                elif hasattr(td, "to_dict"):
                    all_target_docs.append(td.to_dict())
                elif isinstance(td, Document):
                    all_target_docs.append({
                        "doc_id": td.doc_id,
                        "title": td.title,
                        "doc_type": td.doc_type,
                        "content": td.content,
                        "metadata": td.metadata
                    })

        # Normalize target document IDs
        all_target_ids: List[str] = []
        if target_doc_ids:
            for tid in target_doc_ids:
                clean = tid.strip().upper()
                if clean and clean not in all_target_ids:
                    all_target_ids.append(clean)

        if target_doc_id:
            tid_list = target_doc_id if isinstance(target_doc_id, list) else [target_doc_id]
            for tid in tid_list:
                clean = tid.strip().upper()
                if clean and clean not in all_target_ids:
                    all_target_ids.append(clean)

        # Fallback to active pinned doc IDs if no explicit target IDs or external docs provided
        if not all_target_ids and not all_target_docs and self.pinned_doc_ids:
            all_target_ids = list(self.pinned_doc_ids)

        # Runtime configuration passed to all graph nodes
        config = {
            "configurable": {
                "thread_id": active_thread_id,
                "llm": self.llm,
                "repo": self.repo,
                "tools": self.tools,
            }
        }

        # Initial turn inputs (injecting prior session memory if available)
        initial_state = {
            "user_input": user_input,
            "session_id": active_thread_id,
            "user_id": self.user_id,
            "conversation_summary": self.conversation_summary,
            "active_documents": list(self.active_documents),
            "target_document": all_target_docs[0] if len(all_target_docs) == 1 else None,
            "target_documents": all_target_docs,
            "target_doc_id": all_target_ids[0] if len(all_target_ids) == 1 else None,
            "target_doc_ids": all_target_ids,
            "persist_target_document": persist_target_document,
            "messages": [HumanMessage(content=user_input)],
        }

        # Execute workflow through compiled graph
        result_state = self.workflow.invoke(initial_state, config=config)

        # Update in-memory session tracker with latest progressive memory
        self.conversation_summary = result_state.get("conversation_summary", self.conversation_summary)
        self.active_documents = result_state.get("active_documents", self.active_documents)

        # Persist updated session state to sessions/<session_id>.json
        effective_scope = all_target_ids if all_target_ids else ([d["doc_id"] for d in all_target_docs] if all_target_docs else None)
        ephemeral_payload = {
            did: {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "doc_type": doc.doc_type,
                "content": doc.content,
                "metadata": doc.metadata
            }
            for did, doc in self._session_ephemeral_docs.get(active_thread_id, {}).items()
        }
        self.session_manager.record_turn(
            session_id=active_thread_id,
            user_input=user_input,
            result_state=result_state,
            pinned_doc_id=effective_scope,
            ephemeral_documents=ephemeral_payload
        )

        return result_state

    @staticmethod
    def format_output(state: Dict[str, Any]) -> str:
        """
        Renders the final result_state into a well-formatted string for terminal or UI.
        Explicitly flags missing information, multi-document scopes, or hybrid scopes.
        """
        intent: Optional[UserIntent] = state.get("intent")
        current_resp = state.get("current_response")
        actions_taken = state.get("actions_taken", [])
        active_docs = state.get("active_documents", [])

        # Gather target documents and IDs
        target_docs = state.get("target_documents") or []
        if not target_docs and state.get("target_document"):
            td = state.get("target_document")
            target_docs = td if isinstance(td, list) else [td]

        target_ids = state.get("target_doc_ids") or []
        if not target_ids and state.get("target_doc_id"):
            tid = state.get("target_doc_id")
            target_ids = tid if isinstance(tid, list) else [tid]

        output_parts = []

        # Scope formatting
        all_scoped_display = []
        for d in target_docs:
            if isinstance(d, dict):
                all_scoped_display.append(f"📎 {d.get('doc_id', 'UPLOADED')}")
        for tid in target_ids:
            all_scoped_display.append(f"📁 {tid}")

        if all_scoped_display:
            if len(all_scoped_display) == 1:
                output_parts.append(f"[bold magenta]Scope Restricted To Document:[/bold magenta] [yellow]{all_scoped_display[0]}[/yellow]")
            else:
                is_hybrid = bool(target_docs) and bool(target_ids)
                scope_prefix = "Hybrid Scope Restricted To" if is_hybrid else f"Scope Restricted To {len(all_scoped_display)} Documents"
                output_parts.append(f"[bold magenta]{scope_prefix}:[/bold magenta] [yellow]{', '.join(all_scoped_display)}[/yellow]")
        else:
            output_parts.append("[dim cyan]Scope:[/dim cyan] [dim]Dynamic Corpus Search[/dim]")

        # Intent Header
        if intent:
            output_parts.append(
                f"[bold cyan]Intent:[/bold cyan] {intent.intent_type.upper()} "
                f"(Confidence: {intent.confidence:.0%}) | [dim]Reasoning: {intent.reasoning}[/dim]"
            )

        # Execution Lineage (from operator.add reducer)
        lineage_str = " -> ".join(actions_taken)
        output_parts.append(f"[dim]Workflow Lineage:[/dim] [yellow]{lineage_str}[/yellow]\n")

        # Specialist Content & Grounding Checks
        if isinstance(current_resp, AnswerResponse):
            if not current_resp.found or current_resp.confidence == 0.0:
                output_parts.append(f"[bold yellow]Grounding Notice:[/bold yellow] [italic red]{current_resp.answer}[/italic red]\n")
            else:
                output_parts.append(f"[bold green]Answer:[/bold green]\n{current_resp.answer}\n")

            if current_resp.sources:
                output_parts.append(f"[bold]Sources:[/bold] {', '.join(current_resp.sources)}")

        elif isinstance(current_resp, SummaryResponse):
            if not current_resp.found:
                output_parts.append(f"[bold yellow]Grounding Notice:[/bold yellow] [italic red]{current_resp.summary}[/italic red]\n")
            else:
                output_parts.append(f"[bold green]Summary: {current_resp.title}[/bold green]\n{current_resp.summary}\n")
                if current_resp.key_points:
                    output_parts.append("[bold]Key Highlights:[/bold]")
                    for pt in current_resp.key_points:
                        output_parts.append(f"  • {pt}")
            if current_resp.sources:
                output_parts.append(f"\n[bold]Sources:[/bold] {', '.join(current_resp.sources)}")

        elif isinstance(current_resp, CalculationResponse):
            if not current_resp.found:
                output_parts.append(f"[bold yellow]Grounding Notice:[/bold yellow] [italic red]{current_resp.explanation}[/italic red]")
                output_parts.append("[dim]Calculation unavailable due to missing required data in document.[/dim]")
            else:
                output_parts.append(f"[bold green]Calculation Result:[/bold green] [bold cyan]{current_resp.formatted_result}[/bold cyan]")
                output_parts.append(f"[dim]Formula Evaluated:[/dim] {current_resp.expression} = {current_resp.result}")
                output_parts.append(f"\n[bold]Explanation:[/bold]\n{current_resp.explanation}")
                if current_resp.sources:
                    output_parts.append(f"\n[bold]Sources:[/bold] {', '.join(current_resp.sources)}")

        else:
            output_parts.append(f"[green]{str(current_resp)}[/green]")

        if active_docs:
            output_parts.append(f"\n[dim]Contextual Active Docs: {', '.join(active_docs)}[/dim]")

        return "\n".join(output_parts)


# DocGrok alias exports
DocGrokAssistant = DocumentAssistant
DocGrok = DocumentAssistant
