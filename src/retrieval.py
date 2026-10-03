"""
Document repository and dynamic filesystem retrieval engine.
Dynamically discovers and indexes documents from the external `document_corpus/`
directory, providing keyword search, ID resolution, and numerical attribute filtering.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.parsers import parse_file

logger = logging.getLogger("doc_assistant.retrieval")

# Default path to the external document corpus folder anchored to repository root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS_DIR = PROJECT_ROOT / "document_corpus"

# Base benchmark documents protected from accidental permanent deletion
SEED_PROTECTED_DOCS: Set[str] = {
    "FIN-2024-Q1",
    "FIN-2024-Q2",
    "FIN-INV-1092",
    "FIN-BUD-003",
    "MED-CLM-7701",
    "MED-CLM-8842",
    "MED-REP-301",
    "MED-EQP-502",
}


def _format_rel_path(path: Union[str, Path]) -> str:
    """Formats a path relative to PROJECT_ROOT for clean, portable cross-machine logging."""
    p = Path(path)
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


@dataclass
class DocumentChunk:
    """Represents a discrete document or chunk returned from retrieval queries."""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 1.0

    @property
    def title(self) -> str:
        return self.metadata.get("title", self.doc_id)

    @property
    def doc_type(self) -> str:
        return self.metadata.get("doc_type", "general")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "doc_type": self.doc_type,
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
        }


@dataclass
class Document:
    """Internal canonical representation of a document in the repository."""
    doc_id: str
    title: str
    doc_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Dynamic Document Retrieval Engine Class
# ============================================================================

class DocumentRepository:
    """
    Filesystem-backed document repository that dynamically loads and indexes
    documents from the `document_corpus/` folder without hardcoding.
    """

    def __init__(self, corpus_dir: Optional[Path | str] = None):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else DEFAULT_CORPUS_DIR
        self._docs: Dict[str, Document] = {}
        self._persisted_doc_ids: Set[str] = set()
        self.load_from_directory()

    def load_from_directory(self) -> int:
        """
        Scans the corpus directory and dynamically indexes all .json, .md, and .txt files.
        Returns the number of documents loaded.
        """
        self._docs.clear()
        self._persisted_doc_ids.clear()
        if not self.corpus_dir.exists():
            logger.warning(f"Corpus directory '{_format_rel_path(self.corpus_dir)}' does not exist. Creating it.")
            self.corpus_dir.mkdir(parents=True, exist_ok=True)
            return 0

        loaded_count = 0
        for file_path in self.corpus_dir.iterdir():
            if not file_path.is_file():
                continue

            try:
                ext = file_path.suffix.lower()
                if ext == ".json":
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        doc_id = data.get("doc_id", file_path.stem).upper()
                        doc = Document(
                            doc_id=doc_id,
                            title=data.get("title", file_path.stem),
                            doc_type=data.get("doc_type", "general"),
                            content=data.get("content", ""),
                            metadata=data.get("metadata", {})
                        )
                        self._docs[doc_id] = doc
                        self._persisted_doc_ids.add(doc_id)
                        loaded_count += 1

                elif ext in (".md", ".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_text = f.read()
                        doc_id = file_path.stem.upper()
                        doc = Document(
                            doc_id=doc_id,
                            title=file_path.stem.replace("_", " ").title(),
                            doc_type="text_document",
                            content=raw_text,
                            metadata={"filename": file_path.name}
                        )
                        self._docs[doc_id] = doc
                        self._persisted_doc_ids.add(doc_id)
                        loaded_count += 1

                elif ext in (".pdf", ".xlsx", ".xls", ".csv", ".png", ".jpg", ".jpeg", ".webp"):
                    doc_id = file_path.stem.upper()
                    # Only index if not already indexed via companion/persisted json
                    if doc_id not in self._docs:
                        content, dtype, meta = parse_file(file_path)
                        doc = Document(
                            doc_id=doc_id,
                            title=meta.get("title", file_path.stem.replace("_", " ").title()),
                            doc_type=dtype,
                            content=content,
                            metadata=meta
                        )
                        self._docs[doc_id] = doc
                        self._persisted_doc_ids.add(doc_id)
                        loaded_count += 1

            except Exception as e:
                logger.error(f"Error reading document file '{file_path.name}': {e}")

        logger.info(f"Dynamically indexed {loaded_count} documents from '{_format_rel_path(self.corpus_dir)}'.")
        return loaded_count

    def reload(self) -> int:
        """Refreshes the document repository from the filesystem."""
        return self.load_from_directory()

    def add_document(
        self,
        doc_id: str,
        title: str,
        doc_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        persist: bool = False
    ) -> DocumentChunk:
        """
        Adds a document directly to the repository in-memory.
        If persist=True, saves the document as JSON into document_corpus/<doc_id>.json.
        """
        clean_id = doc_id.strip().upper()
        doc_meta = metadata or {}
        doc = Document(
            doc_id=clean_id,
            title=title,
            doc_type=doc_type,
            content=content,
            metadata=doc_meta
        )
        self._docs[clean_id] = doc

        if persist:
            self.persist_document(doc)
            self._persisted_doc_ids.add(clean_id)

        logger.info(f"Added document '{clean_id}' (persist={persist}).")
        return DocumentChunk(
            doc_id=clean_id,
            content=content,
            metadata={"title": title, "doc_type": doc_type, **doc_meta},
            relevance_score=1.0
        )

    def persist_document(self, doc: Document) -> Path:
        """Writes a document object to the corpus directory as a JSON file."""
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.corpus_dir / f"{doc.doc_id}.json"
        payload = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "doc_type": doc.doc_type,
            "content": doc.content,
            "metadata": doc.metadata
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        self._persisted_doc_ids.add(doc.doc_id)
        logger.info(f"Persisted document '{doc.doc_id}' to '{_format_rel_path(file_path)}'.")
        return file_path

    def remove_document(
        self,
        doc_id: str,
        permanent: bool = False,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Removes a document from the repository.

        Args:
            doc_id: The identifier of the document to remove.
            permanent: If True, also permanently deletes matching file(s) from document_corpus/.
                       If False (default), only unloads the document from active memory.
            force: If True, allows permanent deletion of protected seed documents.

        Returns:
            Tuple[bool, str]: (Success boolean, human-readable status message)
        """
        clean_id = doc_id.strip().upper()
        in_memory = clean_id in self._docs

        # Scan for matching corpus files
        matching_files = list(self.corpus_dir.glob(f"{clean_id}.*")) if self.corpus_dir.exists() else []

        if not in_memory and not matching_files:
            return False, f"Document '{clean_id}' was not found in memory or in '{self.corpus_dir.name}'."

        if permanent and clean_id in SEED_PROTECTED_DOCS and not force:
            return (
                False,
                f"Document '{clean_id}' is a protected benchmark seed document. "
                "Permanent deletion was blocked. Pass force=True (or --force in CLI) to override."
            )

        # 1. Unload from active memory
        if in_memory:
            del self._docs[clean_id]
            logger.info(f"Unloaded document '{clean_id}' from memory.")

        # 2. Delete file from corpus directory if permanent removal requested
        deleted_files: List[str] = []
        if permanent:
            self._persisted_doc_ids.discard(clean_id)
            for f in matching_files:
                try:
                    f.unlink()
                    try:
                        rel_path = f.relative_to(PROJECT_ROOT).as_posix()
                    except ValueError:
                        rel_path = f.as_posix()
                    deleted_files.append(rel_path)
                    logger.info(f"Permanently deleted corpus file: {rel_path}")
                except Exception as ex:
                    logger.error(f"Failed to delete corpus file '{f.name}': {ex}")
                    return False, f"Failed to delete '{f.name}': {ex}"

        if permanent:
            file_info = f" (deleted file(s): {', '.join(deleted_files)})" if deleted_files else " (no disk file found)"
            return True, f"Permanently removed document '{clean_id}'{file_info}."
        else:
            return True, f"Unloaded document '{clean_id}' from active memory. (Disk file preserved in {self.corpus_dir.name}/)."

    def clear_ephemeral_documents(self) -> int:
        """
        Removes all non-persisted (ephemeral) documents from the in-memory index.
        Documents permanently stored in document_corpus/ remain untouched.
        Returns the count of cleared ephemeral documents.
        """
        ephemeral_ids = [did for did in self._docs if did not in self._persisted_doc_ids]
        for did in ephemeral_ids:
            del self._docs[did]
        if ephemeral_ids:
            logger.info(f"Cleared {len(ephemeral_ids)} ephemeral documents from repository memory: {ephemeral_ids}")
        return len(ephemeral_ids)

    def is_persisted(self, doc_id: str) -> bool:
        """Checks if a document ID corresponds to a permanently stored corpus document."""
        return doc_id.strip().upper() in self._persisted_doc_ids

    def get_corpus_documents(self) -> List[DocumentChunk]:
        """Return only permanently persisted corpus documents (excluding session-ephemeral uploads)."""
        return [
            DocumentChunk(
                doc_id=d.doc_id,
                content=d.content,
                metadata={"title": d.title, "doc_type": d.doc_type, **d.metadata},
                relevance_score=1.0
            )
            for d in self._docs.values()
            if d.doc_id in self._persisted_doc_ids
        ]

    def load_external_file(
        self,
        file_path: str | Path,
        doc_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        persist: bool = False,
        llm: Optional[Any] = None
    ) -> DocumentChunk:
        """
        Loads an external file from disk (.pdf, .xlsx, .xls, .csv, .png, .jpg, .json, .md, .txt)
        and registers it in the repository.
        Optionally persists it to document_corpus/.
        """
        p = Path(file_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"External document file not found at: {p}")

        clean_id = (doc_id or f"EXT-{p.stem}").strip().upper()
        content, resolved_type, meta = parse_file(p, doc_type=doc_type, llm=llm)
        title = meta.get("title") or p.stem.replace("_", " ").title()

        return self.add_document(
            doc_id=clean_id,
            title=title,
            doc_type=resolved_type,
            content=content,
            metadata=meta,
            persist=persist
        )

    def get_available_documents_summary(self) -> str:
        """
        Dynamically generates a formatted bullet list of all currently indexed
        documents for injection into LLM router prompts.
        """
        if not self._docs:
            return "No documents currently indexed in the repository."

        lines = []
        for doc in self._docs.values():
            key_meta = []
            if "amount" in doc.metadata:
                key_meta.append(f"amount: ${doc.metadata['amount']:,.2f}")
            if "period" in doc.metadata:
                key_meta.append(f"period: {doc.metadata['period']}")
            if "department" in doc.metadata:
                key_meta.append(f"dept: {doc.metadata['department']}")

            meta_str = f" [{', '.join(key_meta)}]" if key_meta else ""
            lines.append(f"- {doc.doc_id}: {doc.title} (Type: {doc.doc_type}){meta_str}")

        return "\n".join(lines)

    def get_document_by_id(self, doc_id: str) -> Optional[DocumentChunk]:
        """Fetch an individual document by its exact or case-insensitive doc_id."""
        clean_id = doc_id.strip().upper()
        doc = self._docs.get(clean_id)
        if doc:
            return DocumentChunk(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata={"title": doc.title, "doc_type": doc.doc_type, **doc.metadata},
                relevance_score=1.0
            )
        return None

    def search_by_keyword(self, query: str, limit: int = 4) -> List[DocumentChunk]:
        """
        Search document corpus using token-weighted keyword matching and return
        ranked DocumentChunk objects.
        """
        if not query or not query.strip():
            return self.get_all_documents()[:limit]

        query_tokens = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-\$]+\b', query) if len(w) > 1]
        if not query_tokens:
            return self.get_all_documents()[:limit]

        scored_results: List[DocumentChunk] = []

        for doc in self._docs.values():
            score = 0.0
            searchable_text = f"{doc.doc_id} {doc.title} {doc.doc_type} {doc.content}".lower()

            for token in query_tokens:
                # Direct ID match gets highest weight
                if token == doc.doc_id.lower():
                    score += 5.0
                elif token in doc.title.lower():
                    score += 2.5
                elif token in searchable_text:
                    score += 1.0

            if score > 0.0:
                scored_results.append(
                    DocumentChunk(
                        doc_id=doc.doc_id,
                        content=doc.content,
                        metadata={"title": doc.title, "doc_type": doc.doc_type, **doc.metadata},
                        relevance_score=round(min(score / (len(query_tokens) * 2.0), 1.0), 3)
                    )
                )

        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_results[:limit]

    def retrieve_by_amount_range(
        self,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None
    ) -> List[DocumentChunk]:
        """Filter documents whose primary financial amount falls within [min_amount, max_amount]."""
        results: List[DocumentChunk] = []
        for doc in self._docs.values():
            amt = doc.metadata.get("amount")
            if amt is not None:
                matches = True
                if min_amount is not None and amt < min_amount:
                    matches = False
                if max_amount is not None and amt > max_amount:
                    matches = False
                if matches:
                    results.append(
                        DocumentChunk(
                            doc_id=doc.doc_id,
                            content=doc.content,
                            metadata={"title": doc.title, "doc_type": doc.doc_type, **doc.metadata},
                            relevance_score=1.0
                        )
                    )
        results.sort(key=lambda x: x.metadata.get("amount", 0), reverse=True)
        return results

    def retrieve_by_doc_type(self, doc_type: str) -> List[DocumentChunk]:
        """Filter documents by specific document category."""
        target = doc_type.strip().lower()
        return [
            DocumentChunk(
                doc_id=d.doc_id,
                content=d.content,
                metadata={"title": d.title, "doc_type": d.doc_type, **d.metadata},
                relevance_score=1.0
            )
            for d in self._docs.values()
            if target in d.doc_type.lower()
        ]

    def get_all_documents(self) -> List[DocumentChunk]:
        """Return all documents currently indexed in the repository."""
        return [
            DocumentChunk(
                doc_id=d.doc_id,
                content=d.content,
                metadata={"title": d.title, "doc_type": d.doc_type, **d.metadata},
                relevance_score=1.0
            )
            for d in self._docs.values()
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate statistical summary of documents in the repository."""
        total_docs = len(self._docs)
        amounts = [d.metadata["amount"] for d in self._docs.values() if "amount" in d.metadata]
        types: Dict[str, int] = {}
        for d in self._docs.values():
            types[d.doc_type] = types.get(d.doc_type, 0) + 1

        return {
            "corpus_directory": _format_rel_path(self.corpus_dir),
            "total_documents": total_docs,
            "documents_with_amounts": len(amounts),
            "total_dollar_volume": sum(amounts),
            "average_amount": round(sum(amounts) / len(amounts), 2) if amounts else 0.0,
            "document_types": types
        }


# Global repository singleton pointing to the external document_corpus/ folder
default_repository = DocumentRepository()
