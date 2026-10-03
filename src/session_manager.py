"""
Session State Persistence Manager.
Persists and reloads conversation sessions as structured JSON files under the sessions/ directory,
enabling users to resume multi-turn conversations and maintain context across application restarts.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Determine Project Root anchored to repository location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SESSIONS_DIR = PROJECT_ROOT / "sessions"

logger = logging.getLogger("doc_assistant.session_manager")


class SessionManager:
    """
    Manages persistent session storage under the sessions/ directory.
    Stores thread lineage, conversation summaries, active document sets,
    pinned document contexts, and historical turn exchanges.
    """

    def __init__(self, sessions_dir: Union[str, Path] = DEFAULT_SESSIONS_DIR):
        path_obj = Path(sessions_dir)
        self.sessions_dir = path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj).resolve()
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Returns the canonical file path for a session ID."""
        clean_id = session_id.strip().replace(" ", "_")
        return self.sessions_dir / f"{clean_id}.json"

    def session_exists(self, session_id: str) -> bool:
        """Checks if a session file already exists on disk."""
        return self._get_session_path(session_id).is_file()

    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: str = "user_default"
    ) -> Dict[str, Any]:
        """Initializes a new session data structure and saves it to disk."""
        sid = session_id or str(uuid.uuid4())[:8]
        now_iso = datetime.now(timezone.utc).isoformat()

        session_data: Dict[str, Any] = {
            "session_id": sid,
            "user_id": user_id,
            "created_at": now_iso,
            "updated_at": now_iso,
            "pinned_doc_id": None,
            "conversation_summary": "",
            "active_documents": [],
            "ephemeral_documents": {},
            "turns": [],
            "metadata": {
                "total_turns": 0,
                "version": "1.0"
            }
        }
        self.save_session(sid, session_data)
        logger.info(f"Initialized new persistent session: {sid}")
        return session_data

    def save_session(self, session_id: str, session_data: Dict[str, Any]) -> Path:
        """Saves session state dictionary to sessions/<session_id>.json."""
        target_path = self._get_session_path(session_id)
        session_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        session_data["metadata"]["total_turns"] = len(session_data.get("turns", []))

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False, default=str)

        try:
            rel_display = target_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel_display = str(target_path)
        logger.debug(f"Saved session state to {rel_display}")
        return target_path

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads and parses a session state dictionary from disk."""
        path = self._get_session_path(session_id)
        if not path.is_file():
            try:
                rel_display = path.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                rel_display = str(path)
            logger.warning(f"Session file not found: {rel_display}")
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded session '{session_id}' with {len(data.get('turns', []))} turns")
            return data
        except Exception as e:
            logger.error(f"Failed to read session file {path}: {e}")
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        Lists all available persisted sessions sorted by updated_at (most recent first).
        Returns metadata summaries for quick display.
        """
        summaries: List[Dict[str, Any]] = []

        if not self.sessions_dir.exists():
            return summaries

        for fpath in self.sessions_dir.glob("*.json"):
            try:
                try:
                    rel_path = fpath.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    rel_path = str(fpath)

                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                turns = data.get("turns", [])
                first_query = turns[0].get("user_input", "") if turns else ""
                clean_first = first_query.strip().replace("\n", " ")

                title = data.get("title")
                if not title:
                    if clean_first:
                        title = (clean_first[:42] + "...") if len(clean_first) > 42 else clean_first
                    elif data.get("conversation_summary"):
                        sum_text = data["conversation_summary"].strip().replace("\n", " ")
                        title = (sum_text[:42] + "...") if len(sum_text) > 42 else sum_text
                    else:
                        title = "New Thread"

                summaries.append({
                    "session_id": data.get("session_id", fpath.stem),
                    "title": title,
                    "first_query": clean_first,
                    "user_id": data.get("user_id", "unknown"),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "turns_count": len(turns),
                    "pinned_doc_id": data.get("pinned_doc_id"),
                    "pinned_doc_ids": data.get("pinned_doc_ids", ([data["pinned_doc_id"]] if data.get("pinned_doc_id") else [])),
                    "active_documents": data.get("active_documents", []),
                    "conversation_summary": data.get("conversation_summary", ""),
                    "ephemeral_documents_count": len(data.get("ephemeral_documents", {})),
                    "file_path": rel_path
                })
            except Exception as e:
                logger.warning(f"Error reading session file {fpath}: {e}")

        # Sort newest first
        summaries.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return summaries

    def delete_session(self, session_id: str) -> bool:
        """Deletes a persisted session file."""
        path = self._get_session_path(session_id)
        if path.is_file():
            try:
                path.unlink()
                logger.info(f"Deleted session file: {path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete session file {path}: {e}")
                return False
        return False

    def record_turn(
        self,
        session_id: str,
        user_input: str,
        result_state: Dict[str, Any],
        pinned_doc_id: Optional[Union[str, List[str]]] = None,
        ephemeral_documents: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extracts execution results from a completed graph turn, appends the turn
        record, updates conversation summary & active documents, and persists to disk.
        """
        session_data = self.load_session(session_id)
        if not session_data:
            session_data = self.create_session(session_id=session_id)

        # Extract & serialize Intent
        intent_obj = result_state.get("intent")
        intent_dict: Optional[Dict[str, Any]] = None
        if intent_obj:
            if hasattr(intent_obj, "model_dump"):
                intent_dict = intent_obj.model_dump()
            elif isinstance(intent_obj, dict):
                intent_dict = intent_obj
            else:
                intent_dict = {"raw": str(intent_obj)}

        # Extract & serialize Response
        resp_obj = result_state.get("current_response")
        resp_dict: Optional[Dict[str, Any]] = None
        if resp_obj:
            if hasattr(resp_obj, "model_dump"):
                resp_dict = resp_obj.model_dump()
                resp_dict["response_type"] = resp_obj.__class__.__name__
            elif isinstance(resp_obj, dict):
                resp_dict = resp_obj
            else:
                resp_dict = {"response_type": "text", "content": str(resp_obj)}

        # Compile turn data
        turn_target_ids = result_state.get("target_doc_ids") or result_state.get("target_doc_id") or pinned_doc_id
        if isinstance(turn_target_ids, list):
            turn_target_display = ", ".join(turn_target_ids)
        else:
            turn_target_display = turn_target_ids

        turn_index = len(session_data.get("turns", [])) + 1
        turn_record = {
            "turn_index": turn_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_input": user_input,
            "intent": intent_dict,
            "actions_taken": result_state.get("actions_taken", []),
            "response": resp_dict,
            "target_doc_id": turn_target_display
        }

        session_data.setdefault("turns", []).append(turn_record)

        # Set human-readable title from first query if not yet set
        if not session_data.get("title") or session_data.get("title") == "New Thread":
            clean_q = user_input.strip().replace("\n", " ")
            session_data["title"] = (clean_q[:42] + "...") if len(clean_q) > 42 else clean_q

        session_data["conversation_summary"] = result_state.get(
            "conversation_summary", session_data.get("conversation_summary", "")
        )
        session_data["active_documents"] = result_state.get(
            "active_documents", session_data.get("active_documents", [])
        )
        if ephemeral_documents is not None:
            session_data["ephemeral_documents"] = ephemeral_documents

        if isinstance(pinned_doc_id, list):
            session_data["pinned_doc_ids"] = pinned_doc_id
            session_data["pinned_doc_id"] = pinned_doc_id[0] if pinned_doc_id else None
        else:
            session_data["pinned_doc_id"] = pinned_doc_id
            session_data["pinned_doc_ids"] = [pinned_doc_id] if pinned_doc_id else []

        self.save_session(session_id, session_data)
        return session_data
