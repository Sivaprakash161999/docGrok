"""
Streamlit Web Interface for the Intelligent Document Assistant.
Provides a modern chat experience with session persistence, dynamic document corpus
browsing, external file ingestion, document pinning, and specialized visual cards
for Q&A, summaries, and mathematical calculations.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import streamlit as st

from src.assistant import DocumentAssistant
from src.retrieval import DocumentChunk
from src.schemas import (
    AnswerResponse,
    CalculationResponse,
    SummaryResponse,
    UserIntent,
)

# Page configuration
st.set_page_config(
    page_title="DocGrok - Document Assistant",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Assistant Initialization & Session Management
# ============================================================================

def get_assistant() -> DocumentAssistant:
    """Retrieves or initializes the DocumentAssistant in Streamlit session state."""
    if "assistant" not in st.session_state:
        try:
            st.session_state.assistant = DocumentAssistant()
        except Exception as e:
            st.error(f"Failed to initialize DocGrok: {e}")
            st.info("Please ensure GEMINI_API_KEY is configured in your .env file.")
            st.stop()
    return st.session_state.assistant


assistant = get_assistant()

# Initialize message history list in session_state if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def sync_chat_history_from_session():
    """Syncs in-memory Streamlit chat_history from the persisted session JSON."""
    session_data = assistant.get_session_history()
    if session_data and "turns" in session_data:
        st.session_state.chat_history = session_data["turns"]
    else:
        st.session_state.chat_history = []


# Initial sync if empty
if not st.session_state.chat_history:
    sync_chat_history_from_session()


# ============================================================================
# Sidebar: Session & Document Corpus Controls
# ============================================================================

with st.sidebar:
    st.title("📑 DocGrok")
    st.caption("Agentic AI Document Intelligence with LangGraph & Google Gemini")
    st.divider()

    # --- Section 1: Session Management ---
    st.subheader("💬 Sessions")

    if "session_status_msg" in st.session_state:
        s_type, s_text = st.session_state.pop("session_status_msg")
        if s_type == "success":
            st.success(s_text, icon="✅")
        else:
            st.error(s_text, icon="⚠️")

    saved_sessions = assistant.list_saved_sessions()
    session_ids = [s["session_id"] for s in saved_sessions]

    if assistant.session_id not in session_ids:
        session_ids.insert(0, assistant.session_id)

    curr_index = session_ids.index(assistant.session_id) if assistant.session_id in session_ids else 0

    def format_session_label(sid: str) -> str:
        s_info = next((s for s in saved_sessions if s["session_id"] == sid), None)
        if not s_info:
            return f"✨ Active Session [{sid}]"

        turns = s_info.get("turns_count", 0)
        if turns == 0:
            return f"✨ New Session [{sid}]"

        raw_title = s_info.get("title") or s_info.get("first_query") or ""
        if not raw_title and s_info.get("conversation_summary"):
            raw_title = s_info["conversation_summary"].strip().replace("\n", " ")
        if not raw_title:
            raw_title = "Conversation"

        short_title = (raw_title[:38] + "...") if len(raw_title) > 38 else raw_title
        pinned = s_info.get("pinned_doc_ids") or []
        doc_hint = f" • 🏷️ {pinned[0]}" if pinned else ""
        turns_label = f"{turns} turn" if turns == 1 else f"{turns} turns"

        return f"💬 {short_title} ({turns_label}{doc_hint}) [{sid}]"

    selected_session_id = st.selectbox(
        "Active Session",
        options=session_ids,
        index=curr_index,
        format_func=format_session_label,
        help="Select a previous conversation to resume from sessions/."
    )

    if selected_session_id != assistant.session_id:
        if assistant.resume_session(selected_session_id):
            sync_chat_history_from_session()
            st.rerun()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("➕ New Session", use_container_width=True):
            new_id = assistant.reset_session()
            sync_chat_history_from_session()
            st.toast(f"Created new session: {new_id}", icon="✨")
            st.rerun()

    with col_s2:
        if st.button("🗑️ Delete", use_container_width=True):
            target_sid = assistant.session_id
            if assistant.session_manager.delete_session(target_sid):
                st.session_state["session_status_msg"] = ("success", f"Deleted session '{target_sid}'.")
                st.toast(f"Deleted session {target_sid}", icon="🗑️")
                if hasattr(assistant, "_session_ephemeral_docs"):
                    assistant._session_ephemeral_docs.pop(target_sid, None)
                assistant.reset_session()
                sync_chat_history_from_session()
                st.rerun()
            else:
                st.session_state["session_status_msg"] = ("error", f"Could not delete session '{target_sid}'.")
                st.rerun()

    # Show rolling summary if present
    if assistant.conversation_summary:
        with st.expander("🧠 Rolling Memory Summary"):
            st.caption(assistant.conversation_summary)

    st.divider()

    # --- Section 2: Document Scope & Corpus ---
    st.subheader("📁 Document Scope")

    all_docs = assistant.repo.get_all_documents()
    doc_map = {f"{d.doc_id} - {d.title}": d.doc_id for d in all_docs}
    id_to_label = {v: k for k, v in doc_map.items()}

    default_selected = [id_to_label[did] for did in assistant.pinned_doc_ids if did in id_to_label]
    selected_scope_labels = st.multiselect(
        "Target Document Focus (Multi-Select)",
        options=list(doc_map.keys()),
        default=default_selected,
        placeholder="Select documents, or leave empty for Dynamic Search...",
        help="Select one or more documents to strictly restrict scope. Leave empty to dynamically search the entire corpus."
    )
    new_selected_ids = [doc_map[label] for label in selected_scope_labels]
    if set(new_selected_ids) != set(assistant.pinned_doc_ids):
        assistant.pin_documents(new_selected_ids)
        st.rerun()

    if assistant.pinned_doc_ids:
        st.info(f"🎯 Scope Locked ({len(assistant.pinned_doc_ids)}): **{', '.join(assistant.pinned_doc_ids)}**")
        if st.button("🔓 Clear Pinned Scope", use_container_width=True):
            assistant.unpin_document()
            st.toast("Cleared pinned scope", icon="🔓")
            st.rerun()

    # Ingest External Document Expander (Multi-File Ingestion)
    with st.expander("📤 Ingest External File(s)"):
        if "upload_status_msg" in st.session_state:
            u_type, u_text = st.session_state.pop("upload_status_msg")
            if u_type == "success":
                st.success(u_text, icon="✅")
            else:
                st.error(u_text, icon="⚠️")

        uploader_nonce = st.session_state.get(f"uploader_nonce_{assistant.session_id}", 0)
        uploader_key = f"file_uploader_{assistant.session_id}_{uploader_nonce}"

        uploaded_files = st.file_uploader(
            "Upload Document(s)",
            type=["json", "txt", "md", "csv", "pdf", "xlsx", "xls", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key=uploader_key,
            help="Upload one or multiple external files to analyze (PDF, Excel, CSV, Infographics, JSON, Markdown, Text)."
        )
        persist_upload = st.checkbox("Persist permanently to document_corpus/", value=False, key=f"persist_upload_{assistant.session_id}")
        custom_prefix = st.text_input("Custom ID Prefix (Optional)", placeholder="e.g. EXT-FIN", key=f"prefix_{assistant.session_id}")

        if uploaded_files:
            col_up_action, col_up_clear = st.columns([3, 1])
            with col_up_action:
                upload_clicked = st.button(
                    f"Upload & Pin Scope ({len(uploaded_files)} file{'s' if len(uploaded_files) > 1 else ''})",
                    use_container_width=True,
                    type="primary"
                )
            with col_up_clear:
                clear_clicked = st.button("❌ Clear", use_container_width=True, help="Clear files from uploader")

            if clear_clicked:
                st.session_state[f"uploader_nonce_{assistant.session_id}"] = uploader_nonce + 1
                st.rerun()

            if upload_clicked:
                newly_loaded: List[str] = []
                for uf in uploaded_files:
                    suffix = Path(uf.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.getvalue())
                        tmp_path = tmp.name

                    try:
                        clean_custom = f"{custom_prefix.strip()}-{Path(uf.name).stem}".upper() if custom_prefix.strip() else f"EXT-{Path(uf.name).stem}".upper()
                        chunk = assistant.load_external_document(
                            file_path=tmp_path,
                            persist=persist_upload,
                            doc_id=clean_custom
                        )
                        newly_loaded.append(chunk.doc_id)
                    except Exception as ex:
                        st.error(f"Error loading '{uf.name}': {ex}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)

                if newly_loaded:
                    for n_id in newly_loaded:
                        if n_id not in assistant.pinned_doc_ids:
                            assistant.pinned_doc_ids.append(n_id)
                    # Advance nonce to reset uploader widget upon successful ingestion
                    st.session_state[f"uploader_nonce_{assistant.session_id}"] = uploader_nonce + 1
                    succ_msg = f"Loaded and pinned {len(newly_loaded)} file(s): {', '.join(newly_loaded)}"
                    st.session_state["upload_status_msg"] = ("success", succ_msg)
                    st.toast(succ_msg, icon="📎")
                    st.rerun()

    # Document Removal Expander
    with st.expander("🗑️ Manage / Remove Document"):
        if "doc_manage_msg" in st.session_state:
            msg_type, msg_text = st.session_state.pop("doc_manage_msg")
            if msg_type == "success":
                st.success(msg_text, icon="✅")
            else:
                st.error(msg_text, icon="⚠️")

        remove_doc_id = st.selectbox(
            "Document to Remove",
            options=[d.doc_id for d in all_docs],
            index=0 if all_docs else None
        )
        perm_delete = st.checkbox("Permanently delete from disk", value=False)
        force_delete = st.checkbox("Force delete (bypass seed protection)", value=False)

        if remove_doc_id and st.button("Remove Document", type="primary", use_container_width=True):
            success, msg = assistant.remove_document(
                doc_id=remove_doc_id,
                permanent=perm_delete,
                force=force_delete
            )
            if success:
                st.session_state["doc_manage_msg"] = ("success", msg)
                st.toast(msg, icon="🗑️")
            else:
                st.session_state["doc_manage_msg"] = ("error", msg)
                st.toast(msg, icon="⚠️")
            st.rerun()

    # View Indexed Corpus
    persisted_corpus = assistant.repo.get_corpus_documents()
    with st.expander(f"📚 Corpus Documents ({len(persisted_corpus)})"):
        for d in persisted_corpus:
            amt = d.metadata.get("amount")
            amt_str = f" (${amt:,.2f})" if amt else ""
            st.markdown(f"**{d.doc_id}**: {d.title}{amt_str}")

    st.divider()

    # System Statistics & Logs
    with st.expander("📊 Statistics & System Status"):
        stats = assistant.repo.get_statistics()
        st.write(f"**Corpus Directory:** `{stats.get('corpus_directory')}`")
        st.write(f"**Total Documents:** {stats.get('total_documents')}")
        st.write(f"**Total Dollar Volume:** ${stats.get('total_dollar_volume', 0):,.2f}")
        st.write(f"**Logs Location:** `logs/assistant.log`")
        st.write(f"**Sessions Location:** `sessions/`")


# ============================================================================
# Main Chat Area: Rendering Turns
# ============================================================================

st.header("DocGrok Assistant")
scope_header = f"Scope: `{assistant.pinned_doc_id}`" if assistant.pinned_doc_id else "Scope: `Corpus Search`"
st.caption(f"Session: `{assistant.session_id}` | {scope_header}")


def render_response(resp_data: Dict[str, Any], actions: List[str]):
    """Renders the specialist agent response with visual lineage and metrics."""
    if not resp_data:
        return

    resp_type = resp_data.get("response_type", "")

    # Lineage Pills
    if actions:
        lineage_badges = " ➔ ".join([f"`{a}`" for a in actions])
        st.markdown(f"<small>Lineage: {lineage_badges}</small>", unsafe_allow_html=True)

    # Q&A Response
    if resp_type == "AnswerResponse":
        found = resp_data.get("found", True)
        answer = resp_data.get("answer", "")
        if not found:
            st.warning(f"⚠️ **Grounding Notice:** {answer}")
        else:
            st.markdown(answer)

        sources = resp_data.get("sources", [])
        if sources:
            st.caption(f"Sources cited: {', '.join(sources)}")

    # Summarization Response
    elif resp_type == "SummaryResponse":
        found = resp_data.get("found", True)
        title = resp_data.get("title", "Summary")
        summary = resp_data.get("summary", "")
        key_points = resp_data.get("key_points", [])

        if not found:
            st.warning(f"⚠️ **Grounding Notice:** {summary}")
        else:
            st.subheader(title)
            st.markdown(summary)
            if key_points:
                st.markdown("**Key Highlights:**")
                for pt in key_points:
                    st.markdown(f"- {pt}")

        sources = resp_data.get("sources", [])
        if sources:
            st.caption(f"Sources cited: {', '.join(sources)}")

    # Calculation Response
    elif resp_type == "CalculationResponse":
        found = resp_data.get("found", True)
        formatted_res = resp_data.get("formatted_result", "")
        expr = resp_data.get("expression", "")
        res_val = resp_data.get("result", 0.0)
        explanation = resp_data.get("explanation", "")

        if not found:
            st.warning(f"⚠️ **Grounding Notice:** {explanation}")
        else:
            st.metric(label="Calculated Result", value=formatted_res)
            # Only render formula block if a non-trivial arithmetic expression was formulated
            if expr and expr.strip().upper() not in ("N/A", "NONE", "NULL", "") and expr.strip() != str(res_val):
                st.code(f"{expr} = {res_val}", language="text")
            st.markdown(f"**Explanation:**\n{explanation}")

        sources = resp_data.get("sources", [])
        if sources:
            st.caption(f"Sources cited: {', '.join(sources)}")

    else:
        # Fallback for generic text
        content = resp_data.get("content") or resp_data.get("answer") or str(resp_data)
        st.markdown(content)


# Display existing conversation history
for turn in st.session_state.chat_history:
    user_msg = turn.get("user_input", "")
    resp = turn.get("response", {})
    actions = turn.get("actions_taken", [])
    intent = turn.get("intent", {})

    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        if intent:
            itype = intent.get("intent_type", "").upper()
            conf = intent.get("confidence", 0.0)
            st.caption(f"Intent: **{itype}** ({conf:.0%}) | Reasoning: {intent.get('reasoning')}")
        render_response(resp, actions)


# ============================================================================
# Chat Input & Turn Execution
# ============================================================================

# Prompt shortcuts / suggestions
if not st.session_state.chat_history:
    st.markdown("##### Suggested Queries:")
    col_q1, col_q2, col_q3 = st.columns(3)
    suggestion_clicked = None
    with col_q1:
        if st.button("💰 What was total Q1 2024 revenue?", use_container_width=True):
            suggestion_clicked = "What was the total revenue in Q1 2024?"
    with col_q2:
        if st.button("📊 Calculate Q1 Net Profit", use_container_width=True):
            suggestion_clicked = "Calculate the net profit for Q1 2024 by subtracting expenses."
    with col_q3:
        if st.button("🏥 Summarize claim MED-CLM-7701", use_container_width=True):
            suggestion_clicked = "Summarize medical claim MED-CLM-7701."

    if suggestion_clicked:
        st.session_state.pending_query = suggestion_clicked

# --- Document Selector & Attachment Settings (Directly Above Chat Box) ---
st.markdown("---")
col_chat_doc, col_chat_persist = st.columns([3, 1])

with col_chat_doc:
    corpus_docs = assistant.repo.get_corpus_documents()
    chat_doc_map = {f"{d.doc_id}: {d.title}": d.doc_id for d in corpus_docs}
    chat_id_to_label = {v: k for k, v in chat_doc_map.items()}

    default_chat_selected = [chat_id_to_label[did] for did in assistant.pinned_doc_ids if did in chat_id_to_label]
    selected_chat_labels = st.multiselect(
        "Scope to Corpus Document(s)",
        options=list(chat_doc_map.keys()),
        default=default_chat_selected,
        placeholder="Select corpus document(s) for query scope (or leave empty for full search)...",
        help="Select one or more corpus documents to constrain the query. Can be combined with attached files for a hybrid scope.",
        label_visibility="collapsed"
    )
    selected_corpus_ids = [chat_doc_map[lbl] for lbl in selected_chat_labels]

with col_chat_persist:
    persist_attached = st.checkbox(
        "💾 Persist to corpus",
        value=False,
        help="If external file(s) are attached via the chat box, save them permanently into document_corpus/."
    )

# Native chat input with multiple file attachments (+) enabled
chat_submission = st.chat_input(
    placeholder="Ask a question, compare documents, or attach files (+)...",
    accept_file="multiple",
    file_type=["json", "txt", "md", "csv", "pdf", "xlsx", "xls", "png", "jpg", "jpeg", "webp"]
)

# Handle input from either typed chat_input or a suggestion button click
raw_text = ""
attached_files = []

if chat_submission is not None:
    if isinstance(chat_submission, str):
        raw_text = chat_submission
    else:
        raw_text = getattr(chat_submission, "text", "") or ""
        attached_files = getattr(chat_submission, "files", []) or []

elif st.session_state.get("pending_query"):
    raw_text = st.session_state.pop("pending_query")

if raw_text or attached_files:
    attached_doc_ids: List[str] = []
    badge_parts: List[str] = []

    # Scenario 1: User attached external file(s) directly in the chat box
    if attached_files:
        for upload in attached_files:
            suffix = Path(upload.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(upload.getvalue())
                tmp_path = tmp.name

            try:
                clean_attached_id = f"EXT-{Path(upload.name).stem}".upper()
                chunk = assistant.load_external_document(
                    file_path=tmp_path,
                    persist=persist_attached,
                    doc_id=clean_attached_id
                )
                attached_doc_ids.append(chunk.doc_id)
                # Pin newly loaded document for ongoing session context
                if chunk.doc_id not in assistant.pinned_doc_ids:
                    assistant.pinned_doc_ids.append(chunk.doc_id)
            except Exception as ex:
                st.error(f"Failed to load attached document '{upload.name}': {ex}")
                st.stop()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        persist_note = " *(persisted)*" if persist_attached else ""
        badge_parts.append(f"📎 Attached ({len(attached_doc_ids)}): `{', '.join(attached_doc_ids)}`{persist_note}")

    # Scenario 2: Corpus documents selected in-chat dropdown or pinned in sidebar
    target_corpus_ids = selected_corpus_ids if selected_corpus_ids else [
        pid for pid in assistant.pinned_doc_ids if pid not in attached_doc_ids
    ]
    # Discard duplicates already in attached_doc_ids
    target_corpus_ids = [cid for cid in target_corpus_ids if cid not in attached_doc_ids]

    if target_corpus_ids:
        badge_parts.append(f"📁 Corpus ({len(target_corpus_ids)}): `{', '.join(target_corpus_ids)}`")

    # Combined Hybrid Scope: attached external documents + selected corpus documents
    effective_target_ids = list(dict.fromkeys(attached_doc_ids + target_corpus_ids))

    if attached_doc_ids and target_corpus_ids:
        attached_badge = f"🎯 **Hybrid Scope:** {' + '.join(badge_parts)}"
    elif attached_doc_ids:
        attached_badge = f"🎯 **Attached Scope:** {badge_parts[0]}"
    elif target_corpus_ids:
        attached_badge = f"🎯 **Corpus Scope:** {badge_parts[0]}"
    else:
        attached_badge = None

    # If user attached files without typing text, provide a sensible default prompt
    if not raw_text.strip() and effective_target_ids:
        raw_text = f"Analyze, compare, and summarize the provided document(s): {', '.join(effective_target_ids)}."

    # 1. Display user query immediately
    with st.chat_message("user"):
        if attached_badge:
            st.caption(attached_badge)
        st.markdown(raw_text)

    # 2. Process message through LangGraph workflow with multi/hybrid scope
    with st.chat_message("assistant"):
        with st.spinner("Processing through specialist agent graph..."):
            result_state = assistant.process_message(
                user_input=raw_text,
                target_doc_ids=effective_target_ids if effective_target_ids else None
            )

        intent_obj = result_state.get("intent")
        current_resp = result_state.get("current_response")
        actions_taken = result_state.get("actions_taken", [])

        # Display intent info
        if intent_obj:
            st.caption(
                f"Intent: **{intent_obj.intent_type.upper()}** ({intent_obj.confidence:.0%}) | "
                f"Reasoning: {intent_obj.reasoning}"
            )

        # Convert current_resp to dict format for rendering
        if current_resp:
            if hasattr(current_resp, "model_dump"):
                resp_payload = current_resp.model_dump()
                resp_payload["response_type"] = current_resp.__class__.__name__
            elif isinstance(current_resp, dict):
                resp_payload = current_resp
            else:
                resp_payload = {"content": str(current_resp), "response_type": "text"}
            render_response(resp_payload, actions_taken)

    # 3. Synchronize updated turns history
    sync_chat_history_from_session()
    st.rerun()
