"""
LangGraph workflow implementation for the Document Assistant.
Defines graph state, specialist agent nodes (Intent Classifier, Q&A,
Summarization, Calculation, Memory Update), conditional routing edges,
and compiled execution graph with InMemorySaver checkpointing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from src.prompts import (
    get_chat_prompt_template,
    get_intent_classification_prompt,
    get_memory_update_prompt,
)
from src.retrieval import DocumentRepository, default_repository
from src.schemas import (
    AgentState,
    AnswerResponse,
    CalculationResponse,
    MemoryUpdate,
    SummaryResponse,
    UserIntent,
)
from src.tools import evaluate_math_expression

logger = logging.getLogger("doc_assistant.agent")


# ============================================================================
# Context Preparation Helper
# ============================================================================

def _build_document_context(state: AgentState, repo: DocumentRepository) -> str:
    """
    Assembles document context strictly respecting user document selection:
    1. Supports multi-document and hybrid scopes (combining multiple uploaded files
       and/or multiple selected corpus documents).
    2. If explicit target document(s) are specified, scope is restricted EXCLUSIVELY
       to those documents.
    3. If no documents are passed or selected, dynamically searches the available corpus.
       If nothing relevant is found, returns an explicit notice (NO arbitrary fallback).
    """
    persist = state.get("persist_target_document", False)

    collected_docs: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()
    missing_ids: List[str] = []

    # 1. Collect external passed documents (target_documents or target_document)
    raw_target_docs = state.get("target_documents") or []
    if state.get("target_document"):
        single_td = state.get("target_document")
        if isinstance(single_td, list):
            raw_target_docs.extend(single_td)
        elif isinstance(single_td, dict):
            raw_target_docs.append(single_td)

    for doc_dict in raw_target_docs:
        if not isinstance(doc_dict, dict):
            continue
        doc_id = doc_dict.get("doc_id", "PASSED_DOC").upper()
        title = doc_dict.get("title", "External User Document")
        doc_type = doc_dict.get("doc_type", "external_document")
        content = doc_dict.get("content", "")
        meta = doc_dict.get("metadata", {})

        if persist:
            repo.add_document(
                doc_id=doc_id,
                title=title,
                doc_type=doc_type,
                content=content,
                metadata=meta,
                persist=True
            )

        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            collected_docs.append({
                "doc_id": doc_id,
                "title": title,
                "doc_type": doc_type,
                "content": content,
                "metadata": meta,
                "source_type": "uploaded"
            })
            logger.info(f"Target scope includes external document '{doc_id}' (persist={persist}).")

    # 2. Collect selected corpus document IDs (target_doc_ids or target_doc_id)
    raw_target_ids: List[str] = []
    if state.get("target_doc_ids"):
        raw_target_ids.extend(state["target_doc_ids"])
    if state.get("target_doc_id"):
        single_id = state.get("target_doc_id")
        if isinstance(single_id, list):
            raw_target_ids.extend(single_id)
        elif isinstance(single_id, str):
            raw_target_ids.append(single_id)

    for rid in raw_target_ids:
        clean_id = rid.strip().upper()
        if not clean_id or clean_id in seen_ids:
            continue
        doc = repo.get_document_by_id(clean_id)
        if doc:
            seen_ids.add(doc.doc_id)
            collected_docs.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "doc_type": doc.doc_type,
                "content": doc.content,
                "metadata": doc.metadata,
                "source_type": "corpus"
            })
            logger.info(f"Target scope includes corpus document '{doc.doc_id}'.")
        else:
            logger.warning(f"Target document ID '{clean_id}' not found in repository.")
            missing_ids.append(clean_id)

    # SCENARIO 1 & 2: Explicit Target Documents Provided (Single, Multi, or Hybrid)
    if collected_docs:
        if len(collected_docs) == 1:
            d = collected_docs[0]
            meta_str = "\n  ".join([f"{k}: {v}" for k, v in d["metadata"].items()]) if d["metadata"] else "None"
            return (
                f"=== TARGET DOCUMENT (RESTRICTED SCOPE: Base your answer ONLY on this document) ===\n"
                f"Document ID: {d['doc_id']}\n"
                f"Title: {d['title']} (Type: {d['doc_type']})\n"
                f"Metadata:\n  {meta_str}\n\n"
                f"Content:\n{d['content']}\n"
            )
        else:
            doc_id_list = ", ".join([d["doc_id"] for d in collected_docs])
            header_lines = [
                f"=== RESTRICTED TARGET DOCUMENTS SCOPE (HYBRID / MULTI-DOCUMENT SCOPE) ===",
                f"Base your answer EXCLUSIVELY on the {len(collected_docs)} documents listed below.",
                f"Active Scoped Documents ({len(collected_docs)}): {doc_id_list}\n"
            ]
            for i, d in enumerate(collected_docs, start=1):
                meta_str = "\n  ".join([f"{k}: {v}" for k, v in d["metadata"].items()]) if d["metadata"] else "None"
                header_lines.append(
                    f"--- Document {i} of {len(collected_docs)}: [{d['doc_id']}] {d['title']} (Type: {d['doc_type']}) ---\n"
                    f"Metadata:\n  {meta_str}\n\n"
                    f"Content:\n{d['content']}\n"
                )
            return "\n".join(header_lines)

    if missing_ids and not collected_docs:
        return f"NOTICE: The requested document ID(s) '{', '.join(missing_ids)}' were not found in the repository."

    # SCENARIO 3: User did not specify any target document -> dynamic search on available corpus
    user_input = state.get("user_input", "")
    active_doc_ids = state.get("active_documents", []) or []

    found_chunks = []
    seen_search_ids = set()

    # If active documents were referenced in recent turns of this session
    for doc_id in active_doc_ids:
        doc = repo.get_document_by_id(doc_id)
        if doc and doc.doc_id not in seen_search_ids:
            seen_search_ids.add(doc.doc_id)
            found_chunks.append(doc)

    # Keyword search across corpus
    search_matches = repo.search_by_keyword(user_input, limit=3)
    for chunk in search_matches:
        if chunk.doc_id not in seen_search_ids:
            seen_search_ids.add(chunk.doc_id)
            found_chunks.append(chunk)

    if not found_chunks:
        logger.info("No relevant documents found in corpus for query.")
        return "NOTICE: No relevant documents found in the available corpus matching the inquiry."

    context_lines = ["=== RELEVANT CORPUS DOCUMENTS (Retrieved from available knowledge base) ==="]
    for chunk in found_chunks:
        context_lines.append(
            f"--- Document ID: {chunk.doc_id} | {chunk.title} ({chunk.doc_type}) ---\n"
            f"{chunk.content}\n"
        )

    return "\n".join(context_lines)


# ============================================================================
# 1. Intent Classification Node
# ============================================================================

def classify_intent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    First node in the graph: classifies user intent to route to the appropriate specialist agent.
    """
    logger.info("Executing classify_intent node.")
    configurable = config.get("configurable", {})
    llm = configurable.get("llm")
    repo: DocumentRepository = configurable.get("repo", default_repository)
    if not llm:
        raise ValueError("LLM instance must be provided in config['configurable']['llm']")

    user_input = state.get("user_input", "")
    conv_history = state.get("conversation_summary", "None (Session Start)")
    active_docs = ", ".join(state.get("active_documents", [])) or "None"
    available_docs_summary = repo.get_available_documents_summary()

    # Configure structured output with UserIntent schema
    structured_classifier = llm.with_structured_output(UserIntent)
    prompt_template = get_intent_classification_prompt()
    formatted_prompt = prompt_template.format_messages(
        available_documents=available_docs_summary,
        conversation_history=conv_history,
        active_documents=active_docs,
        user_input=user_input
    )

    try:
        intent_result: UserIntent = structured_classifier.invoke(formatted_prompt)
    except Exception as e:
        logger.warning(f"Structured intent classification failed: {e}. Defaulting to QA.")
        intent_result = UserIntent(
            intent_type="qa",
            confidence=0.5,
            reasoning=f"Fallback classification due to error: {str(e)}"
        )

    # Route next_step based on classified intent
    if intent_result.intent_type == "qa":
        next_step = "qa_agent"
    elif intent_result.intent_type == "summarization":
        next_step = "summarization_agent"
    elif intent_result.intent_type == "calculation":
        next_step = "calculation_agent"
    else:
        next_step = "qa_agent"

    logger.info(f"Classified intent: {intent_result.intent_type} -> next_step: {next_step}")

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent_result,
        "next_step": next_step
    }


# ============================================================================
# 2. Specialist Agent Nodes
# ============================================================================

def qa_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Q&A Specialist Agent: provides factual answers with document citations.
    """
    logger.info("Executing qa_agent node.")
    configurable = config.get("configurable", {})
    llm = configurable.get("llm")
    repo: DocumentRepository = configurable.get("repo", default_repository)

    user_input = state.get("user_input", "")
    conv_summary = state.get("conversation_summary", "None")
    active_docs = ", ".join(state.get("active_documents", [])) or "None"
    doc_context = _build_document_context(state, repo)

    structured_llm = llm.with_structured_output(AnswerResponse)
    prompt_template = get_chat_prompt_template("qa")
    formatted_messages = prompt_template.format_messages(
        conversation_summary=conv_summary,
        active_documents=active_docs,
        document_context=doc_context,
        user_input=user_input
    )

    answer_response: AnswerResponse = structured_llm.invoke(formatted_messages)

    current_active = list(set(state.get("active_documents", []) + answer_response.sources))

    return {
        "actions_taken": ["qa_agent"],
        "current_response": answer_response,
        "active_documents": current_active,
        "tools_used": ["search_documents", "read_document"]
    }


def summarization_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Summarization Specialist Agent: synthesizes executive summaries and key points.
    """
    logger.info("Executing summarization_agent node.")
    configurable = config.get("configurable", {})
    llm = configurable.get("llm")
    repo: DocumentRepository = configurable.get("repo", default_repository)

    user_input = state.get("user_input", "")
    conv_summary = state.get("conversation_summary", "None")
    active_docs = ", ".join(state.get("active_documents", [])) or "None"
    doc_context = _build_document_context(state, repo)

    structured_llm = llm.with_structured_output(SummaryResponse)
    prompt_template = get_chat_prompt_template("summarization")
    formatted_messages = prompt_template.format_messages(
        conversation_summary=conv_summary,
        active_documents=active_docs,
        document_context=doc_context,
        user_input=user_input
    )

    summary_response: SummaryResponse = structured_llm.invoke(formatted_messages)

    current_active = list(set(state.get("active_documents", []) + summary_response.sources))

    return {
        "actions_taken": ["summarization_agent"],
        "current_response": summary_response,
        "active_documents": current_active,
        "tools_used": ["read_document"]
    }


def calculation_agent(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Calculation Specialist Agent: extracts figures, computes deterministically
    via the safe calculator tool, and produces a structured CalculationResponse.
    """
    logger.info("Executing calculation_agent node.")
    configurable = config.get("configurable", {})
    llm = configurable.get("llm")
    repo: DocumentRepository = configurable.get("repo", default_repository)

    user_input = state.get("user_input", "")
    conv_summary = state.get("conversation_summary", "None")
    active_docs = ", ".join(state.get("active_documents", [])) or "None"
    doc_context = _build_document_context(state, repo)

    structured_llm = llm.with_structured_output(CalculationResponse)
    prompt_template = get_chat_prompt_template("calculation")
    formatted_messages = prompt_template.format_messages(
        conversation_summary=conv_summary,
        active_documents=active_docs,
        document_context=doc_context,
        user_input=user_input
    )

    calc_response: CalculationResponse = structured_llm.invoke(formatted_messages)

    # Deterministic safeguard: execute calculator on extracted expression to verify exact precision
    if calc_response.found and calc_response.expression and calc_response.expression.upper() not in ("N/A", "NONE", ""):
        try:
            evaluated_val = evaluate_math_expression(calc_response.expression)
            calc_response.result = evaluated_val
        except Exception as e:
            logger.warning(f"Calculation tool re-evaluation notice: {e}")

    current_active = list(set(state.get("active_documents", []) + calc_response.sources))

    return {
        "actions_taken": ["calculation_agent"],
        "current_response": calc_response,
        "active_documents": current_active,
        "tools_used": ["calculator", "read_document"]
    }


# ============================================================================
# 3. Conversational Memory Update Node
# ============================================================================

def update_memory(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """
    Memory Update Node: synthesizes ongoing multi-turn context and updates active documents.
    """
    logger.info("Executing update_memory node.")
    configurable = config.get("configurable", {})
    llm = configurable.get("llm")

    prev_summary = state.get("conversation_summary", "None")
    prev_docs = ", ".join(state.get("active_documents", [])) or "None"
    user_input = state.get("user_input", "")
    curr_resp = state.get("current_response")

    # Format string representation of current response
    if hasattr(curr_resp, "answer"):
        resp_text = curr_resp.answer
    elif hasattr(curr_resp, "summary"):
        resp_text = curr_resp.summary
    elif hasattr(curr_resp, "explanation"):
        resp_text = f"{curr_resp.explanation} (Result: {curr_resp.formatted_result})"
    else:
        resp_text = str(curr_resp)

    structured_llm = llm.with_structured_output(MemoryUpdate)
    prompt_template = get_memory_update_prompt()
    formatted_messages = prompt_template.format_messages(
        previous_summary=prev_summary,
        previous_documents=prev_docs,
        user_input=user_input,
        agent_response=resp_text
    )

    # Collect any explicit target doc IDs from current turn to maintain tracking
    turn_target_ids: List[str] = []
    if state.get("target_doc_ids"):
        turn_target_ids.extend(state["target_doc_ids"])
    if state.get("target_doc_id"):
        t_id = state["target_doc_id"]
        if isinstance(t_id, list):
            turn_target_ids.extend(t_id)
        elif isinstance(t_id, str):
            turn_target_ids.append(t_id)
    if state.get("target_documents"):
        turn_target_ids.extend([d.get("doc_id", "") for d in state["target_documents"] if d.get("doc_id")])
    if state.get("target_document"):
        td = state["target_document"]
        if isinstance(td, dict) and td.get("doc_id"):
            turn_target_ids.append(td["doc_id"])
        elif isinstance(td, list):
            turn_target_ids.extend([d.get("doc_id", "") for d in td if isinstance(d, dict) and d.get("doc_id")])

    try:
        mem_update: MemoryUpdate = structured_llm.invoke(formatted_messages)
        updated_summary = mem_update.conversation_summary
        merged_docs = list(dict.fromkeys(state.get("active_documents", []) + mem_update.active_documents + turn_target_ids))
    except Exception as e:
        logger.warning(f"Memory update error: {e}. Retaining prior summary.")
        updated_summary = f"{prev_summary} | User: {user_input} | Agent: {resp_text[:100]}"
        merged_docs = list(dict.fromkeys(state.get("active_documents", []) + turn_target_ids))

    return {
        "actions_taken": ["update_memory"],
        "conversation_summary": updated_summary,
        "active_documents": merged_docs,
        "next_step": "END"
    }


# ============================================================================
# 4. Conditional Edge Router
# ============================================================================

def route_intent(state: AgentState) -> str:
    """Maps next_step from classify_intent to specialist agent nodes."""
    return state.get("next_step", "qa_agent")


# ============================================================================
# 5. Workflow Graph Assembly
# ============================================================================

def create_workflow(checkpointer: Optional[Any] = None) -> Any:
    """
    Constructs and compiles the LangGraph StateGraph workflow with
    specialist nodes, conditional intent routing, and state checkpointing.
    """
    workflow = StateGraph(AgentState)

    # Add all agent nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    # Set Entry Point
    workflow.set_entry_point("classify_intent")

    # Add Conditional Routing from classify_intent to specialist agents
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent"
        }
    )

    # Add Edges from all specialist agents to update_memory
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    # update_memory finishes the turn
    workflow.add_edge("update_memory", END)

    # Compile with InMemorySaver checkpointer for state persistence across invocations
    active_checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
    return workflow.compile(checkpointer=active_checkpointer)
