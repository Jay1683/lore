"""
Graph with two-stage routing (see route_entry / classify_intent_node /
route_after_classification below) and full flow logging.

Every node is wrapped with @log_node, which logs entry, exit, elapsed
time, and which state keys it updated. The two routing functions
(route_entry, route_after_classification) log their decisions directly
since they aren't graph nodes -- they're the conditional-edge functions
LangGraph calls between nodes.

Check logs/agent_run.log after a run to see the full path a request took.
"""

import functools
import time

from langgraph.graph import END, START, StateGraph

from agent.llm import get_llm
from agent.logging_config import logger
from agent.retrieval import retrieve_chunks
from agent.state import AgentState

RAW_CONTEXT_TOKEN_THRESHOLD = 8000


def log_node(node_name):
    """
    Decorator for graph nodes: logs entry, exit, elapsed time, and which
    state keys the node returned (i.e. what it actually changed).
    """

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> dict:
            logger.info(f"-> ENTER {node_name}")
            start = time.perf_counter()
            result = fn(state)
            elapsed = time.perf_counter() - start
            logger.info(
                f"<- EXIT  {node_name} ({elapsed:.2f}s) | "
                f"updated keys: {list(result.keys())}"
            )
            return result

        return wrapper

    return decorator


def _latest_user_query(state: AgentState) -> str:
    last = state["messages"][-1]
    return last.content if hasattr(last, "content") else last[1]


def route_entry(state: AgentState) -> str:
    """Stage 1: is a document loaded at all?"""
    decision = "has_doc" if state.get("doc_token_count") is not None else "no_doc"
    logger.info(
        f"[route_entry] doc_token_count={state.get('doc_token_count')} -> {decision}"
    )
    return decision


@log_node("classify_intent")
def classify_intent_node(state: AgentState) -> dict:
    """Stage 2 classifier: does this question actually need the doc?"""
    llm = get_llm()
    query = _latest_user_query(state)
    doc_name = state.get("doc_name", "the uploaded document")

    classification_prompt = [
        (
            "system",
            f"A document titled '{doc_name}' is available. Decide whether "
            "answering the user's question requires looking into this "
            "document's specific content, or whether it's a general "
            "question you can answer from your own knowledge without it. "
            "General knowledge, definitions, or topics unrelated to the "
            "document's subject matter should be GENERAL even if the "
            "wording sounds like it could be about the document.\n\n"
            "Respond with exactly one word: DOC or GENERAL.",
        ),
        ("user", query),
    ]

    response = llm.invoke(classification_prompt)
    decision_text = response.content.strip().upper()
    is_doc_relevant = decision_text.startswith("DOC")

    logger.info(
        f"[classify_intent] query={query!r} -> raw_response={decision_text!r} "
        f"-> doc_relevant={is_doc_relevant}"
    )

    return {"doc_relevant": is_doc_relevant}


def route_after_classification(state: AgentState) -> str:
    """Stage 2 routing: general knowledge, or doc-relevant (raw vs rag)?"""
    if not state.get("doc_relevant", False):
        decision = "general"
    else:
        token_count = state.get("doc_token_count")
        if token_count is not None and token_count <= RAW_CONTEXT_TOKEN_THRESHOLD:
            decision = "raw"
        else:
            decision = "rag"

    logger.info(
        f"[route_after_classification] doc_relevant={state.get('doc_relevant')} "
        f"token_count={state.get('doc_token_count')} -> {decision}"
    )
    return decision


@log_node("chat")
def chat_node(state: AgentState) -> dict:
    """Plain conversation, no doc context injected."""
    llm = get_llm()
    response = llm.invoke(state["messages"])
    label = "no_doc" if state.get("doc_token_count") is None else "general"
    return {"messages": [response], "route_decision": label}


@log_node("raw_context")
def raw_context_node(state: AgentState) -> dict:
    """Doc is small enough to just include in full."""
    llm = get_llm()

    doc_text = state["doc_text"]
    logger.info(f"[raw_context] injecting full doc, {len(doc_text)} chars")

    context_message = (
        "system",
        "The user has uploaded a document. Use the following content to "
        f"answer their question:\n\n{doc_text}",
    )
    messages_with_context = [context_message] + state["messages"]

    response = llm.invoke(messages_with_context)
    return {"messages": [response], "route_decision": "raw"}


@log_node("rag")
def rag_node(state: AgentState) -> dict:
    """Doc is large enough that we retrieve relevant chunks instead."""
    collection = state.get("retriever_collection")
    if collection is None:
        logger.error("[rag] no retriever_collection in state -- cannot proceed")
        return {
            "messages": [
                (
                    "assistant",
                    "Internal error: RAG path was chosen but no retriever "
                    "collection was provided.",
                )
            ],
            "route_decision": "rag",
        }

    query = _latest_user_query(state)
    relevant_chunks = retrieve_chunks(collection, query, k=4)

    logger.info(f"[rag] query={query!r} retrieved {len(relevant_chunks)} chunks:")
    for i, chunk in enumerate(relevant_chunks):
        preview = chunk[:100].replace("\n", " ")
        logger.info(f"[rag]   chunk {i}: {preview!r}... ({len(chunk)} chars)")

    context_text = "\n\n---\n\n".join(relevant_chunks)
    context_message = (
        "system",
        "The user has uploaded a document too large to include in full. "
        "Below are the most relevant excerpts retrieved for their "
        f"question:\n\n{context_text}",
    )
    messages_with_context = [context_message] + state["messages"]

    llm = get_llm()
    response = llm.invoke(messages_with_context)
    return {"messages": [response], "route_decision": "rag"}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("chat", chat_node)
    graph.add_node("raw_context", raw_context_node)
    graph.add_node("rag", rag_node)

    graph.add_conditional_edges(
        START,
        route_entry,
        {
            "no_doc": "chat",
            "has_doc": "classify_intent",
        },
    )

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "general": "chat",
            "raw": "raw_context",
            "rag": "rag",
        },
    )

    graph.add_edge("chat", END)
    graph.add_edge("raw_context", END)
    graph.add_edge("rag", END)

    logger.info(
        "Graph compiled: __start__ -> [classify_intent?] -> {chat|raw_context|rag} -> __end__"
    )
    return graph.compile()


app = build_graph()
