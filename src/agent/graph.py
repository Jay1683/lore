"""
Phase 3.5 graph: two-stage routing.

Stage 1 (route_entry): is there a doc loaded at all?
    no  -> straight to chat_node
    yes -> classify_intent_node first

Stage 2 (classify_intent_node + route_after_classification): given a doc
IS loaded, does this specific question actually need it?
    The classifier is a small, cheap, deterministic LLM call whose only
    job is to output DOC or GENERAL. This is deliberately separate from
    the main answering call -- it's the same "let the agent decide"
    pattern we'll reuse in Phase 5 for skill selection, just applied one
    level earlier: before deciding *how* to use the doc (raw vs RAG), we
    first decide *whether* to use it at all.

    general -> chat_node       (doc ignored, answered from general knowledge)
    doc, small doc -> raw_context_node
    doc, large doc -> rag_node

Without this stage, a document sitting in context state biased every
answer toward that document even for unrelated questions (e.g. asking a
general trivia question while a textbook PDF is loaded) -- the model would
search the doc for an answer that was never in it.
"""

from langgraph.graph import END, START, StateGraph

from agent.llm import get_llm
from agent.retrieval import retrieve_chunks
from agent.state import AgentState

RAW_CONTEXT_TOKEN_THRESHOLD = 8000


def _latest_user_query(state: AgentState) -> str:
    """
    Pulls the most recent human message's text out of state["messages"].
    Messages can be LangChain message objects (with a .content attribute)
    or plain (role, content) tuples, depending on where they came from --
    this handles both.
    """
    last = state["messages"][-1]
    return last.content if hasattr(last, "content") else last[1]


def route_entry(state: AgentState) -> str:
    """Stage 1: is a document loaded at all?"""
    return "has_doc" if state.get("doc_token_count") is not None else "no_doc"


def classify_intent_node(state: AgentState) -> dict:
    """
    Stage 2 classifier. Asks the LLM a narrow yes/no-shaped question:
    does answering this require the uploaded document's content?

    temperature=0 (the get_llm default) matters here -- this is a
    routing decision, not a creative one, and we want it stable across
    repeated identical questions.
    """
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

    return {"doc_relevant": is_doc_relevant}


def route_after_classification(state: AgentState) -> str:
    """Stage 2 routing: general knowledge, or doc-relevant (raw vs rag)?"""
    if not state.get("doc_relevant", False):
        return "general"

    token_count = state.get("doc_token_count")
    if token_count is not None and token_count <= RAW_CONTEXT_TOKEN_THRESHOLD:
        return "raw"
    return "rag"


def chat_node(state: AgentState) -> dict:
    """
    Plain conversation, no doc context injected. Reached either because
    no doc is loaded, or because the classifier decided this question
    doesn't need the doc that IS loaded.
    """
    llm = get_llm()
    response = llm.invoke(state["messages"])
    label = "no_doc" if state.get("doc_token_count") is None else "general"
    return {"messages": [response], "route_decision": label}


def raw_context_node(state: AgentState) -> dict:
    """Doc is small enough to just include in full."""
    llm = get_llm()

    context_message = (
        "system",
        "The user has uploaded a document. Use the following content to "
        f"answer their question:\n\n{state['doc_text']}",
    )
    messages_with_context = [context_message] + state["messages"]

    response = llm.invoke(messages_with_context)
    return {"messages": [response], "route_decision": "raw"}


def rag_node(state: AgentState) -> dict:
    """Doc is large enough that we retrieve relevant chunks instead."""
    collection = state.get("retriever_collection")
    if collection is None:
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

    # Stage 1: doc loaded at all?
    graph.add_conditional_edges(
        START,
        route_entry,
        {
            "no_doc": "chat",
            "has_doc": "classify_intent",
        },
    )

    # Stage 2: given a doc IS loaded, is it relevant to this question?
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

    return graph.compile()


app = build_graph()
