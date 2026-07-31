"""
Shared state schema for the agent graph.

Every node in a LangGraph graph receives the current state and returns a
dict of updates. LangGraph merges those updates into the state before
handing it to the next node. This file defines what that state looks like.

We're starting minimal on purpose: just enough to prove messages flow
through the graph. We'll grow this in later phases (doc content, routing
decision, retrieved chunks, skill results, etc.) rather than over-designing
it now.
"""

from typing import Annotated, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    `messages` uses the `add_messages` reducer, which means new messages
    returned by a node get *appended* to the existing list rather than
    replacing it. This is the standard pattern for chat-style graphs —
    without the reducer, each node's return value would overwrite the
    whole conversation history instead of extending it.

    The doc-related fields are NotRequired because most turns won't have a
    document attached at all -- Streamlit only populates them when a file
    is loaded in the sidebar.
    """

    messages: Annotated[list, add_messages]

    # Full extracted text of the uploaded document, if any. Set once at
    # upload time in app.py -- the graph never re-extracts it.
    doc_text: NotRequired[str | None]

    # Approximate token count of doc_text, used by the routing node to
    # decide raw-context vs. RAG.
    doc_token_count: NotRequired[int | None]

    # Set by the routing node so downstream code (and our own debugging)
    # can see which path was taken: "no_doc" | "raw" | "rag".
    route_decision: NotRequired[str]

    # Set by classify_intent_node: whether the user's question actually
    # requires the uploaded document, or is unrelated general knowledge.
    # Only meaningful when a doc is loaded -- absent/ignored otherwise.
    doc_relevant: NotRequired[bool]

    # Original filename, used only to give the intent classifier something
    # human-readable to reason about (e.g. "a document titled X.pdf").
    doc_name: NotRequired[str]

    # The Chroma collection to query when the RAG path is taken. Built
    # once at upload time in app.py (see retrieval.build_collection) and
    # passed straight through -- the graph never builds or rebuilds it.
    retriever_collection: NotRequired[object | None]