"""
Phase 2: Streamlit shell.

This file only handles UI state and wiring to the graph. It intentionally
does NOT read, parse, or route the uploaded document yet — that's Phase 3
onward. Right now we're only proving:

  1. Chat history persists correctly across Streamlit reruns
     (Streamlit re-executes this whole script on every interaction, so
     without session_state, "history" would reset every time you type).
  2. A file can be uploaded and its presence is visible to the graph call.
  3. The graph from Phase 1 still answers correctly when driven from here
     instead of a raw python -c call.
"""

import streamlit as st

from agent.documents import count_tokens, extract_text
from agent.graph import RAW_CONTEXT_TOKEN_THRESHOLD, LOADED_SKILLS, WORKSPACE_DIR
from agent.graph import app as agent_app
from agent.retrieval import build_collection

st.set_page_config(page_title="Agentic Doc System", page_icon="🗂️")
st.title("Agentic Doc System")

# --- Session state setup ---------------------------------------------
# `messages` holds the full chat history as LangChain-style (role, content)
# tuples. We keep our OWN copy here rather than relying on the graph's
# internal state, because Streamlit reruns the whole script top-to-bottom
# on every interaction — a fresh `agent_app.invoke()` call has no memory of
# previous turns unless we hand it the full history ourselves each time.
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_doc" not in st.session_state:
    st.session_state.uploaded_doc = None

# --- Sidebar: file upload ----------------------------------------------
with st.sidebar:
    st.subheader("Available skills")
    if LOADED_SKILLS:
        for skill in LOADED_SKILLS.values():
            st.caption(f"**{skill.name}** — {skill.description[:100]}...")
    else:
        st.caption("No skills loaded.")

    st.subheader("Upload a document")
    uploaded_file = st.file_uploader(
        "Choose a file", type=["pdf", "docx"], key="file_uploader"
    )
    if uploaded_file is not None:
        # Only re-extract if this is a genuinely new file, not a rerun of
        # the same one -- Streamlit reruns this script on every
        # interaction, and extraction (especially for PDFs) isn't free.
        if (
            st.session_state.uploaded_doc is None
            or st.session_state.uploaded_doc["name"] != uploaded_file.name
        ):
            file_bytes = uploaded_file.getvalue()
            doc_text = extract_text(file_bytes, uploaded_file.name)
            token_count = count_tokens(doc_text)

            doc_entry = {
                "name": uploaded_file.name,
                "text": doc_text,
                "token_count": token_count,
                "collection": None,
            }

            # Only build embeddings if we'll actually need them -- no
            # point embedding a doc that's going down the raw-context path.
            if token_count > RAW_CONTEXT_TOKEN_THRESHOLD:
                with st.spinner("Indexing document for retrieval..."):
                    doc_entry["collection"] = build_collection(
                        doc_id=uploaded_file.name, text=doc_text
                    )

            st.session_state.uploaded_doc = doc_entry
        st.success(f"Loaded: {uploaded_file.name}")

    if st.session_state.uploaded_doc:
        doc = st.session_state.uploaded_doc
        route = (
            "raw context"
            if doc["token_count"] <= RAW_CONTEXT_TOKEN_THRESHOLD
            else "RAG"
        )
        st.caption(
            f"Active document: {doc['name']} "
            f"(~{doc['token_count']} tokens \u2192 will use **{route}**)"
        )

    st.subheader("Generated files")
    DOWNLOADABLE_EXTENSIONS = {".docx", ".pdf", ".pptx", ".xlsx", ".dotx"}
    workspace_files = (
        sorted(
            p
            for p in WORKSPACE_DIR.glob("*")
            if p.is_file() and p.suffix.lower() in DOWNLOADABLE_EXTENSIONS
        )
        if WORKSPACE_DIR.exists()
        else []
    )

    if workspace_files:
        for f in workspace_files:
            st.download_button(
                label=f"\U0001f4c4 {f.name}",
                data=f.read_bytes(),
                file_name=f.name,
                key=f"download_{f.name}",
            )
    else:
        st.caption("Nothing generated yet.")

# --- Render existing chat history --------------------------------------
for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

# --- Handle new input ----------------------------------------------------
user_input = st.chat_input("Ask something...")

if user_input:
    # Show the user's message immediately
    st.session_state.messages.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build the graph input. If a doc is loaded, its text and token count
    # ride along so the routing node (route_by_doc in graph.py) can decide
    # no_doc / raw / rag -- see Phase 3 notes there.
    graph_input = {"messages": st.session_state.messages}
    doc = st.session_state.uploaded_doc
    if doc is not None:
        graph_input["doc_text"] = doc["text"]
        graph_input["doc_token_count"] = doc["token_count"]
        graph_input["retriever_collection"] = doc["collection"]
        graph_input["doc_name"] = doc["name"]

    # Snapshot which downloadable files exist BEFORE this turn, so we can
    # tell the user exactly what's new afterward (the sidebar section
    # above already ran earlier in this same script pass -- it can't see
    # files this turn is about to create).
    files_before = set(WORKSPACE_DIR.glob("*")) if WORKSPACE_DIR.exists() else set()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = agent_app.invoke(graph_input)
                reply = result["messages"][-1].content
            except Exception as exc:
                # Last line of defense: _safe_llm_invoke inside graph.py
                # already catches LLM-call failures, but this catches
                # anything else in the graph (e.g. a tool execution
                # error) so a single bad turn never takes down the whole
                # app -- the user sees an error message and can keep
                # chatting, instead of Streamlit's raw traceback screen.
                logger_reply = f"Something went wrong processing that request: {type(exc).__name__}: {exc}"
                reply = (
                    "Sorry, I ran into an unexpected error handling that "
                    "request. Check logs/agent_run.log for details, and "
                    "feel free to try again or rephrase."
                )
                st.error(logger_reply)

            files_after = (
                set(WORKSPACE_DIR.glob("*")) if WORKSPACE_DIR.exists() else set()
            )
            new_files = sorted(
                p.name
                for p in (files_after - files_before)
                if p.suffix.lower() in DOWNLOADABLE_EXTENSIONS
            )
            if new_files:
                file_list = ", ".join(new_files)
                reply = (
                    reply
                    + f"\n\n\U0001f4c4 **New file(s) ready in the sidebar:** {file_list}"
                )

            st.markdown(reply)

    st.session_state.messages.append(("assistant", reply))

    if new_files:
        # The sidebar's file list was already rendered earlier in this
        # script pass, before these files existed -- a rerun forces a
        # fresh top-to-bottom pass so it picks them up immediately,
        # instead of only appearing after your NEXT message.
        st.rerun()
