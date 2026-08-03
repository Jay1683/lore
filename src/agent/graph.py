"""
Phase 5 graph: adds a top-level skill-vs-QA triage before everything else.

Overall shape now:

    START
      -> classify_request_type   (LLM call: does this need a skill, or is
                                   it a normal question?)
      -> conditional:
           "skill" -> skill_agent <-> skill_tools   (tool-calling loop --
                                                       see below)
           "qa"    -> [existing doc-routing logic: classify_intent ->
                       chat / raw_context / rag, unchanged from before]

The skill_agent <-> skill_tools loop is the standard LangGraph ReAct
pattern for tool-calling:
    1. skill_agent calls the LLM with tools bound (llm.bind_tools(...)).
    2. If the LLM's response includes tool_calls, route to skill_tools
       (a prebuilt ToolNode that actually executes them) and then loop
       back to skill_agent so the LLM can see the tool's result and
       produce a final natural-language reply.
    3. If the LLM's response has no tool_calls, we're done -- go to END.

This loop is what makes "the agent decides whether to use a skill" a
real mechanism rather than a hardcoded if-statement: the LLM sees each
skill's description as a tool description and calls it (or doesn't)
using normal function-calling, exactly like any other tool.
"""

import functools
import time
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage

from agent.llm import get_llm
from agent.logging_config import logger
from agent.retrieval import retrieve_chunks
from agent.skills import load_skills
from agent.state import AgentState
from agent.tools import build_skill_tools
from agent.code_execution import build_execution_tools

RAW_CONTEXT_TOKEN_THRESHOLD = 8000

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Skills are loaded once, at import time (i.e. once per process, same
# reasoning as the logging setup) -- not re-scanned from disk on every
# request.
SKILLS_DIR = PROJECT_ROOT / "skills"
LOADED_SKILLS = load_skills(SKILLS_DIR)
SKILL_SELECTOR_TOOLS = build_skill_tools(LOADED_SKILLS)

# Single shared workspace directory (this is a personal, single-user
# project -- see code_execution.py's module docstring for why per-session
# isolation isn't built here). Execution tools are built once, same as
# the skill tools, and both sets are bound together to the agent -- so a
# skill-selector call and a follow-up write_file/run_command call happen
# in the same tool-calling loop.
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
EXECUTION_TOOLS = build_execution_tools(WORKSPACE_DIR)

SKILL_AGENT_TOOLS = SKILL_SELECTOR_TOOLS + EXECUTION_TOOLS

logger.info(
    f"Loaded {len(LOADED_SKILLS)} skill(s): {list(LOADED_SKILLS.keys())} | "
    f"agent tools: {[t.name for t in SKILL_AGENT_TOOLS]}"
)


def log_node(node_name):
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


def _extract_decision(content: str, options: list[str], default: str) -> str:
    """
    Pulls a single-word classification decision (e.g. "SKILL"/"QA") out of
    a model response, tolerating extra text around it.

    Why this exists: gpt-oss-120b is a reasoning model, and Groq has a
    documented (if uncommon) issue where reasoning tokens leak into the
    final answer content instead of staying in the separate `reasoning`
    field. A naive `content.startswith("SKILL")` breaks the moment
    there's so much as a leading newline or a stray reasoning sentence.

    Strategy: check the LAST non-empty line first (a model that reasons
    out loud before answering usually puts the real answer last), then
    fall back to searching the whole response, then fall back to
    `default` -- this never raises on an unexpected shape, it just picks
    the safer of the two options.
    """
    text = content.strip().upper()
    if not text:
        return default

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = lines[-1]
        for option in options:
            if option in last_line:
                return option

    for option in options:
        if option in text:
            return option

    return default


def _latest_user_query(state: AgentState) -> str:
    last = state["messages"][-1]
    return last.content if hasattr(last, "content") else last[1]


def _safe_llm_invoke(
    llm, messages, node_name: str, retry_on_tool_failure: bool = False
) -> AIMessage:
    """
    Wraps llm.invoke() so a failure -- a malformed tool-call generation, a
    Groq API error, a rate limit, a network hiccup -- becomes a normal
    AIMessage the graph can continue with, instead of an unhandled
    exception that crashes the whole Streamlit app.

    This is a real production concern, not a hypothetical one: this
    exists because a 400 "Failed to parse tool call arguments as JSON"
    error (from the model generating a malformed/oversized tool call)
    took down the entire app before this wrapper existed. Every node's
    LLM call goes through this now.

    retry_on_tool_failure=True (used only by skill_agent, where large
    generated scripts are the norm) adds ONE retry when the failure looks
    like a tool-call-generation problem specifically. Unlike a blind
    retry, this appends a system message telling the model concretely
    what went wrong -- a model that just generated 300 repeated blank
    paragraphs and got truncated needs to be told that happened, not just
    asked to try again and risk repeating the identical mistake.

    The returned AIMessage has no tool_calls, so if both the original
    call and the retry fail, has_pending_tool_calls() naturally routes to
    "done" and the loop ends cleanly instead of retrying forever.
    """
    try:
        return llm.invoke(messages)
    except Exception as exc:
        error_str = str(exc)
        looks_like_tool_call_failure = any(
            marker in error_str
            for marker in (
                "tool_use_failed",
                "Failed to parse tool call",
                "Failed to call a function",
            )
        )
        looks_like_rate_limit = (
            "rate_limit_exceeded" in error_str or "tokens per minute" in error_str
        )

        if looks_like_rate_limit:
            # A TPM (tokens-per-minute) limit is a rolling window, not a
            # permanent block -- waiting lets it clear. This is an
            # account-tier constraint (the free/on-demand Groq tier caps
            # this fairly low), not something retrying "fixes" for good,
            # but a short wait genuinely resolves the immediate failure
            # most of the time.
            backoff_seconds = 20
            logger.warning(
                f"[{node_name}] hit a Groq rate limit, waiting "
                f"{backoff_seconds}s before retrying once: {error_str[:200]}"
            )
            time.sleep(backoff_seconds)
            try:
                return llm.invoke(messages)
            except Exception as retry_exc:
                logger.error(
                    f"[{node_name}] retry after rate-limit backoff also "
                    f"failed: {type(retry_exc).__name__}: {retry_exc}"
                )
                exc = retry_exc

        elif retry_on_tool_failure and looks_like_tool_call_failure:
            logger.warning(
                f"[{node_name}] tool-call generation failed, retrying once "
                f"with corrective guidance: {error_str[:300]}"
            )
            correction = (
                "system",
                "Your previous attempt failed because the generated content "
                "was too long and got cut off mid-generation, producing "
                "invalid JSON -- this is usually caused by repeating many "
                "near-identical elements (e.g. dozens of blank paragraphs "
                "for spacing) instead of using a spacing/margin property. "
                "Try again with a SHORTER, more concise script: no more "
                "than 2 consecutive blank/repeated elements anywhere, and "
                "use the library's spacing properties for layout instead.",
            )
            try:
                return llm.invoke(list(messages) + [correction])
            except Exception as retry_exc:
                logger.error(
                    f"[{node_name}] retry also failed: "
                    f"{type(retry_exc).__name__}: {retry_exc}"
                )
                exc = retry_exc  # fall through to the generic failure message below

        logger.error(f"[{node_name}] LLM call failed: {type(exc).__name__}: {exc}")
        return AIMessage(
            content=(
                "I ran into an error generating a response "
                f"({type(exc).__name__}). This can happen with long or "
                "complex generations -- try rephrasing your request more "
                "simply, or breaking it into smaller steps."
            )
        )


# --- Stage 0: skill vs QA triage ----------------------------------------


@log_node("classify_request_type")
def classify_request_type_node(state: AgentState) -> dict:
    """
    The very first decision in the graph: does this request want to
    invoke one of our loaded skills, or is it a normal question?
    """
    llm = get_llm(reasoning_effort="low")
    query = _latest_user_query(state)

    if LOADED_SKILLS:
        skill_list = "\n".join(
            f"- {s.name}: {s.description}" for s in LOADED_SKILLS.values()
        )
    else:
        skill_list = "(no skills currently loaded)"

    prompt = [
        (
            "system",
            "You triage requests for an agent with access to these skills:\n"
            f"{skill_list}\n\n"
            "Decide whether the user's request should use one of these "
            "skills, or is a normal question/conversation that doesn't "
            "need any of them. Respond with exactly one word: SKILL or QA.",
        ),
        ("user", query),
    ]
    response = _safe_llm_invoke(llm, prompt, "classify_request_type")
    decision_text = response.content.strip().upper()
    extracted = _extract_decision(decision_text, ["SKILL", "QA"], default="QA")
    request_type = "skill" if extracted == "SKILL" else "qa"

    logger.info(
        f"[classify_request_type] query={query!r} -> raw_response={decision_text[:200]!r} "
        f"-> extracted={extracted!r} -> request_type={request_type}"
    )
    return {"request_type": request_type}


def route_after_request_classification(state: AgentState) -> str:
    """
    Combines the skill/qa decision with the old "has a doc been loaded"
    check (previously its own top-level function) -- both are cheap,
    state-only checks, so folding them into one router keeps the graph
    shape flatter.
    """
    if state.get("request_type") == "skill":
        decision = "skill"
    elif state.get("doc_token_count") is not None:
        decision = "has_doc"
    else:
        decision = "no_doc"

    logger.info(f"[route_after_request_classification] -> {decision}")
    return decision


# --- Skill path: tool-calling loop --------------------------------------


# 8 was too tight for genuine multi-step debugging (write -> run -> fail
# -> install missing dep -> run -> fail differently -> inspect -> fix ->
# run -> success easily needs 10+ turns when each tool call is its own
# iteration, as it is here). 15 gives real debugging room while still
# bounding runaway cost.
MAX_SKILL_LOOP_ITERATIONS = 15

# Prepended on EVERY skill_agent iteration (unlike doc context, which is
# injected once -- this is small and cheap, and needs to survive the
# whole loop, not just the first turn).
#
# Why this exists: without it, the model reliably stops after calling the
# skill-selector tool (e.g. "docx") and just narrates a plausible-sounding
# final answer, without ever calling write_file/run_command to actually
# produce anything. Selecting a skill only loads its instructions into
# context -- it does not execute them. That distinction has to be stated
# explicitly, or the model treats "I read what to do" as equivalent to
# "I did it."
SKILL_AGENT_SYSTEM_PROMPT = (
    "You have tools to select a skill (e.g. docx) and to actually execute "
    "it: write_file, run_command, and list_workspace_files. "
    "IMPORTANT: calling a skill-selector tool only LOADS that skill's "
    "instructions -- it does not create, modify, or produce anything by "
    "itself. If the task requires producing a file, you must follow up "
    "by using write_file to write any necessary script(s), then "
    "run_command to actually execute them. "
    "Before telling the user a file is ready, verify it with "
    "list_workspace_files or by checking that run_command returned "
    "exit_code=0 for the command that should have produced it. Never "
    "claim a file was created without that concrete evidence -- if a "
    "command failed, say so and either retry with a fix or tell the user "
    "what went wrong.\n\n"
    "CRITICAL RULE for scripts you write: NEVER write more than 2 "
    "consecutive blank elements (e.g. `new Paragraph({})`) in a row for "
    "spacing. This is not a style preference -- repeating this pattern "
    "dozens of times has directly caused generation failures before. "
    "For vertical spacing or centering content on a page, use the "
    "library's spacing/margin properties on a SINGLE element instead, "
    'for example: `new Paragraph({ text: "Title", spacing: { before: '
    "4000, after: 400 } })`. A docx-js script for a normal document "
    "should rarely exceed ~80 lines -- if you find yourself writing "
    "many similar repeated lines, stop and use a loop or a spacing "
    "property instead of listing them out individually."
)


def _build_doc_search_tool(collection) -> StructuredTool:
    """
    A per-request tool (built fresh each call, not at module level, since
    it closes over a specific document's Chroma collection) that lets the
    skill agent pull relevant excerpts from a large uploaded document --
    the same retrieve_chunks() used by the QA path's rag_node, just
    exposed as something the agent can call rather than a hardcoded step.
    """

    def _search_doc(query: str) -> str:
        chunks = retrieve_chunks(collection, query, k=4)
        logger.info(
            f"[search_uploaded_document] query={query!r} retrieved {len(chunks)} chunks"
        )
        return "\n\n---\n\n".join(chunks)

    return StructuredTool.from_function(
        func=_search_doc,
        name="search_uploaded_document",
        description=(
            "Search the currently uploaded document for content relevant "
            "to a query. Use this to pull specific information from the "
            "uploaded document when completing a skill-based task (e.g. "
            "summarizing or reformatting it into a new file)."
        ),
    )


@log_node("skill_agent")
def skill_agent_node(state: AgentState) -> dict:
    """
    Calls the LLM with skill-selector AND execution tools bound. The LLM
    decides -- via normal function-calling, across possibly several
    iterations of this loop -- which skill fits, then how to actually use
    write_file/run_command to carry out that skill's instructions.

    On the FIRST iteration only, if a document is loaded, we inject its
    content (or, for large docs, a search tool) as a system message --
    without this, the skill agent has no way to know a document exists at
    all, since unlike the QA path it doesn't go through classify_intent /
    raw_context_node / rag_node.
    """
    current_iterations = state.get("skill_loop_iterations", 0)

    if current_iterations >= MAX_SKILL_LOOP_ITERATIONS:
        # Checked BEFORE calling the LLM, not after: the old version let
        # the LLM generate a full response (wasting real time and money)
        # and then discarded it once the cap was hit, leaving the user
        # with silence -- exactly the "stopped in between" bug this
        # replaces. Now we stop cleanly with an honest summary instead.
        logger.warning(
            f"[skill_agent] hit MAX_SKILL_LOOP_ITERATIONS="
            f"{MAX_SKILL_LOOP_ITERATIONS} -- stopping with a summary "
            "instead of another LLM call"
        )
        summary = AIMessage(
            content=(
                f"I wasn't able to finish this within "
                f"{MAX_SKILL_LOOP_ITERATIONS} tool-calling steps -- the "
                "task hit errors that took multiple attempts to work "
                "through. Check logs/agent_run.log for exactly what was "
                "tried (including any command output), and check the "
                "sidebar for any files that did get created. Feel free "
                "to ask me to continue, ideally with any extra detail "
                "that might help (e.g. if you know what's failing)."
            )
        )
        return {"messages": [summary], "skill_loop_iterations": current_iterations}

    iterations = current_iterations + 1
    tools = list(SKILL_AGENT_TOOLS)
    system_messages = [("system", SKILL_AGENT_SYSTEM_PROMPT)]

    if iterations == 1 and state.get("doc_text") is not None:
        doc_name = state.get("doc_name", "the uploaded document")
        token_count = state.get("doc_token_count") or 0

        if token_count <= RAW_CONTEXT_TOKEN_THRESHOLD:
            doc_context = (
                f"A document named '{doc_name}' has already been uploaded "
                f"by the user -- you do not need to ask them for it. Its "
                f"full content is:\n\n{state['doc_text']}"
            )
        else:
            doc_context = (
                f"A document named '{doc_name}' has already been uploaded "
                f"by the user (too large to include in full -- "
                f"~{token_count} tokens). You do not need to ask them for "
                "it. Use the search_uploaded_document tool to retrieve "
                "relevant excerpts as needed."
            )
            collection = state.get("retriever_collection")
            if collection is not None:
                tools = tools + [_build_doc_search_tool(collection)]

        system_messages = system_messages + [("system", doc_context)]
        logger.info(
            f"[skill_agent] injected doc context for '{doc_name}' "
            f"({'raw' if token_count <= RAW_CONTEXT_TOKEN_THRESHOLD else 'search tool'})"
        )

    messages = system_messages + list(state["messages"])

    llm = get_llm().bind_tools(tools)
    response = _safe_llm_invoke(
        llm, messages, "skill_agent", retry_on_tool_failure=True
    )
    return {"messages": [response], "skill_loop_iterations": iterations}


def has_pending_tool_calls(state: AgentState) -> str:
    """
    The cap check now lives in skill_agent_node (before it even calls the
    LLM) -- this function only needs to check whether the LAST response
    has pending tool calls, since a capped-out summary message naturally
    has none and will route to "done" here without any special-casing.
    """
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "call_tools"
    return "done"


skill_tools_node = ToolNode(SKILL_AGENT_TOOLS)


# --- QA path (unchanged from Phase 3.5) ----------------------------------


@log_node("classify_intent")
def classify_intent_node(state: AgentState) -> dict:
    llm = get_llm(reasoning_effort="low")
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

    response = _safe_llm_invoke(llm, classification_prompt, "classify_intent")
    decision_text = response.content.strip().upper()
    extracted = _extract_decision(decision_text, ["DOC", "GENERAL"], default="GENERAL")
    is_doc_relevant = extracted == "DOC"

    logger.info(
        f"[classify_intent] query={query!r} -> raw_response={decision_text[:200]!r} "
        f"-> extracted={extracted!r} -> doc_relevant={is_doc_relevant}"
    )
    return {"doc_relevant": is_doc_relevant}


def route_after_classification(state: AgentState) -> str:
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
    llm = get_llm()
    response = _safe_llm_invoke(llm, state["messages"], "chat")
    label = "no_doc" if state.get("doc_token_count") is None else "general"
    return {"messages": [response], "route_decision": label}


@log_node("raw_context")
def raw_context_node(state: AgentState) -> dict:
    llm = get_llm()

    doc_text = state["doc_text"]
    logger.info(f"[raw_context] injecting full doc, {len(doc_text)} chars")

    context_message = (
        "system",
        "The user has uploaded a document. Use the following content to "
        f"answer their question:\n\n{doc_text}",
    )
    messages_with_context = [context_message] + state["messages"]

    response = _safe_llm_invoke(llm, messages_with_context, "raw_context")
    return {"messages": [response], "route_decision": "raw"}


@log_node("rag")
def rag_node(state: AgentState) -> dict:
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
    response = _safe_llm_invoke(llm, messages_with_context, "rag")
    return {"messages": [response], "route_decision": "rag"}


# --- Graph assembly -------------------------------------------------------


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_request_type", classify_request_type_node)
    graph.add_node("skill_agent", skill_agent_node)
    graph.add_node("skill_tools", skill_tools_node)
    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("chat", chat_node)
    graph.add_node("raw_context", raw_context_node)
    graph.add_node("rag", rag_node)

    graph.add_edge(START, "classify_request_type")

    graph.add_conditional_edges(
        "classify_request_type",
        route_after_request_classification,
        {
            "skill": "skill_agent",
            "has_doc": "classify_intent",
            "no_doc": "chat",
        },
    )

    # Skill tool-calling loop
    graph.add_conditional_edges(
        "skill_agent",
        has_pending_tool_calls,
        {
            "call_tools": "skill_tools",
            "done": END,
        },
    )
    graph.add_edge("skill_tools", "skill_agent")

    # QA path, unchanged
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
        "Graph compiled: __start__ -> classify_request_type -> "
        "{skill_agent<->skill_tools | classify_intent -> chat/raw_context/rag} -> __end__"
    )
    return graph.compile()


app = build_graph()
