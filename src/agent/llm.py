"""
Groq LLM client, wrapped in one place so every node imports from here
instead of constructing its own client with its own settings.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


def get_llm(
    temperature: float = 0.0,
    max_tokens: int = None,
    reasoning_effort: str = "medium",
) -> ChatGroq:
    """
    Returns a configured Groq chat model.

    temperature=0.0 is the default because most of what this agent will do
    (routing decisions, skill selection, tool calling) benefits from
    deterministic, non-creative output. Bump it up only for nodes that
    generate free-form prose for the user.

    max_tokens=4096 caps how much a single response can generate. This
    matters most for tool-calling nodes: an open model asked to produce a
    long script inside a JSON tool-call argument can fall into repetition
    (e.g. emitting the same blank element hundreds of times) instead of
    finishing cleanly. A cap doesn't prevent that pattern, but it bounds
    the damage -- a shorter, still-malformed generation fails faster and
    cheaper than a 4000-line one.

    reasoning_effort controls how much internal reasoning gpt-oss-120b
    does before answering ("low" | "medium" | "high"). Pass "low" for
    single-word/simple classification calls (classify_request_type,
    classify_intent) -- they don't need deep reasoning, and keeping
    effort low reduces both latency and the (documented, if rare) risk of
    reasoning tokens leaking into the final answer text on Groq.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file in the project "
            "root with: GROQ_API_KEY=your_key_here"
        )

    return ChatGroq(
        model=MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        api_key=api_key,
    )