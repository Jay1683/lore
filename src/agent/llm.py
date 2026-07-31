"""
Groq LLM client, wrapped in one place so every node imports from here
instead of constructing its own client with its own settings.
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"


def get_llm(temperature: float = 0.0) -> ChatGroq:
    """
    Returns a configured Groq chat model.

    temperature=0.0 is the default because most of what this agent will do
    (routing decisions, skill selection, tool calling) benefits from
    deterministic, non-creative output. Bump it up only for nodes that
    generate free-form prose for the user.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Create a .env file in the project "
            "root with: GROQ_API_KEY=your_key_here"
        )

    return ChatGroq(model=MODEL_NAME, temperature=temperature, api_key=api_key)
