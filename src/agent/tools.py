"""
Wraps each loaded Skill as a LangChain tool the agent can choose to call.

This is the actual "agent decides whether to use a skill" mechanism: each
skill becomes a tool whose `description` is copied straight from the
skill's own SKILL.md frontmatter. The LLM sees these tool descriptions
alongside the user's message and decides, via normal function-calling,
whether any of them are a fit -- exactly the same mechanism it uses to
decide whether to call any other tool.

Phase 5 scope: calling the tool loads and returns the skill's full
instructions text -- it does NOT execute anything yet (e.g. it doesn't
run scripts, write files, or touch a real .docx). That's Phase 6. This
mirrors the real progressive-disclosure design of the SKILL.md format:
only the short description sits in the model's context by default: the
full instructions are loaded on demand, only once the agent decides the
skill is actually relevant.
"""

from langchain_core.tools import StructuredTool

from agent.logging_config import logger
from agent.skills import Skill

# Keep the loaded instructions from blowing up the context window if a
# skill's body happens to be huge. 3000 chars is plenty for the agent to
# see what the skill covers without dominating the conversation.
# Lowered from 3000: this gets resent on every subsequent turn in the
# conversation history, same reasoning as MAX_OUTPUT_CHARS in
# code_execution.py -- large one-time tool results compound across a
# multi-step loop and contribute directly to hitting Groq's per-minute
# token limit.
MAX_INSTRUCTIONS_PREVIEW = 2000


def _make_skill_tool(skill: Skill) -> StructuredTool:
    def _invoke_skill(user_request: str) -> str:
        logger.info(f"[skill_tool:{skill.name}] invoked with request={user_request!r}")

        instructions_preview = skill.instructions[:MAX_INSTRUCTIONS_PREVIEW]
        truncated_note = (
            "\n\n[...instructions truncated for preview...]"
            if len(skill.instructions) > MAX_INSTRUCTIONS_PREVIEW
            else ""
        )

        return (
            f"[Skill '{skill.name}' selected for: {user_request!r}]\n\n"
            "Follow the instructions below to complete this request. Use "
            "the write_file tool to create any scripts you need, and "
            "run_command to execute them (e.g. `node script.js`, `npm "
            "install docx` if the docx package isn't already available, "
            "or `pandoc` for reading existing files). Reference files "
            f"for this skill (if any) live at: {skill.path}\n\n"
            f"{instructions_preview}{truncated_note}"
        )

    return StructuredTool.from_function(
        func=_invoke_skill,
        name=skill.name,
        description=skill.description,
    )


def build_skill_tools(skills: dict[str, Skill]) -> list[StructuredTool]:
    return [_make_skill_tool(skill) for skill in skills.values()]
