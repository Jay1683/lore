"""
Scoped code-execution tools for the skill-execution path.

READ THIS BEFORE EXTENDING: this is a *scoped* sandbox, not a *secure*
one. Commands run as your own OS user, with your own permissions, inside
a dedicated workspace directory. There is no container, no network
isolation, and no real privilege boundary -- the agent's generated shell
commands run with exactly the power you have in your own terminal. The
protections here are:

  1. Commands run with cwd fixed to the workspace directory.
  2. A timeout, so a hung command doesn't block the app indefinitely.
  3. A denylist of a few obviously destructive patterns.

None of that is a security guarantee -- a sufficiently determined (or
sufficiently confused) command can still escape the workspace via `cd ..`,
an absolute path, or a pattern we didn't think to block. This is
appropriate for a personal, local, single-user project where you're
reviewing what the agent does (check logs/agent_run.log after every run).
It is NOT appropriate to expose to untrusted users or run as a
multi-tenant service without real sandboxing (a container, gVisor,
Firecracker, etc.) -- that would be a different, much larger project.
"""

import subprocess
from pathlib import Path

from langchain_core.tools import StructuredTool

from agent.logging_config import logger

COMMAND_TIMEOUT_SECONDS = (
    90  # generous -- npm installs and soffice conversions are slow
)
# Lowered from 4000: every run_command result gets resent as part of the
# full conversation history on EVERY subsequent LLM call in the loop
# (chat APIs are stateless -- there's no way to send only "what's new").
# A few large tool outputs compound fast and were a direct contributor to
# hitting Groq's per-minute token limit during a multi-step debug loop.
MAX_OUTPUT_CHARS = 2000

_DENYLIST_SUBSTRINGS = [
    "rm -rf /",
    "rm -rf ~",
    ":(){ :|:& };:",  # fork bomb
    "sudo ",
    "mkfs",
    "dd if=",
    "> /dev/sda",
]


def _is_denied(command: str) -> bool:
    lowered = command.lower()
    return any(bad in lowered for bad in _DENYLIST_SUBSTRINGS)


def build_execution_tools(workspace_dir: Path) -> list[StructuredTool]:
    """
    Builds write_file / run_command / list_workspace_files tools, all
    scoped to workspace_dir. Called once at graph-module import time with
    a fixed workspace directory -- see graph.py.
    """
    workspace_dir.mkdir(parents=True, exist_ok=True)

    def _write_file(filename: str, content: str) -> str:
        target = (workspace_dir / filename).resolve()
        workspace_resolved = workspace_dir.resolve()
        if workspace_resolved not in target.parents and target != workspace_resolved:
            logger.warning(f"[write_file] refused path escape: {filename!r}")
            return f"Refused: '{filename}' resolves outside the workspace."

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info(f"[write_file] wrote {len(content)} chars to {target}")
        return f"Wrote {filename} ({len(content)} chars) to the workspace."

    def _run_command(command: str) -> str:
        if _is_denied(command):
            logger.warning(f"[run_command] DENIED: {command!r}")
            return "Refused: command matched a denylisted destructive pattern."

        logger.info(f"[run_command] running: {command!r} (cwd={workspace_dir})")
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"[run_command] TIMEOUT after {COMMAND_TIMEOUT_SECONDS}s: {command!r}"
            )
            return f"Command timed out after {COMMAND_TIMEOUT_SECONDS}s."

        logger.info(
            f"[run_command] exit_code={result.returncode} "
            f"stdout_len={len(result.stdout)} stderr_len={len(result.stderr)}"
        )
        # Previously only lengths were logged, which is useless for
        # actually diagnosing a failure from the log file alone -- you'd
        # have to ask the user to paste the error separately. Log a
        # preview of the actual content, especially on failure.
        if result.returncode != 0:
            stderr_preview = result.stderr[:800].replace("\n", " | ")
            logger.warning(
                f"[run_command] non-zero exit -- stderr preview: {stderr_preview!r}"
            )
        elif result.stdout:
            stdout_preview = result.stdout[:300].replace("\n", " | ")
            logger.info(f"[run_command] stdout preview: {stdout_preview!r}")
        output = (
            f"exit_code={result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        return output[:MAX_OUTPUT_CHARS]

    def _list_workspace_files() -> str:
        files = sorted(
            p.relative_to(workspace_dir).as_posix()
            for p in workspace_dir.rglob("*")
            if p.is_file()
        )
        return "\n".join(files) if files else "(workspace is empty)"

    write_file_tool = StructuredTool.from_function(
        func=_write_file,
        name="write_file",
        description=(
            "Write a text file (e.g. a Node.js script) into the sandboxed "
            "workspace directory. Provide a relative filename and the "
            "full file content."
        ),
    )
    run_command_tool = StructuredTool.from_function(
        func=_run_command,
        name="run_command",
        description=(
            "Run a shell command inside the sandboxed workspace directory "
            "(e.g. 'node create_doc.js', 'npm install docx', "
            "'pandoc -t markdown file.docx'). Returns exit code, stdout, "
            "and stderr."
        ),
    )
    list_files_tool = StructuredTool.from_function(
        func=_list_workspace_files,
        name="list_workspace_files",
        description=(
            "List all files currently in the sandboxed workspace "
            "directory, to check what a command produced."
        ),
    )

    return [write_file_tool, run_command_tool, list_files_tool]
