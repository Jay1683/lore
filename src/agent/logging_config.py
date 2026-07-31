"""
Logging setup for the agent graph.

Key design point: this module's setup code (creating the FileHandler with
mode="w") runs exactly ONCE per process, not once per Streamlit rerun.

Why that matters: `streamlit run app.py` starts a single long-lived Python
process. Every time you type a message or click something, Streamlit
re-executes app.py's top-to-bottom script -- but Python's module import
system (sys.modules) caches already-imported modules, so `from
agent.logging_config import logger` on rerun #2, #3, #47 just fetches the
already-configured logger instead of re-running this file's setup code.

That gives us exactly the behavior you asked for: the log file resets when
you freshly launch the app (`streamlit run app.py`), and then accumulates
every node/routing event for the rest of that session -- it does NOT wipe
itself after every single message.

If you ever want to force a truly new log per Streamlit *rerun* instead of
per process, that's a different (and noisier) mechanism -- ask if you want
that variant.
"""

import logging
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "agent_run.log"

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# Guard against duplicate handlers -- harmless here since this only
# imports once, but cheap insurance if the module ever gets touched by a
# reload tool.
if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, mode="w")  # "w" = truncate on open
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Don't also bubble up to the root logger -- avoids duplicate lines in
    # Streamlit's own console output.
    logger.propagate = False
