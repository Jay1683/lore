"""
Skill loading + registry.

Skills follow the official Anthropic SKILL.md format: a folder containing
a SKILL.md file with YAML frontmatter (name, description, and sometimes
extra fields like license) followed by a markdown instructions body.

Loading here is deliberately "shallow" -- we read metadata and the full
instructions text once at startup, but don't execute anything. Turning
those instructions into actual action (e.g. running the docx skill's
scripts) is the tool wrapper's job -- see tools.py -- and for now (Phase
5) that wrapper just surfaces the instructions rather than running them.
Real execution is Phase 6.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    description: str
    instructions: str  # markdown body, everything after the frontmatter
    path: Path  # the skill's own folder, in case scripts/ need it later


def _parse_skill_md(skill_md_path: Path) -> Skill:
    raw = skill_md_path.read_text(encoding="utf-8")

    if not raw.startswith("---"):
        raise ValueError(f"{skill_md_path} has no YAML frontmatter")

    # SKILL.md format: ---\n<yaml>\n---\n<markdown body>
    # Splitting on "---" with maxsplit=2 gives ['', '<yaml>', '<body>'].
    parts = raw.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"{skill_md_path} frontmatter is malformed")

    frontmatter = yaml.safe_load(parts[1])
    body = parts[2].strip()

    return Skill(
        name=frontmatter["name"],
        description=frontmatter["description"],
        instructions=body,
        path=skill_md_path.parent,
    )


def load_skills(skills_dir: Path) -> dict[str, Skill]:
    """
    Scans skills_dir for immediate subdirectories containing a SKILL.md,
    parses each, and returns them keyed by skill name.

    A skill folder that exists but has no SKILL.md, or a SKILL.md that
    fails to parse, is skipped with the exception left visible (loud
    failure at startup is better than a silently-missing skill).
    """
    skills: dict[str, Skill] = {}
    if not skills_dir.exists():
        return skills

    for entry in sorted(skills_dir.iterdir()):
        skill_md = entry / "SKILL.md"
        if entry.is_dir() and skill_md.exists():
            skill = _parse_skill_md(skill_md)
            skills[skill.name] = skill

    return skills
