"""VIII - README requirements."""

import re

import pytest

from .helpers import PROJECT_ROOT

README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
HEADINGS = [line.lstrip("#").strip().lower()
            for line in README.splitlines() if line.startswith("#")]


def test_first_line_is_the_italic_42_credit() -> None:
    """Line 1: *This project has been created as part of the 42 ...*."""
    first = README.splitlines()[0]
    assert re.fullmatch(r"\*This project has been created as part of the 42 "
                        r"curriculum by [a-z0-9_-]+(, [a-z0-9_-]+)*\.\*",
                        first), first


@pytest.mark.parametrize("section", [
    "description", "instructions", "resources",
    "system architecture", "chunking strategy", "retrieval method",
    "performance analysis", "design decisions", "challenges faced",
    "example usage"])
def test_required_section(section: str) -> None:
    """Every section required by the subject is present."""
    assert any(heading.startswith(section) for heading in HEADINGS), section


def test_ai_usage_is_described() -> None:
    """Resources explain how AI was used, for which tasks."""
    assert re.search(r"use of ai", README, re.IGNORECASE)


def test_readme_is_written_in_english() -> None:
    """No Cyrillic (or other non-Latin script) text."""
    assert not re.search(r"[\u0400-\u04ff]", README)


@pytest.mark.parametrize("command", ["index", "search", "search_dataset",
                                     "answer", "answer_dataset", "evaluate"])
def test_examples_cover_every_command(command: str) -> None:
    """Example usage shows how to run each command."""
    assert re.search(rf"python -m src(?: --\w+ \S+)* {command}\b", README)


@pytest.mark.parametrize("bonus", ["semantic", "hybrid", "incremental",
                                   "cach", "http"])
def test_bonuses_are_documented(bonus: str) -> None:
    """Each bonus has its own README section."""
    assert any(bonus in heading for heading in HEADINGS), bonus
