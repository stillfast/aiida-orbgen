"""Repository-wide static guarantees that are cheaper to check than to test.

Two classes of mistake kept slipping through unit tests, because neither shows up
until the exact line runs:

* **undefined names** — a refactor moved code around and left a name behind, so a
  rarely-taken branch raises ``NameError``.  Two real examples were found this way:
  ``self.ctx.grid[idx + 1]`` in the iterative grid strategy (``idx`` never existed)
  and a missing ``import json`` in ``cli/run.py``'s ``select`` path.
* **Chinese text in code** — the project is English-only, and a stray comment is
  invisible until someone greps for it.

Both are checked with the installed tooling instead of hand-written heuristics, and
skip cleanly when that tooling is absent so the suite stays runnable anywhere.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = REPO_ROOT / "src" / "aiida_orbgen"

CJK = re.compile(r"[\u4e00-\u9fff]")

# `siab_api.py` sits at the repository root (it is an AiiDA verdi entry point) and
# is therefore easy to forget in a package-only sweep.
EXTRA_ROOTS = [REPO_ROOT / "siab_api.py"]

# Vendored / generated / archived material: not ours to police.
EXCLUDED_PARTS = {"__pycache__", "_removed-20260929"}

# Text files that are "code" for the purposes of this file: anything a maintainer
# reads as source, including the YAML/JSON presets and the docs.
TEXT_SUFFIXES = (".py", ".yml", ".yaml", ".json", ".toml", ".md")


def _iter_sources(*subdirs: str, suffixes: tuple[str, ...] = (".py",)) -> list[Path]:
    found: list[Path] = []
    for subdir in subdirs:
        root = REPO_ROOT / subdir
        if not root.is_dir():
            continue
        found.extend(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix in suffixes
            and not EXCLUDED_PARTS & set(path.relative_to(REPO_ROOT).parts)
        )
    return sorted(found)


def _pyflakes_undefined_names(paths: list[Path]) -> list[str]:
    """Pyflakes messages that mention an undefined name, as ``file:line: msg``."""
    pyflakes_api = pytest.importorskip(
        "pyflakes.api", reason="pyflakes is not installed"
    )
    out, err = io.StringIO(), io.StringIO()
    from pyflakes.reporter import Reporter

    reporter = Reporter(out, err)
    for path in paths:
        pyflakes_api.checkPath(str(path), reporter)
    messages = (out.getvalue() + err.getvalue()).splitlines()
    return [line for line in messages if "undefined name" in line]


def test_no_undefined_names_in_the_package():
    paths = _iter_sources("src/aiida_orbgen")
    paths += [path for path in EXTRA_ROOTS if path.is_file()]
    assert paths, "no sources found — the sweep is looking in the wrong place"

    problems = _pyflakes_undefined_names(paths)
    # Make paths readable: /abs/path.py:12:5: undefined name 'x'
    shown = "\n".join(
        problem.replace(str(REPO_ROOT) + "/", "") for problem in problems
    )
    assert not problems, f"undefined names found:\n{shown}"


def test_sources_contain_no_chinese():
    """The code base is English-only; a Chinese comment is a review finding."""
    offenders: list[str] = []
    paths = _iter_sources("src", "tests", suffixes=TEXT_SUFFIXES)
    paths += [REPO_ROOT / name for name in ("README.md", "REFACTOR-20260929.md")]
    paths += EXTRA_ROOTS
    for path in paths:
        if not path.is_file():
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if CJK.search(line):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()[:80]}"
                )
    assert not offenders, "Chinese text found:\n" + "\n".join(offenders)
