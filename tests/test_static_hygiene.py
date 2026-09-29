"""Repository-wide static guarantees that are cheaper to check than to test.

Two classes of mistake kept slipping through unit tests, because neither shows up
until the exact line runs:

* **undefined names** — a refactor moved code around and left a name behind, so a
  rarely-taken branch raises ``NameError``.  Two real examples were found this way:
  ``self.ctx.grid[idx + 1]`` in the iterative grid strategy (``idx`` never existed)
  and a missing ``import json`` in ``cli/run.py``'s ``select`` path.
* **unused imports** — the same refactors leave imports pointing at code that moved
  away; a growing pile of them hides the one import that is actually missing.
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


def _pyflakes_messages(paths: list[Path]) -> list[str]:
    """Every pyflakes message for *paths*, as ``file:line:col: message``."""
    pyflakes_api = pytest.importorskip(
        "pyflakes.api", reason="pyflakes is not installed"
    )
    out, err = io.StringIO(), io.StringIO()
    from pyflakes.reporter import Reporter

    reporter = Reporter(out, err)
    for path in paths:
        pyflakes_api.checkPath(str(path), reporter)
    return (out.getvalue() + err.getvalue()).splitlines()


def _show(problems: list[str]) -> str:
    """Readable report: paths relative to the repository root."""
    return "\n".join(
        problem.replace(str(REPO_ROOT) + "/", "") for problem in problems
    )


def test_no_undefined_names_in_the_package():
    paths = _iter_sources("src/aiida_orbgen")
    paths += [path for path in EXTRA_ROOTS if path.is_file()]
    assert paths, "no sources found — the sweep is looking in the wrong place"

    problems = [
        message for message in _pyflakes_messages(paths)
        if "undefined name" in message
    ]
    assert not problems, f"undefined names found:\n{_show(problems)}"


def test_package_has_no_unused_imports():
    """An import nobody uses is either a leftover or a bug (a name never read).

    The package is clean as of this commit; keeping it that way is what makes the
    pyflakes "undefined name" check above continue to mean something.  ``siab_api.py``
    at the repository root is exempt: it is a wide API surface for manual use.
    """
    paths = _iter_sources("src/aiida_orbgen")
    problems = [
        message for message in _pyflakes_messages(paths)
        if "imported but unused" in message
    ]
    assert not problems, f"unused imports found:\n{_show(problems)}"


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
