"""Sanity checks on a UPF file, before any DFT is spent on it.

Written after the 2026-09-30 investigation (`project/u_14ve/UPF-INVESTIGATION.md`):
a freshly generated pseudopotential carried **two l=1 projectors with the same cutoff
radius** — the occupied 6P plus the zero-occupation `6P occ 0.00, e −0.25 Ry` state
from the ld1.x input — and ABACUS's LCAO path then did not solve the same Hamiltonian
as its PW path:

* `E_lcao − E_pw = −69.4 eV` (a finite basis cannot be *below* the plane-wave
  reference) with `ecutrho = 600` Ry, and
* `Assertion 'ic<nc' failed` in `module_base/matrix.h` — i.e. an out-of-bounds matrix
  access — as soon as the density grid was refined to 1200 Ry.

Removing that one projector restored the ordinary behaviour (+0.59 eV).  The older
`U.pbe-n-nc.14ve.UPF`, with five projectors (6S, 7S, 6P, 6D, 5F), never had the problem.

The checks here are pure text parsing, so they run in `check` and `run` *before*
anything is submitted, and they never touch the database.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["projector_summary", "upf_warnings"]

_BETA = re.compile(r"<PP_BETA\.(\d+)([^>]*)>")
_ATTR = re.compile(r'(\w+)="([^"]*)"')


def _attributes(block: str) -> dict[str, str]:
    return dict(_ATTR.findall(block))


def _values(text: str, tag: str) -> list[float] | None:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.S)
    if match is None:
        return None
    return [float(v) for v in match.group(1).split()]


def projector_summary(upf_path: str | Path) -> list[dict]:
    """``[{index, l, cutoff_index, label}]`` for every ``PP_BETA`` in the file."""
    text = Path(upf_path).read_text(errors="replace")
    out = []
    for match in _BETA.finditer(text):
        attrs = _attributes(match.group(2))
        out.append({
            "index": int(match.group(1)),
            "l": int(attrs.get("angular_momentum", -1)),
            "cutoff_index": int(attrs.get("cutoff_radius_index", -1)),
            "label": attrs.get("label", ""),
        })
    return sorted(out, key=lambda item: item["index"])


def upf_warnings(upf_path: str | Path) -> list[str]:
    """Human-readable problems with a UPF, phrased for a report or a CLI.

    Only what can be judged from the file itself, and only things that have been seen
    to cost a real run: a duplicate angular-momentum channel, and blocks whose size
    disagrees with the header (the file is then read as garbage).
    """
    path = Path(upf_path)
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return [f"{path.name}: cannot be read ({exc})"]

    header = re.search(r"<PP_HEADER[^>]*>", text)
    if header is None:
        # Not a UPF we can judge (a stub file, a different format).  Saying so on every
        # such file would be noise: ABACUS reports an unreadable pseudopotential by
        # itself, and there is nothing here to check.
        return []
    attrs = _attributes(header.group(0))
    problems: list[str] = []

    projectors = projector_summary(path)
    declared = attrs.get("number_of_proj")
    if declared is not None and int(declared) != len(projectors):
        problems.append(
            f"number_of_proj={declared} but {len(projectors)} PP_BETA blocks are "
            f"present — ABACUS reads the header, so the projectors are mis-paired"
        )

    dij = _values(text, "PP_DIJ")
    if dij is not None and declared is not None and len(dij) != int(declared) ** 2:
        problems.append(
            f"PP_DIJ holds {len(dij)} values, expected {int(declared) ** 2} "
            f"(number_of_proj^2) — the nonlocal strengths are mis-paired"
        )

    # The trigger itself: ABACUS's LCAO path assumes the PP_BETA blocks are grouped by
    # angular momentum (the two-center table is addressed by (l, zeta) while the
    # nonlocal operator is addressed by the order of the file), so an ungrouped file
    # silently pairs labels with the wrong radial functions.
    order = [projector["l"] for projector in projectors]
    if order != sorted(order):
        problems.append(
            f"the PP_BETA blocks are not grouped by angular momentum (order {order}): "
            f"ABACUS's LCAO path assumes they are, so it would use the wrong radial "
            f"function for several (l, zeta) pairs — its total energy lands tens of eV "
            f"away from the plane-wave one and can abort on a refined grid "
            f"(2026-09-30).  The file itself is legal (QE reads it), and renumbering the "
            f"blocks (permuting PP_DIJ the same way) is the same pseudopotential: "
            f"`edit_upf.py <in.UPF> <out.UPF> --sort-by-l` in project/u_14ve/.  The "
            f"ABACUS side of this is being fixed as well, after which such a file can "
            f"be used as it is"
        )

    # More projectors than atomic wavefunctions means at least one projector has no
    # reference state behind it — typically the `occ 0.00` state that ld1.x writes for
    # an "extra" channel.  That is how a file like `U.pbe-n-nc.UPF` ends up with its
    # PP_BETA blocks ordered l = 0,0,1,2,3,1.  The extra channel itself is legitimate
    # (once the blocks are grouped by l, that file reproduces the plane-wave energies to
    # 0.5 eV/atom, see UPF-INVESTIGATION.md); what it costs is the ordering, which is
    # reported separately above.
    wfc = attrs.get("number_of_wfc")
    if wfc is not None and len(projectors) > int(wfc):
        duplicates = _duplicate_channels(projectors, min_l=1)
        problems.append(
            f"{len(projectors)} projectors but only {wfc} atomic wavefunctions: at "
            f"least one projector has no reference state behind it"
            + (f" ({duplicates})" if duplicates else "")
            + " — check whether that projector is needed, and if it is, give it a "
            "distinct cutoff radius and keep the PP_BETA blocks grouped by l"
        )

    return [f"{path.name}: {problem}" for problem in problems]


def _duplicate_channels(projectors: list[dict], *, min_l: int = 0) -> str:
    """Describe projectors that share ``(l, cutoff_index)``, e.g. "l=1 (6P, 6P)"."""
    seen: dict[tuple[int, int], dict] = {}
    described: list[str] = []
    for projector in projectors:
        if projector["l"] < min_l:
            continue
        key = (projector["l"], projector["cutoff_index"])
        first = seen.get(key)
        if first is None:
            seen[key] = projector
            continue
        described.append(
            f"l={projector['l']} twice at cutoff index {projector['cutoff_index']}: "
            f"{first['label']!r} and {projector['label']!r}"
        )
    return "; ".join(described)
