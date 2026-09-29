"""Grid bookkeeping for :class:`aiida_orbgen.workflows.batch.OrbgenGridSearchWorkChain`.

A grid point used to be a bare ``(l_max, r_cut, siab_json)`` triple stored in
``ctx.grid``.  Two call sites unpacked it into **two** names
(``l_max, r_cut = self.ctx.grid[0]``), which raises ``ValueError: too many
values to unpack`` -- so the ``iterative`` strategy, which is the WorkChain's
own default, failed on its very first submission.  Every CLI run passed
``exhaustive`` explicitly and therefore never hit it.

:class:`GridEntry` carries the same three values under names, so the mistake is
no longer expressible.  The pure helpers below are unit-tested in
``tests/test_workflows.py`` (the WorkChain itself needs a daemon, these do not).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "GridEntry",
    "build_cartesian_grid",
    "build_explicit_grid",
    "build_multi_json_grid",
    "cap_grid",
    "has_pending_iterative",
    "work_dir_name",
]


@dataclass(frozen=True)
class GridEntry:
    """One ``(l_max, r_cut)`` candidate together with its SIAB config node."""

    l_max: int
    r_cut: float
    siab_json: Any = None
    index: int = 0

    @property
    def point(self) -> tuple[int, float]:
        """``(l_max, r_cut)`` -- what the report layer and the extras use."""
        return (self.l_max, self.r_cut)

    @property
    def label(self) -> str:
        return f"l_max={self.l_max}, r_cut={self.r_cut:g}"

    def __str__(self) -> str:  # keeps the old log lines readable
        return f"({self.l_max}, {self.r_cut:g})"


def work_dir_name(l_max: int, r_cut: float) -> str:
    """Per-grid-point sub-directory of ``OrbgenGridSearchWorkChain``.

    Delegates to :func:`aiida_orbgen.interfaces.nsw.point_dir_name` so the
    workflow and the report layer agree on ``lmax4_rcut10``; they used to
    disagree (``lmax4_rcut10p0`` vs ``lmax4_rcut10``) for the same point.
    """
    from aiida_orbgen.interfaces.nsw import point_dir_name

    return point_dir_name(l_max, r_cut)


def build_cartesian_grid(
    l_max_candidates: Iterable[int],
    r_cut_candidates: Iterable[float],
    siab_json: Any,
) -> list[GridEntry]:
    """``l_max × r_cut`` product, ordered by ``l_max`` then ``r_cut``.

    The order matters: the ``iterative`` strategy walks the list and
    ``exhaustive`` takes the first acceptable entry, so a sorted grid is what
    makes "the cheapest acceptable basis" well defined.
    """
    entries: list[GridEntry] = []
    for l_max in sorted({int(value) for value in l_max_candidates}):
        for r_cut in sorted({float(value) for value in r_cut_candidates}):
            entries.append(
                GridEntry(l_max=l_max, r_cut=r_cut,
                          siab_json=siab_json, index=len(entries))
            )
    return entries


def build_explicit_grid(pairs: Sequence[Sequence[Any]], siab_json: Any) -> list[GridEntry]:
    """Grid from an explicit ``[[l_max, r_cut], ...]`` list (not a product).

    Needed to fill in a gap in a grid that was already scanned, e.g. only
    ``[(4, 11.0), (4, 12.0)]`` after ``r_cut <= 10`` had been evaluated.
    """
    entries: list[GridEntry] = []
    for pair in pairs:
        values = list(pair)
        if len(values) != 2:
            raise ValueError(
                f"candidates entries must be [l_max, r_cut] pairs, got {pair!r}"
            )
        entries.append(
            GridEntry(l_max=int(values[0]), r_cut=float(values[1]),
                      siab_json=siab_json, index=len(entries))
        )
    return entries


def build_multi_json_grid(pairs: Sequence[tuple[int, float, Any]]) -> list[GridEntry]:
    """One entry per ``orbgen.json`` (multi-JSON mode), order preserved."""
    return [
        GridEntry(l_max=int(l_max), r_cut=float(r_cut),
                  siab_json=node, index=index)
        for index, (l_max, r_cut, node) in enumerate(pairs)
    ]


def has_pending_iterative(
    grid: Sequence[GridEntry],
    n_done: int,
    best: Any = None,
    dry_run: bool = False,
    stop_on_first_valid: bool = True,
) -> bool:
    """Whether the ``iterative`` loop should try another grid point.

    Stops (a) in dry-run mode, (b) as soon as an acceptable point was found --
    unless ``stop_on_first_valid`` is False, which is how "run the whole grid,
    then choose" is expressed --, (c) once every candidate has been tried.
    Extracted from the WorkChain so the stopping rule is testable without a
    daemon.
    """
    if dry_run:
        return False
    if best is not None and stop_on_first_valid:
        return False
    return n_done < len(grid)


def cap_grid(
    grid: Sequence[GridEntry],
    max_l_max: int | None = None,
    max_r_cut: float | None = None,
) -> list[GridEntry]:
    """Apply the optional ``max_l_max`` / ``max_r_cut`` caps, keeping indices.

    ``with_default_abacus`` has always forwarded these two keys, but only
    ``advanced.py`` ever honoured them.
    """
    capped = [
        entry for entry in grid
        if (max_l_max is None or entry.l_max <= int(max_l_max))
        and (max_r_cut is None or entry.r_cut <= float(max_r_cut))
    ]
    return [
        GridEntry(l_max=e.l_max, r_cut=e.r_cut, siab_json=e.siab_json, index=i)
        for i, e in enumerate(capped)
    ]
