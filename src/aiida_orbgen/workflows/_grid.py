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
    "build_multi_json_grid",
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

    ``10.0`` becomes ``10p0`` so the name stays a single path component.  Note
    the report layer deliberately uses a different spelling (``lmax4_rcut10``,
    see ``utils/report/orbitals.point_dir_name``); both are load-bearing for
    existing output trees, so they are *not* unified here.
    """
    return f"lmax{int(l_max)}_rcut{str(r_cut).replace('.', 'p')}"


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
) -> bool:
    """Whether the ``iterative`` loop should try another grid point.

    Stops (a) in dry-run mode, (b) as soon as an acceptable point was found,
    (c) once every candidate has been tried.  Extracted from the WorkChain so
    the stopping rule is testable without a daemon.
    """
    if dry_run:
        return False
    if best is not None:
        return False
    return n_done < len(grid)
