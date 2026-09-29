"""Read a WorkChain's children into the records ``energies.py`` computes with.

The split between this module and :mod:`aiida_orbgen.workflows.energies` is
deliberate:

``energies.py``
    pure arithmetic — pairing PW/LCAO energies, ΔE, ΔE/atom, the tolerance
    verdict.  Testable with plain numbers.
``extract.py`` (here)
    the node-facing half — walk ``children_info``, decide which children are
    usable (``exit_status == 304`` counts as usable, see ``is_soft_success``),
    apply the STRU-orbital guard to LCAO children, read
    ``misc.total_energy``, and hand :class:`ChildEnergy` records over.

Both halves used to be one ~180-line step method inside
``OrbgenCalcWorkChain.extract_energies_step``, where the ΔE arithmetic could only
be exercised by running a WorkChain.  The WorkChain now only narrates the result
and turns it into an exit code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Iterable

from aiida_orbgen.workflows.energies import (
    ChildEnergy,
    is_soft_success,
    pair_energies,
)
from aiida_orbgen.workflows.siab import (
    expected_orbital_filename,
    find_abacus_child,
    verify_stru_orbital,
)

__all__ = [
    "ChildEnergies",
    "collect_child_energies",
]


@dataclass
class ChildEnergies:
    """What the usable children of one ``OrbgenCalcWorkChain`` amount to."""

    energies: dict[str, Any] = field(default_factory=dict)
    entries: list[ChildEnergy] = field(default_factory=list)
    n_lcao_total: int = 0
    n_lcao_valid: int = 0
    n_lcao_skipped: int = 0
    #: Every LCAO child was thrown away by the STRU-orbital guard, i.e. there is no
    #: LCAO data at all: the tolerance cannot be assessed, so it must not be
    #: reported as met (the caller turns this into ``ctx.tolerance_exceeded``).
    lcao_all_skipped: bool = False
    #: Human-facing lines for ``self.report`` — warnings and the skip summary.
    notes: list[str] = field(default_factory=list)

    @property
    def n_pw(self) -> int:
        return int(self.energies.get("n_pw", 0))

    @property
    def usable(self) -> bool:
        """Whether any child contributed an energy."""
        return bool(self.entries)


def collect_child_energies(
    children_info: Iterable[dict],
    *,
    report: Callable[[str], None] | None = None,
    verify_orbital: Callable[[Any, str], bool] | None = None,
) -> ChildEnergies:
    """Turn ``ctx.children_info`` into paired energies plus skip bookkeeping.

    Parameters
    ----------
    children_info : iterable of dict
        the WorkChain's own record of its children (``task``, ``basis``,
        ``n_atoms``, ``node``).
    report : callable, optional
        where warnings go (``self.report`` in a WorkChain).
    verify_orbital : callable, optional
        ``(abacus_calc, expected_filename) -> bool``; defaults to
        :func:`~aiida_orbgen.workflows.siab.verify_stru_orbital`, which is what a
        WorkChain wants.  Injected so the guard can be tested without a real STRU.

    Returns
    -------
    ChildEnergies
        Paired energies (the dict :func:`pair_energies` returns), the records they
        came from, and the LCAO skip counters the caller needs for its exit code.
    """
    children_info = list(children_info)
    if verify_orbital is None:
        verify_orbital = partial(verify_stru_orbital, report=report)

    result = ChildEnergies()
    n_atoms_by_folder = {
        str(item.get("task", "")): int(item.get("n_atoms", 1))
        for item in children_info
    }

    for item in children_info:
        node = item["node"]
        basis = str(item.get("basis", ""))
        if basis == "lcao_nsw":
            result.n_lcao_total += 1

        # 304 counts as a success here (energies.py:is_soft_success): the child ran
        # the whole SCF, only the LCAO accuracy verdict failed.  A genuine failure
        # is not reported here — `finalize` counts those.
        if not is_soft_success(node.is_finished_ok, node.exit_status):
            continue

        # For LCAO children, check that the orbital file the STRU references really
        # is the one AiiDA injected: the aiida-abacus STRU generator can point at
        # the wrong file, and ABACUS then silently returns a wrong energy.  Fake
        # data must be skipped rather than averaged in.
        if basis == "lcao_nsw":
            abacus_calc = find_abacus_child(node)
            expected = expected_orbital_filename(abacus_calc)
            if expected and not verify_orbital(abacus_calc, expected):
                result.n_lcao_skipped += 1
                continue

        misc = node.outputs.misc if hasattr(node.outputs, "misc") else None
        if misc is None:
            result.notes.append(f"  WARNING: node {node.pk} has no misc output")
            continue
        total_energy = misc.get_dict().get("total_energy")
        if total_energy is None:
            result.notes.append(
                f"  WARNING: node {node.pk} misc has no total_energy"
            )
            continue

        folder = str(item.get("task", ""))
        result.entries.append(ChildEnergy(
            folder=folder,
            # "lcao_nsw" is the workflow's name for the LCAO child; the records use
            # the reporting name "lcao" (as the reports and `energies` dict do).
            basis="pw" if basis == "pw" else "lcao",
            energy=float(total_energy),
            n_atoms=n_atoms_by_folder.get(folder, 1),
        ))

    result.energies = pair_energies(result.entries)
    result.n_lcao_valid = sum(1 for entry in result.entries if entry.basis == "lcao")

    if result.n_lcao_total and result.n_lcao_valid == 0:
        result.lcao_all_skipped = True
        result.notes.append(
            f"  ✗ all {result.n_lcao_total} lcao children were skipped because of a "
            f"wrong STRU orbital, so lcao_nsw data is unusable "
            f"(aiida-abacus STRU generation is buggy). "
            f"pw={result.n_pw} is still valid, but lcao:nsw quality cannot be "
            f"assessed."
        )
    elif result.n_lcao_skipped:
        result.notes.append(
            f"  ⚠ {result.n_lcao_skipped}/{result.n_lcao_total} lcao children skipped "
            f"due to a wrong STRU orbital"
        )

    return result
