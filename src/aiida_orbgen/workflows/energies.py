"""ΔE extraction, the tolerance verdict and the 304 rule — one implementation.

Three copies of this logic used to exist: the inline block inside
``OrbgenCalcWorkChain.extract_energies_step``, and the near-identical
soft-success/best-entry code in the grid search's ``iterate_step`` and
``collect_exhaustive_step`` (the "exit 304 is a soft success" rule alone was
written three times, with slightly different consequences).

Everything here is pure: no AiiDA nodes, no filesystem. The WorkChains keep the
provenance-facing parts (reading ``misc.total_energy``, the STRU-orbital guard,
``self.report`` narration).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    "ChildEnergy",
    "SOFT_SUCCESS_EXIT_STATUS",
    "is_soft_success",
    "pair_energies",
    "tolerance_verdict",
    "evaluate_energies",
    "best_of",
    "describe_deltas",
]

#: ``AbacusBaseWorkChain`` exit status of a child that ran to completion
#: (SCF + energy extraction) but whose LCAO accuracy missed the tolerance.
SOFT_SUCCESS_EXIT_STATUS = 304

MEV_PER_EV = 1000.0


@dataclass(frozen=True)
class ChildEnergy:
    """One child calculation's total energy, ready to be paired."""

    folder: str
    basis: str            # "pw" | "lcao"
    energy: float         # eV, as ``misc.total_energy`` reports it
    n_atoms: int = 1


def is_soft_success(
    finished_ok: bool,
    exit_status: int | None,
    has_energies: bool = True,
) -> bool:
    """Whether a child counts as usable data.

    ``exit_status == 304`` (``WARNING_TOLERANCE_EXCEEDED``) means the child ran
    the whole SCF and the energy extraction; only the "LCAO is close enough to
    PW" verdict failed. For a grid search that is *information*, not an error, so
    its energies must still be extracted.
    """
    if finished_ok:
        return True
    return exit_status == SOFT_SUCCESS_EXIT_STATUS and has_energies


def pair_energies(energies: Iterable[ChildEnergy]) -> dict[str, Any]:
    """Pair PW and LCAO energies per geometry and compute ΔE.

    Returns the dict the WorkChains store in ``ctx.energies``:

    ``energies``
        ``{basis: [{folder, energy, pert, n_atoms}, ...]}``
    ``delta_E_per_struct``
        one entry per geometry present in *both* bases, in eV and eV/atom
    ``delta_E_max_eV`` / ``delta_E_max_per_atom_eV``
        the maxima over those geometries (0.0 when nothing could be paired)
    ``n_pw`` / ``n_lcao``
        how many energies each basis contributed
    """
    by_basis: dict[str, list[dict[str, Any]]] = {}
    for item in energies:
        by_basis.setdefault(item.basis, []).append({
            "folder": item.folder,
            "energy": float(item.energy),
            "pert": None,
            "n_atoms": int(item.n_atoms),
        })

    delta_per_struct: list[dict[str, Any]] = []
    pw_by_folder = {entry["folder"]: entry for entry in by_basis.get("pw", [])}
    for e_lcao in by_basis.get("lcao", []):
        e_pw = pw_by_folder.get(e_lcao["folder"])
        if e_pw is None:
            continue
        n_atoms = max(1, int(e_lcao.get("n_atoms", 1)))
        d_e_total = abs(e_lcao["energy"] - e_pw["energy"])
        delta_per_struct.append({
            "folder": e_lcao["folder"],
            "n_atoms": n_atoms,
            # PW energies are the reference, so a missing PW energy means the
            # geometry is simply not comparable (handled by the lookup above).
            "E_pw": e_pw["energy"],
            "E_lcao_nsw": e_lcao["energy"],
            "dE": d_e_total,
            "dE_per_atom": d_e_total / n_atoms,
        })

    delta_max = max((entry["dE"] for entry in delta_per_struct), default=0.0)
    delta_max_per_atom = max(
        (entry["dE_per_atom"] for entry in delta_per_struct), default=0.0
    )
    return {
        "energies": by_basis,
        "delta_E_per_struct": delta_per_struct,
        "delta_E_max_eV": float(delta_max),
        "delta_E_max_meV": float(delta_max * MEV_PER_EV),
        "delta_E_max_per_atom_eV": float(delta_max_per_atom),
        "delta_E_max_per_atom_meV": float(delta_max_per_atom * MEV_PER_EV),
        "n_pw": len(by_basis.get("pw", [])),
        "n_lcao": len(by_basis.get("lcao", [])),
    }


def tolerance_verdict(
    delta_per_atom_meV: float,
    tolerance_meV: float,
) -> tuple[bool, str]:
    """``(acceptable, message)`` for ΔE/atom against the tolerance.

    The standard is 0.1 kcal/mol/**atom** ≈ 4.2 meV/atom (chemical accuracy), so
    the comparison has to use the per-atom number — the per-system one is the
    dimer total and is roughly twice as large.
    """
    if delta_per_atom_meV > tolerance_meV:
        return False, (
            f"✗ tolerance EXCEEDED: ΔE/atom_max={delta_per_atom_meV:.3f} meV "
            f"> tolerance_meV={tolerance_meV:.3f} meV "
            f"(0.1 kcal/mol/atom = 4.2 meV) -> lcao:nsw NOT acceptable"
        )
    return True, (
        f"✓ tolerance OK: ΔE/atom_max={delta_per_atom_meV:.3f} meV "
        f"<= tolerance_meV={tolerance_meV:.3f} meV -> lcao:nsw is acceptable"
    )


def evaluate_energies(energies_dict: dict, tolerance_meV: float) -> dict[str, Any]:
    """Verdict for one grid point, from its ``OrbgenCalcWorkChain`` outputs.

    ``energies_dict`` is the child workchain's ``energies`` output. Used by both
    grid-search strategies so they cannot disagree; ``delta_per_atom_meV`` is
    ``inf`` when the child has no usable ΔE, which makes the point unacceptable
    rather than silently perfect.
    """
    delta_per_atom_meV = float(
        energies_dict.get("delta_E_max_per_atom_meV", float("inf"))
    )
    tolerance_ok, message = tolerance_verdict(delta_per_atom_meV, tolerance_meV)
    return {
        "delta_per_atom_meV": delta_per_atom_meV,
        "tolerance_ok": tolerance_ok,
        "message": message,
    }


def best_of(entries: Sequence[dict], tolerance_meV: float | None = None) -> dict | None:
    """The first acceptable grid entry, or ``None``.

    The grid is ordered by ``(l_max, r_cut)``, so "first acceptable" is also
    "the cheapest acceptable basis" — the same rule the iterative strategy uses
    when it stops early.
    """
    for entry in entries:
        if entry.get("tolerance_ok"):
            return entry
        if tolerance_meV is not None:
            delta = entry.get("delta_per_atom_meV")
            if delta is not None and float(delta) <= float(tolerance_meV):
                return entry
    return None


def describe_deltas(energies: dict) -> list[str]:
    """The ΔE lines to report after an extraction (per system and per geometry).

    Pure formatting of what :func:`pair_energies` returned, so the WorkChain
    reports the same numbers the ``energies`` output node holds.
    """
    lines = [
        f"  ΔE_max (per system)  = {float(energies.get('delta_E_max_meV', 0.0)):.3f} meV "
        f"({float(energies.get('delta_E_max_eV', 0.0)):.6f} eV)",
        f"  ΔE_max (per atom)    = "
        f"{float(energies.get('delta_E_max_per_atom_meV', 0.0)):.3f} meV "
        f"({float(energies.get('delta_E_max_per_atom_eV', 0.0)):.6f} eV)",
    ]
    for entry in energies.get("delta_E_per_struct", []):
        lines.append(
            f"    {entry['folder']:30s}  N={int(entry['n_atoms']):2d}  "
            f"E_pw={float(entry['E_pw']):.6f}  "
            f"E_lcao_nsw={float(entry['E_lcao_nsw']):.6f}  "
            f"dE={float(entry['dE']) * MEV_PER_EV:.3f} meV  "
            f"dE/atom={float(entry['dE_per_atom']) * MEV_PER_EV:.3f} meV"
        )
    return lines
