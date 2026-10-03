"""Shared plumbing of the two value-selection WorkChains.

:class:`~aiida_orbgen.workflows.ecutwfc.OrbgenEcutwfcWorkChain` and
:class:`~aiida_orbgen.workflows.basis.OrbgenBasisScanWorkChain` both submit *many*
children from one ``WorkChain`` call, so neither can use
``OrbgenCalcWorkChain.submit_children`` -- that one hard-codes "basis x dft_entry" of
a single SIAB tree.  What they can share is everything around the submission:

* reading the scheduler options and the ``parameters.input`` overrides out of
  ``abacus.json`` (:func:`child_options`);
* building and filing one child (:func:`submit_child`, :func:`record_child`);
* a geometry name that survives a change of ``(r_cut, l_max, ecutjy)``
  (:func:`geometry_key`) -- SIAB names its folders ``U-dimer-2.80-11au``, so the
  *same* geometry has a different folder name in every candidate's tree and the two
  energies could not be paired;
* turning the collected energies back into the per-candidate table the decision
  functions of :mod:`aiida_orbgen.workflows.ladder` expect
  (:func:`geometries_from_entries`);
* the ``results`` / ``pseudo_family`` outputs every run has to write
  (:func:`write_child_results`).

The children are tagged with a synthetic task name ``"<label>::<geometry>"``: the
record keeps the candidate the child belongs to and the geometry it ran, which is
exactly what ``children_info`` entries are keyed by downstream
(:func:`~aiida_orbgen.workflows.extract.collect_child_energies` reads ``task``,
``basis`` and ``n_atoms``).
"""

from __future__ import annotations

import copy
import os
from typing import Any, Iterable, Mapping

from aiida.engine import append_
from aiida.plugins import WorkflowFactory

from aiida_orbgen.static.defaults import (
    DEFAULT_CODE_LABEL,
    DEFAULT_MAX_MEMORY_KB,
    DEFAULT_NUM_MPI,
    DEFAULT_QUEUE_NAME,
    DEFAULT_WALLCLOCK_SECONDS,
)
from aiida_orbgen.workflows.energies import is_soft_success
from aiida_orbgen.workflows.results import create_family_label, create_final_results
from aiida_orbgen.workflows.siab import build_abacus_child_inputs, n_atoms_from_stru

__all__ = [
    "cell_edges_bohr",
    "child_options",
    "geometry_key",
    "geometries_from_entries",
    "n_primitive_functions",
    "rcut_fits_cell",
    "siab_cache_hint",
    "siab_tree_problem",
    "record_child",
    "seconds_of",
    "split_task",
    "submit_child",
    "synthetic_task",
    "write_child_results",
]

#: Separator of the synthetic task name ``"<label>::<geometry>"``.
_TASK_SEPARATOR = "::"


# ---------------------------------------------------------------------------
#  abacus.json -> child options
# ---------------------------------------------------------------------------
def child_options(workchain, *, family_label: str | None = None) -> dict[str, Any]:
    """The scheduler/AiiDA options every child of ``workchain`` is submitted with.

    Same precedence as ``OrbgenCalcWorkChain.submit_children``: the ``code_label`` and
    ``family_label`` inputs win over ``abacus.json``.  ``parameters`` is the *whole*
    ``abacus.parameters`` mapping (not only its ``input`` sub-dict), so a preset that
    carries sibling keys -- ``pw.yml`` has ``max_iterations`` -- keeps them.
    """
    abacus = workchain.ctx.abacus_cfg.get("abacus", {}) or {}
    metadata = (abacus.get("metadata", {}) or {}).get("options", {}) or {}
    num_mpi = metadata.get("num_mpiprocs_per_machine", DEFAULT_NUM_MPI)
    if num_mpi is None:
        num_mpi = metadata.get("num_mpi", DEFAULT_NUM_MPI)
    if "code_label" in workchain.inputs:
        code_label = str(workchain.inputs.code_label.value)
    else:
        code_label = str(abacus.get("code", DEFAULT_CODE_LABEL))
    return {
        "code_label": code_label,
        "family_label": (
            str(family_label) if family_label is not None
            else workchain._effective_family_label()
        ),
        "queue_name": str(metadata.get("queue_name", DEFAULT_QUEUE_NAME)),
        "num_mpi": int(num_mpi),
        "wallclock": int(metadata.get("max_wallclock_seconds", DEFAULT_WALLCLOCK_SECONDS)),
        "max_memory_kb": int(metadata.get("max_memory_kb", DEFAULT_MAX_MEMORY_KB)),
        "parameters": copy.deepcopy(abacus.get("parameters", {}) or {}),
    }


def with_input_overrides(
    options: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """``options["parameters"]`` with ``overrides`` merged into its ``input`` dict.

    ``build_abacus_child_inputs`` lets ``parameters["input"]`` win over the INPUT SIAB
    generated, so the merge has to happen on the ``input`` sub-dict: replacing the
    mapping outright would silently drop every other override of ``abacus.json``.
    """
    parameters = copy.deepcopy(dict(options.get("parameters") or {}))
    if overrides:
        merged = dict(parameters.get("input") or {})
        merged.update(dict(overrides))
        parameters["input"] = merged
    return parameters


# ---------------------------------------------------------------------------
#  naming: the candidate and the geometry of one child
# ---------------------------------------------------------------------------
def geometry_key(dft_entry: Mapping[str, Any]) -> str:
    """A name for one reference geometry that does not mention the basis parameters.

    SIAB writes ``U-dimer-2.80-11au`` / ``U-monomer-11au``, so pairing the energies of
    two different ``(r_cut, l_max, ecutjy)`` candidates by folder name would fail for
    every geometry.  ``proto`` ("dimer"/"monomer") plus ``pert`` identify the geometry
    itself, and the monomer name must contain "monomer" because that is how
    ``ladder._atomization_difference`` finds the reference monomer.
    """
    proto = str(dft_entry.get("proto") or "").strip()
    pert = dft_entry.get("pert")
    if proto == "monomer":
        return "monomer"
    if pert is None:
        return str(dft_entry.get("folder") or "geometry")
    name = f"{proto or 'geometry'}-{float(pert):g}"
    return name


def synthetic_task(label: str, geometry: str) -> str:
    """``"r11_l4_j125::dimer-2.8"`` -- the ``task`` of one child of a scan."""
    return f"{label}{_TASK_SEPARATOR}{geometry}"


def split_task(task: str) -> tuple[str, str]:
    """Inverse of :func:`synthetic_task`; a task without a separator is all label."""
    label, separator, geometry = str(task).partition(_TASK_SEPARATOR)
    if not separator:
        return label, label
    return label, geometry


# ---------------------------------------------------------------------------
#  submission
# ---------------------------------------------------------------------------
def submit_child(
    workchain,
    *,
    dft_entry: Mapping[str, Any],
    basis: str,
    options: Mapping[str, Any],
    parameters: Mapping[str, Any] | None = None,
) -> tuple[Any, int]:
    """Submit one ``abacus.base`` child; returns ``(node, n_atoms)``.

    Raises whatever ``build_abacus_child_inputs`` raises (a missing STRU, an
    unreadable INPUT): the caller decides whether that is fatal for the candidate.
    """
    child_inputs = build_abacus_child_inputs(
        dict(dft_entry),
        basis=basis,
        code_label=options["code_label"],
        family_label=options["family_label"],
        parameters=dict(parameters or {}),
        queue_name=options["queue_name"],
        num_mpi=options["num_mpi"],
        wallclock=options["wallclock"],
        max_memory_kb=options["max_memory_kb"],
    )
    child_cls = WorkflowFactory(workchain._child_workchain_entry_point)
    node = workchain.submit(child_cls, **child_inputs)
    n_atoms = n_atoms_from_stru(dft_entry.get("stru"), report=workchain.report)
    return node, n_atoms


def record_child(
    workchain,
    *,
    node,
    label: str,
    geometry: str,
    basis: str,
    n_atoms: int,
) -> dict[str, Any]:
    """File a submitted child in ``ctx.children_info`` and wait for it.

    ``collect_child_energies`` reads ``task`` / ``basis`` / ``n_atoms``; the extras
    carry the same information for anyone reading the provenance directly.
    """
    entry = {
        "task": synthetic_task(label, geometry),
        "basis": basis,
        "node": node,
        "n_atoms": int(n_atoms),
    }
    workchain.ctx.children_info.append(entry)
    workchain.to_context(children=append_(node))
    extras = {
        "task": geometry,
        "label": label,
        "basis": basis,
        "n_atoms": str(n_atoms),
    }
    try:
        node.base.extras.set_many(extras)
    except Exception as exc:  # noqa: BLE001 -- provenance extras are cosmetic
        workchain.report(f"  WARNING: could not tag node {node.pk}: {exc}")
    return entry


def seconds_of(node) -> float | None:
    """Wall-clock seconds a finished child occupied, from its own timestamps.

    Used as the cost proxy of a candidate: it is what the queue actually billed, and
    it is available without reading any output file.
    """
    try:
        return float((node.mtime - node.ctime).total_seconds())
    except Exception:  # noqa: BLE001 -- a node without timestamps is not fatal
        return None


# ---------------------------------------------------------------------------
#  collection
# ---------------------------------------------------------------------------
def geometries_from_entries(
    entries: Iterable[Any],
    *,
    pw_reference: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{label: {geometry: {"e_nsw"/"e_pw": eV, "n_atoms": N}}}`` from child records.

    ``entries`` are the records of
    :class:`~aiida_orbgen.workflows.extract.ChildEnergies` (``folder``, ``basis``,
    ``energy``, ``n_atoms``).  ``pw_reference`` supplies the PW energies that were
    computed once for the whole scan (``{geometry: {"energy": eV, "n_atoms": N}}``),
    so a candidate only has to bring its own LCAO numbers.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for entry in entries:
        label, geometry = split_task(entry.folder)
        slot = out.setdefault(label, {}).setdefault(
            geometry, {"n_atoms": int(entry.n_atoms)}
        )
        if entry.basis == "pw":
            slot["e_pw"] = float(entry.energy)
            slot["n_atoms_pw"] = int(entry.n_atoms)
        else:
            slot["e_nsw"] = float(entry.energy)
    if pw_reference:
        for label, geometries in out.items():
            for geometry, slot in geometries.items():
                if "e_pw" in slot:
                    continue
                reference = pw_reference.get(geometry)
                if reference is None:
                    continue
                slot["e_pw"] = float(
                    reference.get("energy", reference.get("e_pw"))
                )
                slot["n_atoms"] = int(
                    reference.get("n_atoms") or slot.get("n_atoms") or 1
                )
                slot["pw_from_reference"] = True
    return out


def siab_tree_problem(info: Mapping[str, Any] | None) -> str | None:
    """``None`` when a SIAB result's files are on disk, else why they are not.

    ``run_siab_pipeline`` is a ``calcfunction``, so AiiDA's cache hands back the *stored
    Dict* of an earlier run -- with the paths of that run's output directory.  If those
    files were deleted or the directory was reused, every child built from that Dict
    fails with a bare ``FileNotFoundError`` on an INPUT/STRU path, which says nothing
    about the cause.  A scan generates one tree per candidate and can check all of them
    for the price of a few ``stat`` calls.
    """
    if not info:
        return "the SIAB result is empty"
    orb_path = info.get("orb_path")
    if orb_path and not os.path.isfile(str(orb_path)):
        return f"the primitive orbital is gone: {orb_path}"
    dft = info.get("dft") or []
    if not dft:
        return "the SIAB result has no DFT job"
    for entry in dft:
        stru = entry.get("stru")
        if stru and not os.path.isfile(str(stru)):
            return f"the generated STRU is gone: {stru}"
        inp = entry.get("input")
        if inp and not os.path.isfile(str(inp)):
            return f"the generated INPUT is gone: {inp}"
    return None


def siab_cache_hint(output_dir: str) -> str:
    """What to do when a SIAB tree of a cached result has disappeared."""
    return (
        "the SIAB pipeline result came from AiiDA's cache but the files of "
        f"{output_dir} are gone: use a fresh output_dir (or remove the cached node), "
        "or the scan will keep building children from those paths"
    )


def cell_edges_bohr(parsed_stru: Mapping[str, Any] | None) -> list[float]:
    """The three cell edges of a parsed STRU, in Bohr.

    SIAB writes ``LATTICE_CONSTANT`` (Bohr) times dimensionless ``LATTICE_VECTORS``, so
    the edge length is ``lattice_constant * |vector|``.  Going through ASE instead would
    mean dividing by ``interfaces.stru.BOHR_TO_ANG``, which despite its name holds the
    number of Bohr per Angstrom -- a factor 3.6 away from what a reader expects.  The
    r_cut check below is in Bohr because ``r_cut`` is, and it must not be off by 3.6.
    """
    if not parsed_stru:
        return []
    constant = parsed_stru.get("lattice_constant")
    vectors = parsed_stru.get("lattice_vectors")
    if constant is None or not vectors:
        return []
    edges = []
    for vector in vectors:
        try:
            norm = sum(float(component) ** 2 for component in vector) ** 0.5
        except (TypeError, ValueError):
            return []
        edges.append(float(constant) * norm)
    return edges


def rcut_fits_cell(r_cut: float, edges: Iterable[float], *, whats: str = "") -> str | None:
    """``None`` when ``r_cut <= half the smallest edge``, else the complaint.

    Past half the edge the two-centre tables describe neighbours that cannot exist in
    the cell, so the basis is silently a different (worse) one: better to refuse.
    """
    edges = [float(edge) for edge in edges if edge]
    if not edges:
        return None
    half = min(edges) / 2.0
    if float(r_cut) <= half + 1.0e-6:
        return None
    where = f" of {whats}" if whats else ""
    return (
        f"r_cut={float(r_cut):g} au exceeds half the smallest cell edge "
        f"({min(edges):.3f} au -> {half:.3f} au){where}"
    )


def n_primitive_functions(r_cut: float, ecutjy: float, l_max: int) -> int | None:
    """Primitive functions per atom of one candidate -- the cost proxy.

    ``nchi`` grows with ``r_cut``, ``ecutjy`` and ``l_max``, and the two-centre table
    cost of a LCAO run grows roughly with ``nchi**2``, so it ranks candidates the same
    way the run time does without waiting for a run (``ladder.basis_table`` uses it
    only when a candidate has no measured ``seconds`` yet).
    """
    from aiida_orbgen.interfaces.nsw import compute_nbes_per_l

    try:
        nbes = compute_nbes_per_l(float(r_cut), float(ecutjy), int(l_max))
    except Exception:  # noqa: BLE001 -- a proxy may legitimately be unavailable
        return None
    return int(sum(int(n) * (2 * l + 1) for l, n in enumerate(nbes)))


# ---------------------------------------------------------------------------
#  outputs
# ---------------------------------------------------------------------------
def write_child_results(workchain) -> tuple[int, int]:
    """Write the ``results`` and ``pseudo_family`` outputs; ``(n_ok, n_failed)``.

    ``OrbgenCalcWorkChain.finalize`` does the same once for a single grid point; a
    scan writes one ``results`` dict for all of its children, so the loop lives here
    instead of being copied into both scans.
    """
    results = []
    n_ok = n_failed = 0
    for item in workchain.ctx.children_info:
        node = item["node"]
        ok = is_soft_success(node.is_finished_ok, node.exit_status)
        if ok:
            n_ok += 1
        else:
            n_failed += 1
        results.append({
            "task": item["task"],
            "basis": item["basis"],
            "pk": node.pk,
            "exit_status": node.exit_status,
            "ok": ok,
            "seconds": seconds_of(node),
        })

    workchain.out("results", create_final_results(
        l_max_val=workchain.inputs.l_max.value,
        r_cut_val=workchain.inputs.r_cut.value,
        results_list=results,
    ))
    family_labels = getattr(workchain.ctx, "family_labels", None)
    if family_labels:
        workchain.out("pseudo_family", create_family_label(family_labels[-1]))
    elif getattr(workchain.ctx, "pseudo_family_label", None):
        workchain.out("pseudo_family", create_family_label(workchain.ctx.pseudo_family_label))
    return n_ok, n_failed
