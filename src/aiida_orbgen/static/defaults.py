"""
aiida_orbgen.static.defaults - default parameters

Central location for every tunable parameter of ``aiida-orbgen``. CLI / Python
API callers can:

- ``import`` them directly and use them as read-only constants
- override them with kwargs at the entry points (``build_submission_config`` etc.)
- customise them by copying this file and passing it via ``--config-file`` (TODO)

After modifying this file, run ``pytest tests/`` to confirm nothing broke.
"""
from __future__ import annotations

from typing import Any, Dict


# ---------------------------------------------------------------------------
# AiiDA / scheduler
# ---------------------------------------------------------------------------

# AiiDA code label already registered in the database (verdi code list)
DEFAULT_CODE_LABEL: str = "abacus_lts@yeesuan"

# Slurm queue name
DEFAULT_QUEUE_NAME: str = "q_ysuan"

# Number of MPI processes per node
DEFAULT_NUM_MPI: int = 56

# Wallclock per node (seconds)
DEFAULT_WALLCLOCK_SECONDS: int = 6 * 60 * 60  # 6 h

DEFAULT_MAX_MEMORY_KB: int = 350 * 1024 * 1024

# Maximum number of workflow retries
DEFAULT_MAX_ITERATIONS: int = 3


# ---------------------------------------------------------------------------
# resources sub-dictionary (plugged into metadata.options)
# ---------------------------------------------------------------------------

DEFAULT_RESOURCES: Dict[str, Any] = {
    "num_machines": 1,
    "num_mpiprocs_per_machine": DEFAULT_NUM_MPI,
    "tot_num_mpiprocs": DEFAULT_NUM_MPI,
}


# ---------------------------------------------------------------------------
# Full metadata.options template
# ---------------------------------------------------------------------------

DEFAULT_METADATA_OPTIONS: Dict[str, Any] = {
    "resources": dict(DEFAULT_RESOURCES),
    "max_wallclock_seconds": DEFAULT_WALLCLOCK_SECONDS,
    "max_memory_kb": DEFAULT_MAX_MEMORY_KB,
    "queue_name": DEFAULT_QUEUE_NAME,
    "withmpi": True,
}


# ---------------------------------------------------------------------------
# AiiDA-managed INPUT keys
#
# These keys are not forwarded to the ``parameters`` of the abacus.base
# workchain, because AiiDA / pseudo_family handles them:
#   - pseudo_dir / orbital_dir  : injected automatically by AiiDA (from pseudo_family)
#   - stru_file  / kpoint_file  : we pass them explicitly via StructureData / KpointsData
#   - wannier_card / suffix     : not needed
#   - calculation               : decided by the workflow (default scf)
#
# ``out_wfc_lcao`` is deliberately **not** filtered any more (2026-09-20): the
# SIAB pipeline sets it to 1 so that ABACUS writes ``WFC_NAO_GAMMA1.txt``, which
# ``SIAB.spillage`` needs to fit the reference orbitals. Dropping it here meant
# every AiiDA-run LCAO job silently produced no wavefunctions, and the spillage
# step then had to rebuild them from ``data-0-H``/``data-0-S``
# (``aiida_orbgen.utils.report.assemble``). PW jobs never see the key:
# ``build_abacus_child_inputs`` pops it for ``basis == "pw"``.
# ---------------------------------------------------------------------------

AIIDA_MANAGED_KEYS: frozenset = frozenset({
    "pseudo_dir",
    "orbital_dir",
    "stru_file",
    "kpoint_file",
    "wannier_card",
    "suffix",
    "calculation",
    "basis_type",  # set dynamically by the WorkChain from the basis list
    "bessel_nao_rcut",
    "bessel_nao_lmax",
    "bessel_nao_tolerence",
    "bessel_descriptor",
    "bessel_smooth",
    "bessel_ecut",
    "bessel_rcut",
    "bessel_sigma",
    "bessel_screen_coeff",
})


# ---------------------------------------------------------------------------
# Default family label template
#
# Example: siab-u-nr-pbe-z6-nsw-9au-100Ry-g
#
# The placeholders are filled in by ``resolve_family_label()``:
#   {element}  {rel}  {pp_xc}  {zval}  {rcut}  {ecut}  {lmax}
#   zval: number of valence electrons (extracted from z_valence in the UPF
#   PP_HEADER), used to distinguish different pseudopotentials
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_LABEL_TEMPLATE: str = (
    "siab-{element}-{rel}-{pp_xc}-z{zval}-nsw-{rcut}au-{ecut}Ry-{lmax}"
)

# Order of the family label fields (used to extract them from ORB / UPF file names)
FAMILY_LABEL_KEYS: tuple = (
    "element", "rel", "pp_xc", "zval", "rcut", "ecut", "lmax",
)


# ---------------------------------------------------------------------------
# INPUT parameter overrides
#
# These fields are forcibly overridden after the SIAB ``generate_incar`` writes
# the INPUT file, in order to correct SIAB defaults (for example changing
# ``ks_solver=genelpa`` into ``scalapack_gvx``).
#
# Key = INPUT field name (standard ABACUS), value = override value.
# After a modification, the next ``run`` / ``report`` will generate the INPUT
# with the new value.
# Note: these values end up in the ``parameters.input`` submitted to the final
# abacus.base calculation.
# ---------------------------------------------------------------------------

INPUT_OVERRIDES: Dict[str, Any] = {
    "ks_solver": "scalapack_gvx",
}


def apply_input_overrides(params: Dict[str, Any]) -> Dict[str, Any]:
    """Override ``params`` with ``INPUT_OVERRIDES`` (in-place + return).

    Keys managed by AiiDA (pseudo_dir / orbital_dir / ...) are filtered out too.
    """
    from aiida_orbgen.static.defaults import AIIDA_MANAGED_KEYS
    out = {k: v for k, v in params.items() if k not in AIIDA_MANAGED_KEYS}
    out.update(INPUT_OVERRIDES)
    return out


# ---------------------------------------------------------------------------
# Top-level ``DEFAULTS`` view (convenient for ``from aiida_orbgen.static import DEFAULTS``)
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "code_label": DEFAULT_CODE_LABEL,
    "queue_name": DEFAULT_QUEUE_NAME,
    "num_mpi": DEFAULT_NUM_MPI,
    "wallclock_seconds": DEFAULT_WALLCLOCK_SECONDS,
    "max_memory_kb": DEFAULT_MAX_MEMORY_KB,
    "max_iterations": DEFAULT_MAX_ITERATIONS,
    "resources": DEFAULT_RESOURCES,
    "metadata_options": DEFAULT_METADATA_OPTIONS,
    "aiida_managed_keys": AIIDA_MANAGED_KEYS,
    "family_label_template": DEFAULT_FAMILY_LABEL_TEMPLATE,
}
