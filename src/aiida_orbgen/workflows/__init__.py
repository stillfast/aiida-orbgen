"""WorkChain implementations for aiida_orbgen.

Two-level WorkChain architecture:

- :class:`OrbgenCalcWorkChain`  (entry point: ``orbgen.calc``)
  — one ``(l_max, r_cut)``: SIAB pipeline + {PW, LCAO:nsw} x N structures, ΔE
- :class:`OrbgenGridSearchWorkChain`  (entry point: ``orbgen.gridsearch``)
  — many candidates, pick the cheapest acceptable one by ΔE

Both classes live in :mod:`aiida_orbgen.workflows.batch` -- the module
``pyproject.toml`` registers.  This package used to re-export
``OrbgenGridSearchWorkChain`` from ``advanced`` instead, so
``WorkflowFactory("orbgen.gridsearch")`` and
``from aiida_orbgen.workflows import OrbgenGridSearchWorkChain`` returned two
*different* implementations (colliding exit-code numbers, same
``process_label``, indistinguishable in provenance).  There is one source now.

Module layout
--------

``batch.py``     the two WorkChains (orchestration only: submit → collect → decide)
``_grid.py``     grid candidates and the stopping rule (pure logic)
``siab.py``      everything that talks to SIAB, generates/parses its files, and
                 checks the NUMERICAL_ORBITAL reference of an ABACUS child
``energies.py``  the ΔE arithmetic, the tolerance verdict, the 304-is-a-success rule
``extract.py``   reading children (``misc.total_energy``, the STRU guard) into records
``results.py``   assembly of the output Dicts (four calcfunctions)

``advanced.py`` was deleted on 2026-09-29: its three unique capabilities
(``stop_on_first_valid``, an explicit ``candidates`` list, and the
``max_l_max`` / ``max_r_cut`` caps) were ported into the grid search in
``batch``; see ``_removed-20260929/``.
"""

from aiida_orbgen.workflows.batch import (
    OrbgenCalcWorkChain,
    OrbgenGridSearchWorkChain,
)
from aiida_orbgen.workflows.siab import (
    build_abacus_child_inputs,
    n_atoms_from_stru,
    run_siab_pipeline,
)

__all__ = [
    "OrbgenCalcWorkChain",
    "OrbgenGridSearchWorkChain",
    "build_abacus_child_inputs",
    "n_atoms_from_stru",
    "run_siab_pipeline",
]
