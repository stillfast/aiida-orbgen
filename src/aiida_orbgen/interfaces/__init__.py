"""
Interfaces for aiida_orbgen

Unified interface layer that exposes a small, high-level API.
The ``data`` layer handles AiiDA node conversion; the ``workflows`` layer
handles AiiDA workflows.
This layer offers functional entry points that are easy to test and call.

Main API
--------

NSW (primitive orbitals)
~~~~~~~~~~~~~~
- :func:`generate_nsw`              generate a single primitive spherical Bessel .orb orbital
- :func:`compute_nbes_per_l`        count the Bessel functions of each angular-momentum channel

INPUT (ABACUS main input)
~~~~~~~~~~~~~~~~~~~~~
- :func:`generate_incar`            build the ABACUS INPUT from a SIAB JSON
- :func:`parse_incar`               parse an INPUT file back into a dict

STRU (ABACUS structure)
~~~~~~~~~~~~~~~~~~
- :func:`generate_stru`             build the ABACUS STRU from a SIAB JSON
- :func:`dft_folder_name`           canonical SIAB DFT job folder name
- :func:`generate_atom_coords`      compute the atomic coordinates of a SIAB geometry prototype
- :func:`parse_stru`                parse a STRU file back into a dict
- :func:`read_stru_as_ase`          read a STRU file into an ASE ``Atoms`` object

Pipeline (one shot)
~~~~~~~~~~~~~~~
- :func:`generate_all`              generate NSW plus several INPUT/STRU in one go
- :func:`generate_all_from_json`    one-shot generation from a JSON path

Underlying dependencies
--------
Every interface delegates the actual computation to ``SIAB``
(ABACUS-CSW-NAO), keeping the defaults and the file formats consistent with
the SIAB mainline.
"""

from aiida_orbgen.interfaces.nsw import (
    generate_nsw,
    compute_nbes_per_l,
    folder_rcut,
    apply_grid_point,
)
from aiida_orbgen.interfaces.incar import (
    generate_incar,
    parse_incar,
)
from aiida_orbgen.interfaces.stru import (
    generate_stru,
    dft_folder_name,
    generate_atom_coords,
    parse_stru,
    read_stru_as_ase,
    verify_ase_atoms,
    params_stru_to_ase,
    params_stru_to_ase_validate,
)
from aiida_orbgen.interfaces.pipeline import (
    generate_all,
    generate_all_from_json,
)

__all__ = [
    # NSW
    "generate_nsw",
    "compute_nbes_per_l",
    "folder_rcut",
    "apply_grid_point",
    # INPUT (INCAR)
    "generate_incar",
    "parse_incar",
    # STRU
    "generate_stru",
    "dft_folder_name",
    "generate_atom_coords",
    "parse_stru",
    "read_stru_as_ase",
    "verify_ase_atoms",
    "params_stru_to_ase",
    "params_stru_to_ase_validate",
    # Pipeline
    "generate_all",
    "generate_all_from_json",
]
