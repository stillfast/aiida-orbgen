"""
NSW (Numerical Spherical Wave) interface

Generate NSW primitive spherical Bessel orbital files.

Corresponding file: {elem}_{xc}_{rcut}au_{ecut}Ry_{nzeta_str}.orb
Example: U_gga_9au_100Ry_27s27p26d26f25g.orb

Internal dependencies:
- SIAB.spillage.radial._nbes      : count the Bessel functions per angular momentum
- SIAB.spillage.radial.jl_raw     : generate truncated spherical Bessel functions
- SIAB.spillage.radial.jl_reduce  : reduction of the 'reduced' type
- SIAB.spillage.orbio.write_nao   : write the .orb file
"""

import copy
import os
from typing import Optional

import numpy as np

__all__ = [
    "generate_nsw",
    "compute_nbes_per_l",
    "folder_rcut",
    "apply_grid_point",
    "point_dir_name",
    "LEGACY_POINT_DIR_SUFFIX",
]


def folder_rcut(r_cut: float):
    """``r_cut`` as SIAB names it in directory / file names.

    SIAB builds both the reference-geometry directory (``dft_folder``) and the
    final orbital file name (``SIAB.io.convention.orb``) with
    ``str(r_cut) + 'au'``, so the *type* of the value decides the name: ``10``
    gives ``10au`` while ``10.0`` gives ``10.0au``.  The trees produced by
    :func:`generate_all` are normalised to the integer spelling (see
    ``interfaces/pipeline.generate_all`` and ``generate_nsw``, which both
    truncate), so that spelling is the canonical one -- otherwise a workchain
    run and a later spillage run would disagree about the same grid point and
    SIAB would look for folders that do not exist.

    ``9.0`` → ``9``, ``9.4`` → ``9.4``.
    """
    value = float(r_cut)
    return int(value) if value.is_integer() else value


#: The workflow used to spell the per-grid-point directory ``lmax4_rcut10p0``
#: while the report layer used ``lmax4_rcut10`` for the same point.  The report
#: spelling is the canonical one now (it is also what ``static.dft_roots`` keys
#: use); the legacy spelling is still *read*, so existing run trees keep working.
LEGACY_POINT_DIR_SUFFIX = "p"


def point_dir_name(l_max: int, r_cut: float) -> str:
    """Directory name of one grid point, e.g. ``lmax4_rcut10``.

    ``r_cut`` is rendered the way SIAB names its folders (``10``, not ``10.0``),
    so the directory of a grid point and the reference-geometry folders inside
    it agree.
    """
    return f"lmax{int(l_max)}_rcut{folder_rcut(r_cut)}"


def legacy_point_dir_name(l_max: int, r_cut: float) -> str:
    """The pre-2026-09-29 spelling (``lmax4_rcut10p0``), for lookups only."""
    return f"lmax{int(l_max)}_rcut{str(r_cut).replace('.', LEGACY_POINT_DIR_SUFFIX)}"


def apply_grid_point(config: dict, l_max: int, r_cut: float) -> dict:
    """Return ``config`` with one ``(l_max, r_cut)`` grid point applied.

    This is the **only** implementation of that override.  It used to exist
    twice -- ``workflows/batch.run_siab_pipeline`` wrote ``[float(r_cut)]``
    while ``utils/report/orbitals._apply_grid_point`` wrote the integer
    spelling -- so the same grid point produced differently named primitive and
    final orbitals depending on which layer had derived the config.

    The input mapping is not modified.
    """
    config = copy.deepcopy(config)
    config["bessel_nao_rcut"] = [folder_rcut(r_cut)]
    geoms = config.get("geoms")
    if geoms:
        geoms[0]["lmaxmax"] = int(l_max)
    return config


def compute_nbes_per_l(rcut: float, ecut: float, lmaxmax: int,
                       primitive_type: str = "reduced") -> list:
    """
    Compute the number of Bessel functions for each angular momentum.

    Corresponds to SIAB.spillage.radial._nbes.

    Parameters
    ----------
    rcut : float
        Cutoff radius (Bohr)
    ecut : float
        Kinetic-energy cutoff (Ry)
    lmaxmax : int
        Maximum angular momentum
    primitive_type : str
        Primitive basis type: 'reduced' or 'normalized'

    Returns
    -------
    list[int]
        [nbes_l0, nbes_l1, ..., nbes_l{lmaxmax}]
    """
    from SIAB.spillage.radial import _nbes

    nbes_list = [_nbes(l, rcut, ecut) for l in range(lmaxmax + 1)]

    if primitive_type == "reduced":
        # The 'reduced' type drops one function per l (reduction condition)
        nbes_list = [n - 1 for n in nbes_list]

    return nbes_list


def generate_nsw(
    json_config: dict,
    output_dir: str = "./primitive_jy",
    lmaxmax: Optional[int] = None,
    dr: float = 0.01,
) -> str:
    """
    Generate an NSW primitive orbital file.

    Parameters
    ----------
    json_config : dict
        JSON config dict, which should contain:
        - element: element symbol (e.g. "U")
        - bessel_nao_rcut: list of cutoff radii (e.g. [9])
        - ecutjy: spherical-Bessel kinetic-energy cutoff (e.g. 100)
        - primitive_type: 'reduced' or 'normalized' (default 'reduced')
        - xc: functional name (default 'gga')
    output_dir : str
        Output directory (default "./primitive_jy")
    lmaxmax : int, optional
        Maximum angular momentum; read from geoms[0] by default, 4 if absent
    dr : float
        Radial grid step (Bohr), default 0.01

    Returns
    -------
    str
        Absolute path of the generated .orb file

    Examples
    --------
    >>> config = {
    ...     "element": "U",
    ...     "bessel_nao_rcut": [9],
    ...     "ecutjy": 100,
    ...     "primitive_type": "reduced",
    ... }
    >>> orb_path = generate_nsw(config, output_dir="./primitive_jy")
    >>> print(orb_path)
    /home/user/project/primitive_jy/U_gga_9au_100Ry_27s27p26d26f25g.orb
    """
    # Parse the arguments
    elem = json_config["element"]
    rcut_list = json_config["bessel_nao_rcut"]
    rcut_raw = rcut_list[0] if isinstance(rcut_list, list) else rcut_list
    rcut = float(rcut_raw)
    # Keep the original type (int or float) so the output format matches SIAB
    rcut_for_write = rcut_raw if isinstance(rcut_raw, (int, float)) else rcut
    ecut_raw = json_config["ecutjy"]
    ecut = float(ecut_raw)
    ecut_for_write = ecut_raw if isinstance(ecut_raw, (int, float)) else ecut
    primitive_type = json_config.get("primitive_type", "reduced")
    xc = json_config.get("xc", "gga")

    # If lmaxmax was not given, try to read it from geoms
    if lmaxmax is None:
        geoms = json_config.get("geoms", [])
        if geoms and "lmaxmax" in geoms[0]:
            lmaxmax = geoms[0]["lmaxmax"]
        else:
            lmaxmax = 4

    # Number of Bessel functions per angular momentum
    nbes_list = compute_nbes_per_l(rcut, ecut, lmaxmax, primitive_type)

    # Build the nzeta string
    nzeta_str = _nzeta_to_string(nbes_list)

    # Build the file name
    filename = f"{elem}_{xc}_{int(rcut)}au_{int(ecut)}Ry_{nzeta_str}.orb"

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    # Call SIAB's internal routine to generate the orbitals
    nr = int(rcut / dr) + 1
    r_grid = np.linspace(0, rcut, nr)

    # Generate truncated spherical Bessel functions
    chi = []
    for l in range(lmaxmax + 1):
        nq = nbes_list[l] if primitive_type == "normalized" else nbes_list[l] + 1
        zeta_functions = []
        for q in range(nq):
            chi_lq = _jl_raw(l, q, r_grid, rcut=rcut, deriv=0)
            zeta_functions.append(chi_lq)
        chi.append(zeta_functions)

    # Apply the reduction for the 'reduced' type
    if primitive_type == "reduced":
        chi_reduced = []
        for l in range(lmaxmax + 1):
            nq = nbes_list[l] + 1
            if nq < 2:
                chi_reduced.append(chi[l])
                continue
            T = _jl_reduce(l, nq, rcut, from_raw=True)
            chi_l_reduced = [T[:, k] @ np.array(chi[l]) for k in range(nq - 1)]
            chi_reduced.append(chi_l_reduced)
        chi = chi_reduced

    # Write the file
    _write_nao(
        fpath=filepath,
        elem=elem,
        ecut=ecut_for_write,
        rcut=rcut_for_write,
        nr=nr,
        dr=dr,
        chi=chi,
    )

    return os.path.abspath(filepath)


def _nzeta_to_string(nzeta: list) -> str:
    """Convert the nzeta list into a zeta string."""
    SPECTRUM = "spdfghijklmnopqrstuvwxyz"
    return "".join(
        f"{nz}{sym}" for nz, sym in zip(nzeta, SPECTRUM) if nz > 0
    )


def _jl_raw(l: int, q: int, r: np.ndarray, rcut: Optional[float] = None,
            deriv: int = 0) -> np.ndarray:
    """Local wrapper around SIAB.spillage.radial.jl_raw."""
    from SIAB.spillage.radial import jl_raw
    return jl_raw(l, q, r, rcut=rcut, deriv=deriv)


def _jl_reduce(l: int, n: int, rcut: float, from_raw: bool = True) -> np.ndarray:
    """Local wrapper around SIAB.spillage.radial.jl_reduce."""
    from SIAB.spillage.radial import jl_reduce
    return jl_reduce(l, n, rcut, from_raw=from_raw)


def _write_nao(fpath: str, elem: str, ecut: float, rcut: float,
               nr: int, dr: float, chi: list) -> None:
    """Local wrapper around SIAB.spillage.orbio.write_nao."""
    from SIAB.spillage.orbio import write_nao
    write_nao(fpath, elem, ecut, rcut, nr, dr, chi)
