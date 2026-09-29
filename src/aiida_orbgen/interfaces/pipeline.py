"""
Pipeline interface

Generate a complete ABACUS calculation directory from a SIAB JSON config in
one shot:

- ``primitive_jy/{elem}_{xc}_{rcut}au_{ecut}Ry_{nzeta}.orb``  (NSW primitive orbital)
- ``{proto}-{pert}-{rcut}au/INPUT``                          (ABACUS main input)
- ``{proto}-{pert}-{rcut}au/STRU``                           (ABACUS structure)

Corresponds to the first-stage products of SIAB's ``SIAB.orbgen.main`` workflow.

Bond lengths (``pertmags``) are delegated entirely to SIAB:

- list: expand over the given bond lengths in order
- ``"auto"``: call ``SIAB.abacus.blscan.blgen`` (equivalent to
  ``SIAB.abacus.api._build_pert(pertmags='auto')``), which uses SIAB's internal
  default bond-length tables such as ``CellGenerator.get_dimer_bond_length``
- ``"scan"``: like ``"auto"``, but also screens bond lengths against existing DFT results
"""

import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path

from aiida_orbgen.interfaces.nsw import generate_nsw
from aiida_orbgen.interfaces.incar import generate_incar
from aiida_orbgen.interfaces.stru import generate_stru, dft_folder_name

__all__ = ["generate_all", "generate_all_from_json"]


def _resolve_pertmags(
    json_config: dict,
    proto: str,
    override: Optional[List[float]],
    bond_length: Optional[float],
    geom: Optional[dict] = None,
) -> List[float]:
    """
    Decide the list of bond lengths to expand over.  Priority:

    1. ``override``: an explicitly passed list
    2. ``bond_length``: a single value -> wrapped into a one-element list
    3. *geom*'s ``pertmags`` (default: ``geoms[0]``): a list is used as is; the
       strings ``"auto"``/``"scan"`` call SIAB
    4. *geom*'s ``pertmags``: a single number -> wrapped into a one-element list
    5. fallback: call SIAB's ``_build_pert(pertmags='auto')`` for the default
       bond lengths

    ``geoms`` cannot list a ``monomer``: SIAB's ``GeomAssert`` only accepts
    dimer/trimer/square/tetrahedron/octahedron/cube there, and the monomer of the
    ``atomic`` initial guess is a job SIAB appends by itself (mirrored by
    :func:`generate_all`).

    No "automatic" path hard-codes numbers in the interface layer.
    """
    if override is not None:
        return [float(p) for p in override]
    if bond_length is not None:
        return [float(bond_length)]

    geoms = json_config.get("geoms", [])
    if geom is None:
        geom = geoms[0] if geoms else {}
    pertmags_raw = geom.get("pertmags")

    if isinstance(pertmags_raw, str):
        # "auto" / "scan" → handed over to SIAB entirely
        return _siab_pertmags(json_config, proto, pertmags_raw)
    if isinstance(pertmags_raw, list):
        return [float(p) for p in pertmags_raw]
    if isinstance(pertmags_raw, (int, float)):
        return [float(pertmags_raw)]

    # The top-level JSON has no geoms or pertmags: use the SIAB default
    return _siab_pertmags(json_config, proto, "auto")


def _siab_pertmags(
    json_config: dict,
    proto: str,
    mode: str,
) -> List[float]:
    """
    Call SIAB's ``_build_pert`` / ``blgen`` to obtain automatic bond lengths.

    The two modes SIAB supports:
    - ``"auto"``: SIAB consults the ``CellGenerator.{proto}_bond_length(elem)`` table
    - ``"scan"``: like ``"auto"``, but also reads existing DFT results to screen them
      (only the initial list is produced here; screening is left to later steps)
    """
    from SIAB.abacus.api import _build_pert

    elem = json_config["element"]
    geoms = json_config.get("geoms", [])
    pertkind = geoms[0].get("pertkind", "stretch") if geoms else "stretch"

    return list(_build_pert(elem, proto, pertkind, mode))


def generate_all(
    json_config: dict,
    output_root: str = ".",
    bond_length: Optional[float] = None,
    pertmags: Optional[list] = None,
    proto: Optional[str] = None,
    nspin: Optional[int] = None,
    lattice_constant: Optional[float] = None,
    lmaxmax: Optional[int] = None,
    dr: float = 0.01,
    nsw_dirname: str = "primitive_jy",
) -> Dict[str, Any]:
    """
    Generate an NSW orbital plus several INPUT/STRU job directories in one shot.

    Parameters
    ----------
    json_config : dict
        SIAB JSON config dict (project/pbe/pbe_orbgen.json style)
    output_root : str
        Root output directory.  All paths are relative to it:
        - NSW:    {output_root}/{nsw_dirname}/{orb_name}
        - INPUT:  {output_root}/{dft_folder}/INPUT
        - STRU:   {output_root}/{dft_folder}/STRU
    bond_length : float, optional
        Bond length of a single DFT job (Angstrom).  If given, only one job
        directory is generated.
        Mutually exclusive with ``pertmags``.
    pertmags : list, optional
        Explicit list of bond lengths.  If given, expand over them in order.
        When omitted:
        - JSON ``geoms[0].pertmags`` is a list → expand over the list
        - JSON ``geoms[0].pertmags`` is ``"auto"``/``"scan"`` → call the SIAB default
    proto : str, optional
        Geometry prototype (overrides the JSON setting)
    nspin : int, optional
        Spin polarisation (1 or 2)
    lattice_constant : float, optional
        Lattice constant (Bohr, overrides the JSON setting)
    lmaxmax : int, optional
        Maximum angular momentum (inferred from geoms[0].lmaxmax)
    dr : float
        Radial grid step of the NSW orbital (Bohr), default 0.01
    nsw_dirname : str
        Directory name of the NSW orbital (default "primitive_jy")

    Returns
    -------
    dict
        - nsw:         absolute path of the NSW orbital file
        - nsw_filename: basename of the NSW orbital file
        - pertmags:    the bond-length list actually used
        - dft:         list of {folder, pert, input, stru}

    Examples
    --------
    >>> import json
    >>> with open("pbe_orbgen.json") as f:
    ...     config = json.load(f)
    >>> # The JSON says "pertmags": "auto", so SIAB generates the bond lengths
    >>> result = generate_all(config, output_root="./generated")
    >>> # Same as project/pbe/: U-dimer-1.89-9au ... U-dimer-4.50-9au are generated
    >>> # automatically
    >>> print([d["folder"] for d in result["dft"]])
    ['U-dimer-1.89-9au', 'U-dimer-2.09-9au', 'U-dimer-2.75-9au',
     'U-dimer-3.50-9au', 'U-dimer-4.50-9au']
    """
    # 1. NSW primitive orbital: generated only once
    nsw_dir = os.path.join(output_root, nsw_dirname)
    nsw_path = generate_nsw(
        json_config,
        output_dir=nsw_dir,
        lmaxmax=lmaxmax,
        dr=dr,
    )
    orb_filename = os.path.basename(nsw_path)

    # 2. Decide the list of DFT jobs to expand over
    elem = json_config["element"]
    if proto is None and json_config.get("geoms"):
        proto = json_config["geoms"][0].get("proto", "dimer")
    proto = proto or "dimer"

    rcut_raw = json_config.get("bessel_nao_rcut", [9])
    rcut_raw_first = rcut_raw[0] if isinstance(rcut_raw, list) else rcut_raw
    # Keep the original type (int / float) so that 9 -> 9.0 does not change
    # the folder name
    if isinstance(rcut_raw_first, float) and rcut_raw_first.is_integer():
        rcut_for_folder: Any = int(rcut_raw_first)
    else:
        rcut_for_folder = rcut_raw_first

    # 3. Generate INPUT/STRU for every reference geometry.
    #
    #    Every ``geoms`` entry is expanded, not just ``geoms[0]``: SIAB's spillage
    #    step needs the *monomer* reference (`SIAB/driver/main.py` sets
    #    ``model_kwargs['jobdir'] = dft_folder(elem, 'monomer', 0, rcut=...)`` for its
    #    ``atomic`` initial guess), and a job folder is the only way that data ever
    #    reaches the workflow.  With a dimer-only list SIAB dies at the very end with
    #    ``FileNotFoundError: 'U-monomer-9au/OUT.ABACUS/INPUT'`` — after every child
    #    has been paid for (2026-09-30).
    geom_entries = json_config.get("geoms") or [{}]
    dft_results: List[Dict[str, Any]] = []
    seen_folders: set[str] = set()
    pertmags_first: List[float] = []

    for index, geom in enumerate(geom_entries):
        geom = geom or {}
        geom_proto = proto if (index == 0 and proto) else geom.get("proto") or proto or "dimer"
        geom_perts = _resolve_pertmags(
            json_config,
            geom_proto,
            pertmags if index == 0 else None,
            bond_length if index == 0 else None,
            geom=geom,
        )
        if index == 0:
            pertmags_first = geom_perts

        for pert in geom_perts:
            folder = dft_folder_name(elem, geom_proto, pert, rcut=rcut_for_folder)
            if folder in seen_folders:
                # e.g. the same geometry listed twice, or a monomer whose folder
                # name ignores the perturbation
                continue
            seen_folders.add(folder)
            dft_root = os.path.join(output_root, folder)
            os.makedirs(dft_root, exist_ok=True)

            input_path = generate_incar(
                json_config,
                output_path=os.path.join(dft_root, "INPUT"),
            )
            stru_path = generate_stru(
                json_config,
                output_path=os.path.join(dft_root, "STRU"),
                proto=geom_proto,
                bond_length=pert,
                nspin=geom.get("nspin", nspin),
                lattice_constant=geom.get("celldm", lattice_constant),
                orb_filename=orb_filename,
            )
            dft_results.append({
                "folder": folder,
                "pert": pert,
                "proto": geom_proto,
                "input": os.path.abspath(input_path),
                "stru": os.path.abspath(stru_path),
            })

    # 4. The `atomic` initial guess needs a *monomer* reference, and SIAB adds that job
    #    itself (SIAB/abacus/api.py:build_abacus_jobs appends proto='monomer' with
    #    nbands=69 when spill_guess == 'atomic').  `geoms` cannot name it -- SIAB's
    #    GeomAssert only accepts dimer/trimer/square/tetrahedron/octahedron/cube -- so
    #    it is appended here: the workflow then computes it as an ordinary child, its
    #    data comes back through AiiDA, and SIAB finds a *completed* folder instead of
    #    either crashing on the missing `OUT.<suffix>/INPUT` (2026-09-30) or running
    #    that DFT on whatever machine executes `report`.
    if str(json_config.get("spill_guess", "atomic")) == "atomic":
        monomer_folder = dft_folder_name(elem, "monomer", 0.0, rcut=rcut_for_folder)
        if monomer_folder not in seen_folders:
            seen_folders.add(monomer_folder)
            dft_root = os.path.join(output_root, monomer_folder)
            os.makedirs(dft_root, exist_ok=True)
            # SIAB passes nbands=69 for this job specifically (`param_specific`); it is
            # what the atomic guess indexes its bands against, so it must not inherit
            # the dimers' value.
            monomer_config = dict(json_config)
            monomer_config["nbands"] = int(
                json_config.get("__iop_spill_guess_atomic_nbands__", 69)
            )
            input_path = generate_incar(
                monomer_config,
                output_path=os.path.join(dft_root, "INPUT"),
            )
            geom0 = geom_entries[0] or {}
            stru_path = generate_stru(
                json_config,
                output_path=os.path.join(dft_root, "STRU"),
                proto="monomer",
                bond_length=0.0,
                nspin=geom0.get("nspin", nspin),
                lattice_constant=geom0.get("celldm", lattice_constant),
                orb_filename=orb_filename,
            )
            dft_results.append({
                "folder": monomer_folder,
                "pert": 0.0,
                "proto": "monomer",
                "input": os.path.abspath(input_path),
                "stru": os.path.abspath(stru_path),
            })

    return {
        "nsw": os.path.abspath(nsw_path),
        "nsw_filename": orb_filename,
        # the perturbations of the first geometry: what the dimers were expanded
        # over, and what the report shows
        "pertmags": pertmags_first,
        "dft": dft_results,
    }


def generate_all_from_json(
    json_path: str,
    output_root: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    One-shot generation of NSW + INPUT + STRU from a JSON file path.

    Parameters
    ----------
    json_path : str
        Path of the SIAB JSON config file
    output_root : str, optional
        Root output directory (default: the directory containing the JSON file)
    **kwargs
        Passed through to :func:`generate_all`

    Returns
    -------
    dict
        Same as :func:`generate_all`
    """
    json_path = os.path.abspath(json_path)
    with open(json_path, "r") as f:
        config = json.load(f)

    if output_root is None:
        output_root = str(Path(json_path).parent)

    return generate_all(config, output_root=output_root, **kwargs)
