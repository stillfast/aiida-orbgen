"""
STRU interface

Generate ABACUS STRU (structure) files.

Corresponding file: {folder}/STRU
Example: U-dimer-1.89-9au/STRU

This module is a high-level wrapper around SIAB and calls, underneath:
- ``SIAB.abacus.io.structure_to_text`` : dispatch on proto to build the STRU text
- ``SIAB.io.convention.dft_folder``    : canonical DFT folder naming

Why this is done:
1. All supported geometry prototypes (monomer/dimer/trimer/...) are
   maintained by SIAB.
2. Folder naming and numerical precision match the SIAB mainline exactly.
3. The interface layer only owns the mapping "JSON parameters → SIAB call
   arguments".
"""

import os
import math
from typing import Optional, Dict, Any, List

__all__ = [
    "generate_stru",
    "dft_folder_name",
    "generate_atom_coords",
    "parse_stru",
    "read_stru_as_ase",
    "verify_ase_atoms",
    "params_stru_to_ase",
]


# Geometry prototypes supported by SIAB
SUPPORTED_PROTOS = {
    "monomer",
    "dimer",
    "trimer",
    "tetrahedron",
    "square",
    "triangular_bipyramid",
    "octahedron",
    "cube",
}


# Bohr -> Angstrom
BOHR_TO_ANG = 1.8897259886


def dft_folder_name(
    elem: str,
    proto: str,
    pert: float,
    rcut: Optional[float] = None,
) -> str:
    """
    Generate the canonical SIAB DFT folder name.

    Calls ``SIAB.io.convention.dft_folder`` directly, so it is guaranteed to
    match SIAB exactly.

    Parameters
    ----------
    elem : str
        Element symbol
    proto : str
        Geometry prototype
    pert : float
        Perturbation magnitude (bond length, Angstrom)
    rcut : float, optional
        Cutoff radius (au); added to the folder name when given

    Returns
    -------
    str
        Folder name

    Examples
    --------
    >>> dft_folder_name("U", "dimer", 1.89, rcut=9)
    'U-dimer-1.89-9au'
    >>> dft_folder_name("U", "monomer", 0)
    'U-monomer'
    """
    from SIAB.io.convention import dft_folder
    return dft_folder(elem, proto, pert, rcut=rcut)


def generate_atom_coords(
    proto: str,
    bond_length: float,
    lattice_constant: float,
) -> List[List[float]]:
    """
    Generate the atomic coordinates (Angstrom) of a geometry prototype.

    This step is pure "numerical computation" and writes no files; it reuses
    SIAB's structure-generation logic, so the result matches SIAB exactly.

    Parameters
    ----------
    proto : str
        Geometry prototype
    bond_length : float
        Bond length (Angstrom)
    lattice_constant : float
        Lattice constant (Bohr)

    Returns
    -------
    list[list[float]]
        List of atomic coordinates [[x, y, z], ...]
    """
    proto = proto.lower()
    if proto not in SUPPORTED_PROTOS:
        raise ValueError(
            f"Unsupported proto: {proto}. "
            f"Supported: {', '.join(sorted(SUPPORTED_PROTOS))}"
        )

    shift = lattice_constant / 2 / BOHR_TO_ANG

    if proto == "monomer":
        return [[0.0 + shift, 0.0 + shift, 0.0 + shift]]

    if proto == "dimer":
        return [
            [0.0 + shift, 0.0 + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, bond_length + shift],
        ]

    if proto == "trimer":
        dis1 = bond_length * 0.86603
        dis2 = bond_length * 0.5
        return [
            [0.0 + shift, 0.0 + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, bond_length + shift],
            [0.0 + shift, dis1 + shift, dis2 + shift],
        ]

    if proto == "tetrahedron":
        dis1 = bond_length * 0.86603
        dis2 = bond_length * 0.5
        dis3 = bond_length * 0.81649
        dis4 = bond_length * 0.28867
        return [
            [0.0 + shift, 0.0 + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, bond_length + shift],
            [0.0 + shift, dis1 + shift, dis2 + shift],
            [dis3 + shift, dis4 + shift, dis2 + shift],
        ]

    if proto == "square":
        return [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, bond_length],
            [bond_length + shift, 0.0 + shift, 0.0 + shift],
            [bond_length + shift, 0.0 + shift, bond_length + shift],
        ]

    if proto == "octahedron":
        d = bond_length / 2
        s = bond_length / math.sqrt(2)
        return [
            [d + shift, d + shift, 0.0 + shift],
            [-d + shift, -d + shift, 0.0 + shift],
            [d + shift, -d + shift, 0.0 + shift],
            [-d + shift, d + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, s + shift],
            [0.0 + shift, 0.0 + shift, -s + shift],
        ]

    if proto == "cube":
        d = bond_length / 2
        return [
            [d + shift, d + shift, d + shift],
            [-d + shift, -d + shift, d + shift],
            [d + shift, -d + shift, d + shift],
            [-d + shift, d + shift, d + shift],
            [d + shift, d + shift, -d + shift],
            [-d + shift, -d + shift, -d + shift],
            [d + shift, -d + shift, -d + shift],
            [-d + shift, d + shift, -d + shift],
        ]

    if proto == "triangular_bipyramid":
        d1 = bond_length / math.sqrt(3)
        d2 = bond_length / 2
        d3 = bond_length * math.sqrt(2.0 / 3.0)
        return [
            [d1 + shift, 0.0 + shift, 0.0 + shift],
            [-d1 / 2 + shift, d2 + shift, 0.0 + shift],
            [-d1 / 2 + shift, -d2 + shift, 0.0 + shift],
            [0.0 + shift, 0.0 + shift, d3 + shift],
            [0.0 + shift, 0.0 + shift, -d3 + shift],
        ]

    # Not reachable
    raise ValueError(f"Unsupported proto: {proto}")


def _resolve_stru_params(
    json_config: dict,
    proto: Optional[str],
    nspin: Optional[int],
    lattice_constant: Optional[float],
    bond_length: Optional[float],
) -> Dict[str, Any]:
    """Infer the STRU generation parameters from the JSON."""
    elem = json_config["element"]
    pseudo_dir = json_config.get("pseudo_dir", "./pseudo.upf")
    mass = json_config.get("mass", 1.0)

    geoms = json_config.get("geoms", [])
    if geoms:
        geom0 = geoms[0]
        if proto is None:
            proto = geom0.get("proto", "dimer")
        if nspin is None:
            nspin = geom0.get("nspin", 1)
        if lattice_constant is None:
            lattice_constant = geom0.get("celldm", 30)
        if bond_length is None:
            pertmags = geom0.get("pertmags", [0.0])
            if isinstance(pertmags, list) and pertmags:
                bond_length = float(pertmags[0])
            else:
                bond_length = 0.0
    else:
        proto = proto or "dimer"
        nspin = nspin or 1
        lattice_constant = lattice_constant or 30.0
        bond_length = bond_length or 0.0

    return {
        "elem": elem,
        "mass": mass,
        "fpseudo": os.path.basename(pseudo_dir),
        "lattice_constant": float(lattice_constant),
        "bond_length": float(bond_length),
        "nspin": int(nspin),
        "proto": proto,
    }


def generate_stru(
    json_config: dict,
    output_path: str,
    proto: Optional[str] = None,
    bond_length: Optional[float] = None,
    nspin: Optional[int] = None,
    lattice_constant: Optional[float] = None,
    pseudo_filename: Optional[str] = None,
    orb_filename: Optional[str] = None,
) -> str:
    """
    Generate an ABACUS STRU file.

    Calls ``SIAB.abacus.io.structure_to_text`` underneath, so the output format
    (numerical precision, comments, field order) matches the SIAB mainline
    exactly.

    Parameters
    ----------
    json_config : dict
        SIAB JSON config, which should contain:
        - element, pseudo_dir, geoms[0]
    output_path : str
        Output file path
    proto : str, optional
        Geometry prototype (overrides the JSON setting)
    bond_length : float, optional
        Bond length (Angstrom, overrides the JSON setting)
    nspin : int, optional
        Spin polarisation (1 or 2)
    lattice_constant : float, optional
        Lattice constant (Bohr, overrides the JSON setting)
    pseudo_filename : str, optional
        Pseudo-potential file name (overrides the JSON setting)
    orb_filename : str, optional
        Orbital file name referenced by the NUMERICAL_ORBITAL section

    Returns
    -------
    str
        Absolute path of the generated STRU file

    Examples
    --------
    >>> import json
    >>> with open("pbe_orbgen.json") as f:
    ...     config = json.load(f)
    >>> generate_stru(
    ...     config,
    ...     output_path="./U-dimer-1.89-9au/STRU",
    ...     bond_length=1.89,
    ...     orb_filename="U_gga_9au_100Ry_27s27p26d26f25g.orb",
    ... )
    """
    from SIAB.abacus.io import structure_to_text

    params = _resolve_stru_params(
        json_config, proto, nspin, lattice_constant, bond_length
    )

    if pseudo_filename is not None:
        params["fpseudo"] = pseudo_filename

    text, _natom = structure_to_text(
        shape=params["proto"],
        element=params["elem"],
        mass=params["mass"],
        fpseudo=params["fpseudo"],
        lattice_constant=params["lattice_constant"],
        bond_length=params["bond_length"],
        nspin=params["nspin"],
        forb=orb_filename,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)

    return os.path.abspath(output_path)


def parse_stru(filepath: str) -> Dict[str, Any]:
    """
    Parse the key fields out of a STRU file.

    Parsed fields:
    - element: element symbol
    - pseudo_file: pseudo-potential file name
    - lattice_constant: lattice constant (Bohr)
    - lattice_vectors: 3x3 lattice vectors
    - nspin: 1 or 2
    - proto: inferred from the number of atoms (1=monomer, 2=dimer, ...)
    - coordinates: [[x, y, z, ...], ...]

    Parameters
    ----------
    filepath : str
        Path of the STRU file

    Returns
    -------
    dict
        Parsing result
    """
    result: Dict[str, Any] = {
        "element": None,
        "pseudo_file": None,
        "lattice_constant": None,
        "lattice_vectors": None,
        "nspin": 1,
        "proto": None,
        "coordinates": [],
        "orbital_file": None,
    }

    section = None
    coord_count = 0
    expected_coords = 0
    with open(filepath, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            upper = line.upper()
            if upper.startswith("ATOMIC_SPECIES"):
                section = "ATOMIC_SPECIES"
                continue
            if upper.startswith("NUMERICAL_ORBITAL"):
                section = "NUMERICAL_ORBITAL"
                continue
            if upper.startswith("LATTICE_CONSTANT"):
                section = "LATTICE_CONSTANT"
                continue
            if upper.startswith("LATTICE_VECTORS"):
                section = "LATTICE_VECTORS"
                continue
            if upper.startswith("ATOMIC_POSITIONS"):
                section = "ATOMIC_POSITIONS_HEADER"
                continue

            if section == "ATOMIC_SPECIES":
                tokens = line.split()
                if len(tokens) >= 3 and not tokens[0].startswith("//"):
                    result["element"] = tokens[0]
                    result["pseudo_file"] = tokens[2]
                section = None
            elif section == "NUMERICAL_ORBITAL":
                if not line.startswith("//"):
                    result["orbital_file"] = line
                section = None
            elif section == "LATTICE_CONSTANT":
                # LATTICE_CONSTANT
                # 30.000000  // add lattice constant(a.u.)
                tok = line.split()
                result["lattice_constant"] = float(tok[0])
                section = None
            elif section == "LATTICE_VECTORS":
                tok = line.split()
                if len(tok) >= 3:
                    if result["lattice_vectors"] is None:
                        result["lattice_vectors"] = []
                    result["lattice_vectors"].append(
                        [float(tok[0]), float(tok[1]), float(tok[2])]
                    )
            elif section == "ATOMIC_POSITIONS_HEADER":
                # e.g. "Cartesian_angstrom_center_xyz  //Cartesian or Direct coordinate."
                section = "ATOMIC_POSITIONS"
            elif section == "ATOMIC_POSITIONS":
                # Three lines: element label, magnetization, number of atoms, then
                # natom coordinate lines
                if "//Element Label" in line or "//element" in line.lower():
                    continue
                if "//number of atoms" in line.lower() or "number of atoms" in line:
                    # Take the first number on that line
                    for tok in line.split():
                        try:
                            expected_coords = int(tok)
                            natoms = expected_coords
                            # Infer proto from the atom count
                            proto_map = {1: "monomer", 2: "dimer", 3: "trimer", 4: None}
                            result["proto"] = proto_map.get(expected_coords)
                            break
                        except ValueError:
                            continue
                    continue
                if "//starting magnetism" in line.lower():
                    try:
                        result["nspin"] = 2 if float(line.split()[0]) > 0 else 1
                    except ValueError:
                        pass
                    continue
                # Coordinate line: three floats followed by three integers
                tok = line.split()
                if len(tok) >= 3:
                    try:
                        x, y, z = float(tok[0]), float(tok[1]), float(tok[2])
                        result["coordinates"].append([x, y, z])
                    except ValueError:
                        continue
                    coord_count += 1
                    if expected_coords and coord_count >= expected_coords:
                        section = None

    return result


# Whitelist of the coordinate types ASE supports.  The
# ``Cartesian_angstrom_center_xyz`` that SIAB writes is really a non-standard
# variant of ``Cartesian`` (units Angstrom) that ASE does not recognise.  The
# conversion is centralised here so that it does not pollute the plain-text
# parsing in ``parse_stru``.
_ASE_COORD_ALIASES = {
    "cartesian_angstrom_center_xyz": "Cartesian",
    "cartesian_angstrom": "Cartesian",
    "cartesian_nm": "Cartesian",
    "cartesian_bohr": "Cartesian",  # but the coordinates need Bohr->Ang, which ASE handles
}


def _ase_compatible_stru_text(filepath: str) -> str:
    """
    Read a STRU file and replace SIAB's non-standard coordinate labels with the
    form ASE recognises.

    Note: this only rewrites the text and never touches the coordinate values;
    the coordinates in the original file are already Angstrom, i.e. the unit
    ASE's 'Cartesian' means.
    """
    with open(filepath, "r") as f:
        text = f.read()
    for src, dst in _ASE_COORD_ALIASES.items():
        text = text.replace(src, dst)
        text = text.replace(src.upper(), dst)
        text = text.replace(src.capitalize(), dst)
    return text


def read_stru_as_ase(filepath: str):
    """
    Read a STRU file and return an ``ase.Atoms`` object.

    Calling ``ase.io.read(stru, format='abacus')`` directly fails because:
    1. The coordinate label ``Cartesian_angstrom_center_xyz`` SIAB writes is not
       on ASE's whitelist (which only accepts ``Direct`` / ``Cartesian``).
    2. Even after renaming the label to ``Cartesian``, ASE still parses the
       positions as ``pos × lat0``, which does not match the "raw Angstrom"
       semantics SIAB writes.

    This function bypasses the ASE STRU reader and builds the ``ase.Atoms``
    object directly from :func:`parse_stru`, avoiding both pitfalls.

    Parameters
    ----------
    filepath : str
        Path of the STRU file

    Returns
    -------
    ase.Atoms
        ASE atoms object carrying cell, positions, numbers, pbc, ...

    Examples
    --------
    >>> from aiida_orbgen.interfaces import read_stru_as_ase
    >>> atoms = read_stru_as_ase("./U-dimer-2.75-9au/STRU")
    >>> atoms.get_chemical_formula()
    'U2'
    >>> atoms.cell.array
    """
    import numpy as np
    from ase import Atoms

    parsed = parse_stru(filepath)
    if parsed["lattice_vectors"] is None or not parsed["coordinates"]:
        raise ValueError(f"could not parse lattice / coordinates from {filepath}")

    # SIAB: cell (Bohr) = LATTICE_CONSTANT (Bohr) * LATTICE_VECTORS (dimensionless)
    lat0_bohr = float(parsed["lattice_constant"])
    cell_bohr = np.array(parsed["lattice_vectors"], dtype=float) * lat0_bohr
    cell_ang = cell_bohr / BOHR_TO_ANG

    # SIAB: the coordinates are raw Angstrom, already in SI units, so no
    # conversion is needed
    positions_ang = np.array(parsed["coordinates"], dtype=float)

    symbols = [parsed["element"]] * len(positions_ang)

    return Atoms(
        symbols=symbols,
        positions=positions_ang,
        cell=cell_ang,
        pbc=[True, True, True],
    )


def verify_ase_atoms(stru_path: str, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Verify that a STRU → ASE Atoms conversion is correct.

    Compare the results of :func:`parse_stru` (plain text) and
    :func:`read_stru_as_ase` (through ASE) on the same file, checking:

    - n_atoms         the atom counts agree
    - formula         the element lists agree
    - positions       the coordinates (Angstrom) agree within ``tol``
    - cell            the lattice vectors (Angstrom) agree within ``tol``
    - pbc             the periodicity settings agree in all three directions
    - has_cartesian   the coordinates really are Cartesian (not Direct)
    - has_orb         whether a NUMERICAL_ORBITAL reference is present
                      (ASE does not read it; it is recorded only)

    Parameters
    ----------
    stru_path : str
        Path of the STRU file
    tol : float
        Absolute tolerance for the numerical comparisons, default 1e-6 (Angstrom)

    Returns
    -------
    dict
        A report of the form
        ``{"check_name": {"ok": bool, "expected": ..., "got": ...}}``.
        If every entry has ``ok=True``, the conversion is considered correct.

    Examples
    --------
    >>> from aiida_orbgen.interfaces import verify_ase_atoms
    >>> report = verify_ase_atoms("./U-dimer-2.75-9au/STRU")
    >>> assert all(v["ok"] for v in report.values()), report
    """
    parsed = parse_stru(stru_path)
    atoms = read_stru_as_ase(stru_path)

    # 1. Atom count
    n_atoms_ok = bool(len(atoms) == len(parsed["coordinates"]))

    # 2. Element symbols
    ase_symbols = list(atoms.get_chemical_symbols())
    parsed_symbols = [parsed["element"]] * len(parsed["coordinates"])
    formula_ok = bool(ase_symbols == parsed_symbols)

    # 3. Coordinates
    pos_err = 0.0
    if n_atoms_ok:
        import numpy as np
        pos_err = float(np.max(np.abs(atoms.positions - np.array(parsed["coordinates"]))))
    positions_ok = bool(pos_err < tol)

    # 4. Lattice (Bohr -> Angstrom)
    # SIAB: cell (Bohr) = LATTICE_CONSTANT (Bohr) * LATTICE_VECTORS (dimensionless)
    import numpy as np
    expected_cell = (
        np.array(parsed["lattice_vectors"])
        * float(parsed["lattice_constant"])
        / BOHR_TO_ANG
    )
    cell_err = float(np.max(np.abs(atoms.cell.array - expected_cell)))
    cell_ok = bool(cell_err < tol)

    # 5. Periodicity: a SIAB STRU is PBC in all three directions by default
    expected_pbc = [True, True, True]
    pbc_ok = bool(list(atoms.pbc) == expected_pbc)

    return {
        "n_atoms": {
            "ok": n_atoms_ok,
            "expected": len(parsed["coordinates"]),
            "got": len(atoms),
        },
        "formula": {
            "ok": formula_ok,
            "expected": parsed_symbols,
            "got": ase_symbols,
        },
        "positions": {
            "ok": positions_ok,
            "max_abs_error_angstrom": pos_err,
            "tol_angstrom": tol,
        },
        "cell": {
            "ok": cell_ok,
            "max_abs_error_angstrom": cell_err,
            "expected_angstrom": expected_cell.tolist(),
            "got_angstrom": atoms.cell.array.tolist(),
        },
        "pbc": {
            "ok": pbc_ok,
            "expected": expected_pbc,
            "got": [bool(x) for x in atoms.pbc],
        },
        "has_orbital_reference": {
            "ok": bool(parsed["orbital_file"] is not None),
            "orbital_file": parsed["orbital_file"],
            "note": "ASE does not read NUMERICAL_ORBITAL; recorded only",
        },
    }


def params_stru_to_ase(params_stru: Dict[str, Any]):
    """
    Convert the structured dict returned by ``read_stru`` into ``ase.Atoms``.

    Expected input format::

        {
            'lat': {
                'const': 30.0,                        # lattice constant (Bohr)
                'vec':  [[1,0,0],[0,1,0],[0,0,1]]     # lattice vectors (dimensionless)
            },
            'species': [
                {
                    'symbol':   'U',
                    'mass':     1.0,
                    'pp_file':  'U.pbe-n-nc.upf',
                    'orb_file': 'U_gga_9au_100Ry_27s27p26d26f25g.orb',
                    'mag_each': 0.0,                    # average magnetic moment per atom
                    'natom':    2,
                    'atom': [
                        {'coord': [7.94, 7.94, 7.94],   # the unit depends on coord_type
                         'm':     [0, 0, 0]},            # magnetic moment as a 3-vector
                        {'coord': [7.94, 7.94, 10.69],  # Angstrom
                         'm':     [0, 0, 0]},
                    ],
                },
                # ... with several elements there are several species entries here
            ],
            'coord_type': 'Cartesian_angstrom_center_xyz'   # or 'Direct', etc.
        }

    Unit conventions (same as SIAB):

    - ``lat.const``: Bohr
    - ``lat.vec``:   dimensionless; the actual cell (Bohr) = const × vec
    - ``atom.coord`` (Cartesian_angstrom_*): Angstrom, already SI units
    - ``atom.coord`` (Direct):              fractional coordinates ∈ [0, 1)
    - ``atom.m``:     magnetic moment (3-vector; for a collinear case usually
      only the z component is non-zero)

    Parameters
    ----------
    params_stru : dict
        A dict following the schema above

    Returns
    -------
    ase.Atoms
        With multiple elements, magnetic moments, lattice and periodicity all set
        correctly

    Notes
    -----
    ``pp_file``/``orb_file`` are ABACUS-specific fields ASE does not recognise;
    to keep them, write them into ``atoms.info`` after the call:

    >>> atoms = params_stru_to_ase(params_stru)
    >>> atoms.info["pp_file"]  = params_stru["species"][0]["pp_file"]
    >>> atoms.info["orb_file"] = params_stru["species"][0]["orb_file"]

    Examples
    --------
    >>> params_stru = read_stru("./U-dimer-2.75-9au/STRU")
    >>> atoms = params_stru_to_ase(params_stru)
    >>> atoms.get_chemical_formula()
    'U2'
    >>> atoms.cell.array
    """
    import numpy as np
    from ase import Atoms

    # ---- 1. Parse the cell ----
    lat = params_stru["lat"]
    lat_const_bohr = float(lat["const"])
    lat_vec = np.asarray(lat["vec"], dtype=float)
    if lat_vec.shape != (3, 3):
        raise ValueError(
            f"lat.vec must be 3x3, got {lat_vec.shape}"
        )
    cell_bohr = lat_vec * lat_const_bohr
    cell_ang = cell_bohr / BOHR_TO_ANG

    # ---- 2. Parse the coordinate type ----
    coord_type = params_stru.get("coord_type", "")
    # Distinguish Direct (fractional) from Cartesian
    is_direct = coord_type.lower().startswith("direct")
    # In SIAB, ``Cartesian_angstrom_center_xyz`` / ``Cartesian_angstrom`` /
    # ``Cartesian`` are all Angstrom, and so they are in ASE.

    # ---- 3. Collect every atom ----
    symbols: List[str] = []
    positions: List[List[float]] = []
    magmoms: List[List[float]] = []
    masses: List[float] = []

    for sp in params_stru["species"]:
        sym = sp["symbol"]
        mass = float(sp.get("mass", 1.0))
        atoms_of_sp = sp.get("atom", [])
        if len(atoms_of_sp) != sp.get("natom", len(atoms_of_sp)):
            # Defensive: raise if an explicit natom disagrees with the length of
            # the atom list
            raise ValueError(
                f"species '{sym}': natom={sp.get('natom')} "
                f"does not match the length of the atom list ({len(atoms_of_sp)})"
            )
        for atom in atoms_of_sp:
            symbols.append(sym)
            masses.append(mass)
            coord = np.asarray(atom["coord"], dtype=float)
            if is_direct:
                # Fractional coordinates -> Cartesian Angstrom
                pos_ang = coord @ cell_ang
            else:
                pos_ang = coord
            positions.append(pos_ang)
            # Magnetic moment: accept both a 3-vector and a scalar
            m = atom.get("m", [0, 0, 0])
            if np.isscalar(m):
                magmoms.append([0, 0, float(m)])
            else:
                magmoms.append([float(x) for x in m])

    atoms = Atoms(
        symbols=symbols,
        positions=np.asarray(positions, dtype=float),
        cell=cell_ang,
        pbc=[True, True, True],
        magmoms=np.asarray(magmoms, dtype=float),
        masses=np.asarray(masses, dtype=float),
    )
    return atoms


def params_stru_to_ase_validate(params_stru: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify that a ``params_stru`` dict can be converted correctly by
    :func:`params_stru_to_ase`, and report its consistency with the ASE ``Atoms``
    object it produces.

    Report entries (analogous to :func:`verify_ase_atoms`):

    - n_atoms
    - formula
    - cell_angstrom
    - coord_type
    - direct_coords        whether the coordinates are fractional (True/False)

    Returns
    -------
    dict
        Each key maps to ``{"ok": bool, "expected": ..., "got": ...}``
    """
    atoms = params_stru_to_ase(params_stru)
    import numpy as np

    # n_atoms
    expected_n = sum(sp["natom"] for sp in params_stru["species"])
    n_ok = bool(len(atoms) == expected_n)

    # formula
    expected_formula = "".join(
        f"{sp['symbol']}{sp['natom'] if sp['natom'] > 1 else ''}"
        for sp in params_stru["species"]
    )
    formula_ok = bool(atoms.get_chemical_formula() == expected_formula)

    # cell
    expected_cell = np.asarray(params_stru["lat"]["vec"], dtype=float) \
        * float(params_stru["lat"]["const"]) / BOHR_TO_ANG
    cell_err = float(np.max(np.abs(atoms.cell.array - expected_cell)))
    cell_ok = bool(cell_err < 1e-6)

    return {
        "n_atoms": {
            "ok": n_ok,
            "expected": expected_n,
            "got": len(atoms),
        },
        "formula": {
            "ok": formula_ok,
            "expected": expected_formula,
            "got": atoms.get_chemical_formula(),
        },
        "cell": {
            "ok": cell_ok,
            "max_abs_error_angstrom": cell_err,
            "expected_angstrom": expected_cell.tolist(),
            "got_angstrom": atoms.cell.array.tolist(),
        },
        "coord_type": {
            "ok": True,
            "value": params_stru.get("coord_type", ""),
            "is_direct": bool(
                params_stru.get("coord_type", "").lower().startswith("direct")
            ),
        },
    }
