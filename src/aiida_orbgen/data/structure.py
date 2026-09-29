"""
StructureData: convert an ABACUS STRU file into an ase.Atoms object

This module converts a SIAB-style STRU file into an ase.Atoms instance.

ABACUS STRU file format:
    ATOMIC_SPECIES
    <elem> <mass> <pseudo_path>

    [NUMERICAL_ORBITAL]
    <orb_path>

    LATTICE_CONSTANT
    <latconst>  // add lattice constant(a.u.)

    LATTICE_VECTORS
    <v1x> <v1y> <v1z>
    <v2x> <v2y> <v2z>
    <v3x> <v3y> <v3z>

    ATOMIC_POSITIONS
    <coordinate_type>  // Cartesian_angstrom_center_xyz, Direct, etc.
    <elem_label>
    <magnetization>
    <natoms>
    <x> <y> <z> <m1 m2 m3>  // for each atom

Data flow:
    STRU file → parse_stru() → ase.Atoms (with cell, positions, symbols, etc.)
"""

import os
import re
import math
from typing import Optional, List, Dict, Tuple, Union

import numpy as np

# Try to import ase
try:
    from ase import Atoms
    from ase.io import read as ase_read
    ASE_AVAILABLE = True
except ImportError:
    ASE_AVAILABLE = False
    Atoms = None  # type: ignore


__all__ = [
    "StructureData",
    "parse_stru_to_ase",
    "stru_to_ase",
]


def check_ase():
    """Check whether ase is available."""
    if not ASE_AVAILABLE:
        raise ImportError(
            "ase is not available. "
            "Please install it via: pip install ase"
        )


def parse_stru_to_ase(stru_path: str) -> "Atoms":
    """
    Parse a STRU file and return an ase.Atoms object.

    Parameters
    ----------
    stru_path : str
        Path to the STRU file

    Returns
    -------
    ase.Atoms
        Containing atoms, cell, positions and related information

    Examples
    --------
    >>> atoms = parse_stru_to_ase("./U-dimer-1.89-9au/STRU")
    >>> print(atoms)
    Atoms(symbols='U2', pbc=True, cell=[30, 30, 30])
    >>> print(atoms.positions)
    [[ 7.93765873  7.93765873  7.93765873]
     [ 7.93765873  7.93765873  9.82765873]]
    """
    check_ase()

    if not os.path.isfile(stru_path):
        raise FileNotFoundError(f"STRU file not found: {stru_path}")

    with open(stru_path, "r") as f:
        content = f.read()

    # Parse section by section
    sections = _split_sections(content)

    # 1. Parse ATOMIC_SPECIES
    species_info = _parse_atomic_species(sections.get("ATOMIC_SPECIES", ""))
    element = species_info["element"]
    mass = species_info["mass"]

    # 2. Parse NUMERICAL_ORBITAL (optional)
    orb_path = None
    if "NUMERICAL_ORBITAL" in sections:
        orb_path = _parse_numerical_orbital(sections["NUMERICAL_ORBITAL"])

    # 3. Parse LATTICE_CONSTANT (Bohr)
    latconst = _parse_lattice_constant(sections.get("LATTICE_CONSTANT", ""))

    # 4. Parse LATTICE_VECTORS
    lattice_vectors = _parse_lattice_vectors(sections.get("LATTICE_VECTORS", ""))

    # 5. Parse ATOMIC_POSITIONS
    positions, magnetization = _parse_atomic_positions(
        sections.get("ATOMIC_POSITIONS", ""),
        latconst,
        lattice_vectors,
    )

    # Build the cell (in Angstrom, for ase compatibility)
    cell_angstrom = np.array(lattice_vectors) * latconst

    # Build the ase.Atoms
    atoms = Atoms(
        symbols=[element] * len(positions),
        positions=positions,
        cell=cell_angstrom,
        pbc=True,
    )
    atoms.set_masses([mass] * len(positions))

    # Set the magnetization (if any)
    if magnetization is not None and len(positions) > 0:
        # In ase each atom can carry a magnetic moment
        magmoms = [float(magnetization)] * len(positions)
        atoms.set_initial_magnetic_moments(magmoms)

    # Store extra information
    atoms.info["element"] = element
    atoms.info["lattice_constant_bohr"] = latconst
    if orb_path is not None:
        atoms.info["numerical_orbital"] = orb_path
    if magnetization is not None:
        atoms.info["magnetization"] = float(magnetization)

    return atoms


def stru_to_ase(stru_path: str) -> "Atoms":
    """Alias for ``parse_stru_to_ase``."""
    return parse_stru_to_ase(stru_path)


class StructureData:
    """
    AiiDA structure data node wrapper.

    Internally represented as an ase.Atoms object.
    Supports conversion between a STRU file and ase.Atoms.

    Examples
    --------
    >>> sd = StructureData("./U-dimer-1.89-9au/STRU")
    >>> atoms = sd.to_ase()
    >>> print(atoms)
    """

    def __init__(self, stru_path: Optional[str] = None, atoms: Optional["Atoms"] = None):
        """
        Initialise StructureData.

        Parameters
        ----------
        stru_path : str, optional
            Path to the STRU file
        atoms : ase.Atoms, optional
            An ase.Atoms object
        """
        check_ase()

        if stru_path is not None:
            self.atoms = parse_stru_to_ase(stru_path)
            self.stru_path = stru_path
        elif atoms is not None:
            self.atoms = atoms
            self.stru_path = None
        else:
            raise ValueError("Either stru_path or atoms must be provided")

    @classmethod
    def from_stru(cls, stru_path: str) -> "StructureData":
        """Create from a STRU file."""
        return cls(stru_path=stru_path)

    @classmethod
    def from_ase(cls, atoms: "Atoms") -> "StructureData":
        """Create from an ase.Atoms."""
        return cls(atoms=atoms)

    def to_ase(self) -> "Atoms":
        """Return the ase.Atoms object."""
        return self.atoms

    def to_stru(self, output_path: str) -> str:
        """
        Convert ase.Atoms back to the STRU format.

        Parameters
        ----------
        output_path : str
            Output path for the STRU file

        Returns
        -------
        str
            Absolute path of the output file
        """
        # Extract information
        element = self.atoms.get_chemical_symbols()[0]
        mass = self.atoms.get_masses()[0]

        # Lattice (assumed cubic)
        cell = self.atoms.get_cell()
        # Use the average diagonal element as lattice_constant
        avg_latconst = np.mean(np.diag(cell))  # Angstrom
        latconst_bohr = avg_latconst / 1.8897259886  # Bohr

        # Lattice vectors
        lattice_vectors = cell.array / avg_latconst  # normalised

        # Positions
        positions = self.atoms.get_positions()  # Angstrom
        # Convert to fractional (relative to the normalised cell)
        fractional = self.atoms.get_scaled_positions()

        # Magnetization
        magmom = self.atoms.get_initial_magnetic_moments()
        magnetization = magmom[0] if len(magmom) > 0 else 0.0

        # Generate the STRU
        lines = []
        lines.append("ATOMIC_SPECIES")
        lines.append(f"{element} {mass:.6f} ./pseudo.upf")
        lines.append("")
        if "numerical_orbital" in self.atoms.info:
            lines.append("NUMERICAL_ORBITAL")
            lines.append(self.atoms.info["numerical_orbital"])
            lines.append("")
        lines.append("LATTICE_CONSTANT")
        lines.append(f"{latconst_bohr:.6f}  // add lattice constant(a.u.)")
        lines.append("LATTICE_VECTORS")
        for vec in lattice_vectors:
            lines.append(f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}")
        lines.append("ATOMIC_POSITIONS")
        lines.append("Cartesian_angstrom_center_xyz  //Cartesian or Direct coordinate.")
        lines.append(f"{element}      //Element Label")
        lines.append(f"{magnetization:.2f}     //starting magnetism")
        lines.append(f"{len(positions)}       //number of atoms")
        for pos in positions:
            # Subtract the center shift
            lines.append(
                f"{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} 0 0 0"
            )

        with open(output_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        return os.path.abspath(output_path)

    @property
    def element(self) -> str:
        return self.atoms.get_chemical_symbols()[0]

    @property
    def natoms(self) -> int:
        return len(self.atoms)

    @property
    def cell(self) -> np.ndarray:
        return self.atoms.get_cell().array

    @property
    def positions(self) -> np.ndarray:
        return self.atoms.get_positions()

    def __repr__(self) -> str:
        return f"StructureData({self.atoms})"


# ============================================================================
# Internal helper functions
# ============================================================================

def _split_sections(content: str) -> Dict[str, str]:
    """
    Split the content of a STRU file into sections.

    Returns
    -------
    dict
        {section_name: section_content}
    """
    section_names = [
        "ATOMIC_SPECIES", "NUMERICAL_ORBITAL", "LATTICE_CONSTANT",
        "LATTICE_VECTORS", "ATOMIC_POSITIONS",
    ]

    sections = {}
    lines = content.splitlines()

    current_section = None
    current_lines: List[str] = []

    for line in lines:
        line_stripped = line.strip()
        # Skip comments
        if "//" in line:
            line_stripped = line_stripped.split("//")[0].strip()
        if not line_stripped:
            continue

        # Check whether a new section starts here
        matched_section = None
        for name in section_names:
            if line_stripped == name:
                matched_section = name
                break

        if matched_section:
            # Store the previous section
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines)
            current_section = matched_section
            current_lines = []
        else:
            if current_section is not None:
                current_lines.append(line)

    # Store the last section
    if current_section is not None:
        sections[current_section] = "\n".join(current_lines)

    return sections


def _parse_atomic_species(content: str) -> dict:
    """
    Parse the ATOMIC_SPECIES section.

    Format: <elem> <mass> <pseudo_path>
    """
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        raise ValueError("ATOMIC_SPECIES section is empty")
    first_line = lines[0].split("//")[0].strip()
    tokens = first_line.split()
    return {
        "element": tokens[0],
        "mass": float(tokens[1]),
        "pseudo_path": tokens[2] if len(tokens) > 2 else None,
    }


def _parse_numerical_orbital(content: str) -> str:
    """
    Parse the NUMERICAL_ORBITAL section.

    Format: <orb_path>
    """
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        return None
    return lines[0]


def _parse_lattice_constant(content: str) -> float:
    """
    Parse the LATTICE_CONSTANT section (returned in Bohr).

    Format: <latconst>  // add lattice constant(a.u.)
    """
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if not lines:
        raise ValueError("LATTICE_CONSTANT section is empty")
    # Strip the // comment
    first_line = lines[0].split("//")[0].strip()
    return float(first_line)


def _parse_lattice_vectors(content: str) -> List[List[float]]:
    """
    Parse the LATTICE_VECTORS section.

    Format: three lines, three numbers each
    """
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    if len(lines) < 3:
        raise ValueError(f"LATTICE_VECTORS needs 3 lines, got {len(lines)}")
    vectors = []
    for line in lines[:3]:
        tokens = line.split()
        vectors.append([float(t) for t in tokens[:3]])
    return vectors


def _parse_atomic_positions(
    content: str,
    latconst: float,
    lattice_vectors: List[List[float]],
) -> Tuple[np.ndarray, Optional[float]]:
    """
    Parse the ATOMIC_POSITIONS section.

    Format:
        <coord_type>  // Cartesian_angstrom_center_xyz, Direct, etc.
        <elem_label>
        <magnetization>
        <natoms>
        <x> <y> <z> <m1 m2 m3>  // for each atom
    """
    # Strip the // comments
    lines = []
    for line in content.splitlines():
        # Strip everything after //
        if "//" in line:
            line = line.split("//")[0]
        line = line.strip()
        if line:
            lines.append(line)

    if len(lines) < 4:
        raise ValueError("ATOMIC_POSITIONS section too short")

    coord_type = lines[0].split()[0] if lines[0] else "Cartesian_angstrom"
    elem_label = lines[1] if len(lines) > 1 else None
    magnetization = float(lines[2]) if len(lines) > 2 else 0.0
    natoms = int(lines[3]) if len(lines) > 3 else 0

    positions = []
    shift = latconst / 2  # Bohr (Cartesian_angstrom_center_xyz)

    for i in range(natoms):
        idx = 4 + i
        if idx >= len(lines):
            break
        tokens = lines[idx].split()
        if len(tokens) < 3:
            continue
        x, y, z = float(tokens[0]), float(tokens[1]), float(tokens[2])

        if "Cartesian" in coord_type:
            # Already in Angstrom, but the center shift still has to be removed
            # shift = latconst / 2 / 1.8897259886 (Angstrom)
            shift_angstrom = shift / 1.8897259886
            x -= shift_angstrom
            y -= shift_angstrom
            z -= shift_angstrom
            positions.append([x, y, z])
        elif "Direct" in coord_type:
            # Direct coordinates: fractional, converted to Cartesian
            # fractional coordinates * cell = Cartesian coordinates
            frac = np.array([x, y, z])
            cell = np.array(lattice_vectors) * latconst
            cart = frac @ cell
            positions.append(cart.tolist())
        else:
            # Default to Cartesian (Angstrom)
            positions.append([x, y, z])

    return np.array(positions), magnetization
