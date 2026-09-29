"""
Interface tests for aiida_orbgen.interfaces

Tests for the three core interfaces:
1. generate_nsw: generate the raw NSW orbital
2. generate_incar: generate the INPUT file
3. generate_stru: generate the STRU file

How to run:
    cd /home/liguozhou/abacus/calculations/orbgen/aiida_orbgen
    pytest tests/test_interfaces.py -v
    # or
    python -m pytest tests/test_interfaces.py -v
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import pytest

# Add src to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# Add SIAB to path (ABACUS-CSW-NAO library)
#
# ``append`` instead of ``insert(0, ...)``: this machine has two SIAB checkouts, and
# ``/home/liguozhou/install/ABACUS-CSW-NAO`` is not patched with ``simpson``, so under scipy >= 1.12
# ``from scipy.integrate import simps`` raises ImportError immediately. Inserting it at the front would
# shadow the working checkout (``.../orbgen/ABACUS-CSW-NAO`` in the workspace, already changed to
# ``from scipy.integrate import simpson as simps``), and pollute every test that runs later in the
# same process. It is appended as a fallback path, so it takes effect only when no usable SIAB exists.
SIAB_PATH = Path("/home/liguozhou/install/ABACUS-CSW-NAO")
if SIAB_PATH.exists() and str(SIAB_PATH) not in sys.path:
    sys.path.append(str(SIAB_PATH))

# Check whether SIAB is available
try:
    import SIAB  # noqa: F401
    SIAB_AVAILABLE = True
except ImportError:
    SIAB_AVAILABLE = False

# pytest skip decorator
siab_required = pytest.mark.skipif(
    not SIAB_AVAILABLE,
    reason="SIAB library not available. Please install from /home/liguozhou/install/ABACUS-CSW-NAO"
)

from aiida_orbgen.interfaces.nsw import generate_nsw, compute_nbes_per_l
from aiida_orbgen.interfaces.incar import generate_incar, parse_incar
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


# ============================================================================
# Test configuration (the JSON configuration for element U)
# ============================================================================

@pytest.fixture
def u_json_config():
    """JSON configuration for element U (corresponds to project/pbe/pbe_orbgen.json)."""
    return {
        "element": "U",
        "pseudo_dir": "./U.pbe-n-nc.upf",
        "fit_basis": "jy",
        "ecutwfc": 150,
        "ecutjy": 100,
        "bessel_nao_rcut": [9],
        "primitive_type": "reduced",
        "smearing_method": "gauss",
        "smearing_sigma": "0.02",
        "mixing_type": "broyden",
        "mixing_beta": 0.4,
        "mixing_ndim": 12,
        "spill_guess": "atomic",
        "optimizer": "scipy.bfgs",
        "max_steps": 10000,
        "nthreads_rcut": 4,
        "geoms": [
            {
                "proto": "dimer",
                "pertkind": "stretch",
                "pertmags": [1.89, 2.09, 2.75],
                "nbands": 40,
                "nspin": 1,
                "lmaxmax": 4,
            }
        ],
        "orbitals": [
            {
                "nzeta": [1, 1, 1, 1, 0],
                "geoms": [0],
                "nbands": "occ",
                "checkpoint": None,
            },
            {
                "nzeta": [2, 2, 2, 2, 1],
                "geoms": [0],
                "nbands": "occ*2",
                "checkpoint": 0,
            },
        ],
    }


@pytest.fixture
def temp_dir():
    """Temporary directory fixture."""
    tmp = tempfile.mkdtemp(prefix="aiida_orbgen_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 1. NSW interface tests
# ============================================================================

class TestNswInterface:
    """NSW interface tests."""

    @siab_required
    def test_compute_nbes_per_l(self):
        """Test the Bessel function count computation."""
        # Element U: rcut=9, ecut=100, lmaxmax=4
        nbes = compute_nbes_per_l(rcut=9, ecut=100, lmaxmax=4,
                                   primitive_type="reduced")
        assert len(nbes) == 5  # l = 0,1,2,3,4
        # nbes may be numpy.int64, so convert it to a Python int
        nbes_int = [int(n) for n in nbes]
        assert all(isinstance(n, int) for n in nbes_int)
        assert all(n >= 0 for n in nbes_int)
        # The actual values depend on the JLZEROS table
        # Here we only check that the counts are reasonable
        assert sum(nbes_int) > 0

    @siab_required
    def test_compute_nbes_normalized(self):
        """normalized type has one extra function per l."""
        nbes_reduced = compute_nbes_per_l(
            rcut=9, ecut=100, lmaxmax=2, primitive_type="reduced"
        )
        nbes_normalized = compute_nbes_per_l(
            rcut=9, ecut=100, lmaxmax=2, primitive_type="normalized"
        )
        for n_r, n_n in zip(nbes_reduced, nbes_normalized):
            assert n_n == n_r + 1

    def test_generate_nsw_u(self, u_json_config, temp_dir):
        """Test generating the NSW orbital for element U."""
        output_dir = os.path.join(temp_dir, "primitive_jy")
        orb_path = generate_nsw(
            u_json_config,
            output_dir=output_dir,
            lmaxmax=4,
        )

        # Verify that the file exists
        assert os.path.exists(orb_path)

        # Verify the file name
        basename = os.path.basename(orb_path)
        assert basename.startswith("U_gga_9au_100Ry_")
        assert basename.endswith(".orb")

        # Verify that the file is not empty
        assert os.path.getsize(orb_path) > 0

        # Verify the file content
        with open(orb_path, "r") as f:
            content = f.read()
        # It should at least contain the element symbol and the parameters
        assert "9" in content  # rcut
        assert "100" in content  # ecut
        assert "0.01" in content  # dr

    @siab_required
    def test_generate_nsw_filename_pattern(self, u_json_config, temp_dir):
        """Verify the NSW file name pattern."""
        orb_path = generate_nsw(
            u_json_config,
            output_dir=temp_dir,
        )
        basename = os.path.basename(orb_path)
        # Expected: U_gga_9au_100Ry_27s27p26d26f25g.orb
        # The actual counts depend on the JLZEROS table and may differ slightly
        assert basename.startswith("U_gga_9au_100Ry_")
        # Verify the nzeta_str part (matching digit+letter combinations)
        import re
        m = re.match(r"U_gga_9au_100Ry_((?:\d+[spdfghijklmnopqrstuvwxyz]+)+)\.orb", basename)
        assert m is not None, f"Invalid filename: {basename}"


# ============================================================================
# 2. INCAR interface tests
# ============================================================================

class TestIncarInterface:
    """INCAR interface tests."""

    def test_generate_incar_u(self, u_json_config, temp_dir):
        """Test generating the INPUT file for element U."""
        output_path = os.path.join(temp_dir, "U-dimer-1.89-9au", "INPUT")
        input_path = generate_incar(u_json_config, output_path)

        assert os.path.exists(input_path)
        assert os.path.isfile(input_path)

        # Verify the file content
        with open(input_path, "r") as f:
            content = f.read()

        # It must start with INPUT_PARAMETERS
        assert content.startswith("INPUT_PARAMETERS")

        # The key parameters must be present
        assert "ecutwfc" in content
        assert "150" in content  # ecutwfc = 150
        assert "basis_type" in content
        assert "lcao" in content
        assert "nspin" in content
        assert "smearing_method" in content
        assert "gauss" in content
        assert "mixing_type" in content
        assert "broyden" in content
        assert "bessel_nao_rcut" in content
        assert "9" in content
        assert "gamma_only" in content
        assert "out_wfc_lcao" in content
        assert "out_mat_hs" in content

    def test_incar_roundtrip(self, u_json_config, temp_dir):
        """Test INPUT file write/read round-trip consistency."""
        output_path = os.path.join(temp_dir, "INPUT")

        # Write
        generate_incar(u_json_config, output_path)

        # Read back
        params = parse_incar(output_path)

        # Verify the key parameters
        assert str(params.get("ecutwfc")) == "150"
        assert params.get("basis_type") == "lcao"
        assert str(params.get("nspin")) == "1"
        assert params.get("smearing_method") == "gauss"
        assert params.get("mixing_type") == "broyden"
        assert "0.4" in str(params.get("mixing_beta", "0"))
        assert "0.02" in str(params.get("smearing_sigma", "0"))
        assert "9" in str(params.get("bessel_nao_rcut", "0"))

    def test_incar_lcao_default(self, temp_dir):
        """Test the default parameters in LCAO mode."""
        config = {
            "element": "Si",
            "ecutwfc": 60,
            "ecutjy": 40,
            "nspin": 1,
            "basis_type": "lcao",  # lcao mode
            "bessel_nao_rcut": [7],
        }
        output_path = os.path.join(temp_dir, "INPUT")
        generate_incar(config, output_path)

        with open(output_path) as f:
            content = f.read()

        # LCAO mode should add these parameters automatically
        assert "ks_solver" in content
        assert "genelpa" in content
        assert "out_mat_hs" in content
        assert "1 12" in content
        assert "out_wfc_lcao" in content

    def test_incar_no_autoset(self, temp_dir):
        """Test disabling auto-fill."""
        config = {"element": "Si", "ecutwfc": 60}
        output_path = os.path.join(temp_dir, "INPUT")
        generate_incar(config, output_path, auto_set=False)

        with open(output_path) as f:
            content = f.read()

        # There should be no default values
        assert "ks_solver" not in content
        assert "smearing_method" not in content


# ============================================================================
# 3. STRU interface tests
# ============================================================================

class TestStruInterface:
    """STRU interface tests."""

    def test_dft_folder_name_dimer(self):
        """Test the dimer folder name."""
        assert dft_folder_name("U", "dimer", 1.89, rcut=9) == "U-dimer-1.89-9au"
        assert dft_folder_name("Si", "dimer", 2.0, rcut=7) == "Si-dimer-2.00-7au"

    def test_dft_folder_name_monomer(self):
        """Test the monomer folder name (no pert)."""
        assert dft_folder_name("U", "monomer", 0) == "U-monomer"

    def test_dft_folder_name_no_rcut(self):
        """Test the folder name without rcut."""
        assert dft_folder_name("U", "dimer", 1.89) == "U-dimer-1.89"

    def test_atom_coords_dimer(self):
        """Test the dimer atomic coordinates."""
        coords = generate_atom_coords("dimer", bond_length=1.89, lattice_constant=30.0)
        assert len(coords) == 2
        # The first atom is at (shift, shift, shift)
        # The second atom is at (shift, shift, bond_length + shift)
        shift = 30.0 / 2 / 1.8897259886
        assert abs(coords[0][0] - shift) < 1e-6
        assert abs(coords[0][1] - shift) < 1e-6
        assert abs(coords[0][2] - shift) < 1e-6
        assert abs(coords[1][2] - 1.89 - shift) < 1e-6

    def test_atom_coords_monomer(self):
        """Test the monomer atomic coordinates."""
        coords = generate_atom_coords("monomer", bond_length=0.0, lattice_constant=30.0)
        assert len(coords) == 1

    def test_atom_coords_trimer(self):
        """Test the trimer atomic coordinates."""
        coords = generate_atom_coords("trimer", bond_length=2.0, lattice_constant=30.0)
        assert len(coords) == 3

    def test_atom_coords_octahedron(self):
        """Test the octahedron atomic coordinates."""
        coords = generate_atom_coords("octahedron", bond_length=2.0, lattice_constant=30.0)
        assert len(coords) == 6

    def test_atom_coords_cube(self):
        """Test the cube atomic coordinates."""
        coords = generate_atom_coords("cube", bond_length=2.0, lattice_constant=30.0)
        assert len(coords) == 8

    def test_atom_coords_invalid_proto(self):
        """Test an invalid proto."""
        with pytest.raises(ValueError):
            generate_atom_coords("invalid_proto", 2.0, 30.0)

    def test_generate_stru_u(self, u_json_config, temp_dir):
        """Test generating the STRU file for element U."""
        output_path = os.path.join(temp_dir, "U-dimer-1.89-9au", "STRU")
        orb_filename = "U_gga_9au_100Ry_27s27p26d26f25g.orb"
        stru_path = generate_stru(
            u_json_config,
            output_path=output_path,
            bond_length=1.89,
            orb_filename=orb_filename,
        )

        assert os.path.exists(stru_path)

        # Verify the content
        with open(stru_path, "r") as f:
            content = f.read()

        # Sections that must be present
        assert "ATOMIC_SPECIES" in content
        assert "U" in content
        assert "U.pbe-n-nc.upf" in content
        assert "NUMERICAL_ORBITAL" in content
        assert orb_filename in content
        assert "LATTICE_CONSTANT" in content
        assert "LATTICE_VECTORS" in content
        assert "ATOMIC_POSITIONS" in content
        assert "Cartesian_angstrom_center_xyz" in content

        # Verify the atomic coordinates: first atom (shift, shift, shift), second (shift, shift, bond_length+shift)
        # shift = 30.0 / 2 / 1.8897259886 = 7.93765873
        # The z coordinate of the second atom = 1.89 + 7.93765873 = 9.82765873
        shift = 30.0 / 2 / 1.8897259886
        expected_z = 1.89 + shift
        assert f"{expected_z:.8f}" in content

    def test_generate_stru_monomer(self, u_json_config, temp_dir):
        """Test generating the monomer STRU."""
        u_json_config["geoms"][0]["proto"] = "monomer"
        u_json_config["geoms"][0]["pertmags"] = [0.0]

        output_path = os.path.join(temp_dir, "U-monomer", "STRU")
        stru_path = generate_stru(
            u_json_config,
            output_path=output_path,
            bond_length=0.0,
        )

        assert os.path.exists(stru_path)
        with open(stru_path, "r") as f:
            content = f.read()
        assert "ATOMIC_POSITIONS" in content
        # monomer has only 1 atom
        assert "1       //number of atoms" in content

    def test_generate_stru_nspin2(self, u_json_config, temp_dir):
        """Test the starting magnetization for nspin=2."""
        u_json_config["geoms"][0]["nspin"] = 2
        output_path = os.path.join(temp_dir, "U-dimer-1.89-9au", "STRU")
        generate_stru(u_json_config, output_path, bond_length=1.89)

        with open(output_path, "r") as f:
            content = f.read()
        # with nspin=2, starting_magnetization = 2.0
        assert "2.00" in content


# ============================================================================
# 3.5 ASE conversion tests
# ============================================================================

class TestAseConversion:
    """read_stru_as_ase + verify_ase_atoms tests."""

    def test_read_stru_as_ase_dimer(self, u_json_config, temp_dir):
        """Read the dimer STRU and verify that the ASE conversion is correct."""
        from ase.io import read as ase_read
        from ase import Atoms

        # First generate the STRU
        output_path = os.path.join(temp_dir, "U-dimer-2.75-9au", "STRU")
        generate_stru(
            u_json_config,
            output_path=output_path,
            bond_length=2.75,
            orb_filename="U_gga_9au_100Ry_27s27p26d26f25g.orb",
        )

        # 1. Reading it directly with ASE should fail
        with pytest.raises(AssertionError):
            ase_read(output_path, format="abacus")

        # 2. Reading it with our read_stru_as_ase should succeed
        atoms = read_stru_as_ase(output_path)
        assert isinstance(atoms, Atoms)
        assert atoms.get_chemical_formula() == "U2"
        assert len(atoms) == 2
        assert all(atoms.pbc)

        # 3. Verify the positions: the first atom is near (7.9377, 7.9377, 7.9377)
        import numpy as np
        shift = 30.0 / 2 / 1.8897259886
        np.testing.assert_allclose(
            atoms.positions[0], [shift, shift, shift], atol=1e-6
        )
        # The second atom: z = shift + 2.75
        np.testing.assert_allclose(
            atoms.positions[1], [shift, shift, shift + 2.75], atol=1e-6
        )

        # 4. Verify the cell: 15.875 Å (30 Bohr / 1.8897)
        cell_ang = 30.0 / 1.8897259886
        np.testing.assert_allclose(
            atoms.cell.array, np.eye(3) * cell_ang, atol=1e-6
        )

    def test_verify_ase_atoms_dimer(self, u_json_config, temp_dir):
        """verify_ase_atoms report should be all ok."""
        output_path = os.path.join(temp_dir, "U-dimer-2.75-9au", "STRU")
        generate_stru(
            u_json_config,
            output_path=output_path,
            bond_length=2.75,
            orb_filename="U_gga_9au_100Ry_27s27p26d26f25g.orb",
        )
        report = verify_ase_atoms(output_path)
        for name, info in report.items():
            if isinstance(info, dict) and "ok" in info:
                assert info["ok"], f"{name} failed: {info}"
        assert report["n_atoms"]["got"] == 2
        assert report["formula"]["got"] == ["U", "U"]
        assert report["positions"]["max_abs_error_angstrom"] < 1e-6
        assert report["cell"]["max_abs_error_angstrom"] < 1e-6
        assert report["pbc"]["got"] == [True, True, True]
        assert report["has_orbital_reference"]["orbital_file"] == \
            "U_gga_9au_100Ry_27s27p26d26f25g.orb"

    def test_read_stru_as_ase_monomer(self, u_json_config, temp_dir):
        """Read the monomer STRU: 1 atom."""
        u_json_config["geoms"][0]["proto"] = "monomer"
        u_json_config["geoms"][0]["pertmags"] = [0.0]
        output_path = os.path.join(temp_dir, "U-monomer-9au", "STRU")
        generate_stru(u_json_config, output_path, bond_length=0.0)

        atoms = read_stru_as_ase(output_path)
        assert len(atoms) == 1
        assert atoms.get_chemical_formula() == "U"

        report = verify_ase_atoms(output_path)
        assert report["n_atoms"]["ok"] is True
        assert report["positions"]["ok"] is True
        assert report["cell"]["ok"] is True


class TestParamsStruToAse:
    """params_stru dict → ase.Atoms conversion tests."""

    def _make_params_stru(self, bond_length=2.75, nspin=1):
        """Build a dimer-style params_stru."""
        mag_each = 0.0 if nspin == 1 else 2.0
        m_vec = [0, 0, 0] if nspin == 1 else [0, 0, mag_each]
        return {
            "lat": {
                "const": 30.0,
                "vec": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            "species": [{
                "symbol": "U",
                "mass": 1.0,
                "pp_file": "U.pbe-n-nc.upf",
                "orb_file": "U_gga_9au_100Ry_27s27p26d26f25g.orb",
                "mag_each": mag_each,
                "natom": 2,
                "atom": [
                    {"coord": [7.93765873, 7.93765873, 7.93765873], "m": m_vec},
                    {"coord": [7.93765873, 7.93765873,
                               7.93765873 + bond_length], "m": m_vec},
                ],
            }],
            "coord_type": "Cartesian_angstrom_center_xyz",
        }

    def test_basic_conversion(self):
        """Basic conversion: U-dimer."""
        atoms = params_stru_to_ase(self._make_params_stru(2.75))
        assert atoms.get_chemical_formula() == "U2"
        assert len(atoms) == 2
        assert all(atoms.pbc)
        import numpy as np
        np.testing.assert_allclose(
            atoms.cell.array, np.eye(3) * 15.875317469822937, atol=1e-8
        )
        # Positions (Angstrom)
        assert atoms.positions[0, 2] == 7.93765873
        assert abs(atoms.positions[1, 2] - 10.68765873) < 1e-6

    def test_magnetic_moments_collinear(self):
        """Collinear magnetic moments: m = [0, 0, 2.0] kept in 3-vector form."""
        params_stru = self._make_params_stru(nspin=2)
        atoms = params_stru_to_ase(params_stru)
        m = atoms.get_initial_magnetic_moments()
        # ASE keeps the full 3-vector (it does not collapse to the z component)
        import numpy as np
        np.testing.assert_allclose(m, [[0, 0, 2.0], [0, 0, 2.0]], atol=1e-8)

    def test_masses_preserved(self):
        """Masses are transferred correctly."""
        params_stru = self._make_params_stru()
        params_stru["species"][0]["mass"] = 238.03
        atoms = params_stru_to_ase(params_stru)
        import numpy as np
        np.testing.assert_allclose(atoms.get_masses(), [238.03, 238.03])

    def test_multi_species(self):
        """Multiple species: e.g. U-O."""
        params_stru = self._make_params_stru()
        # Append an O species at the end
        params_stru["species"].append({
            "symbol": "O",
            "mass": 15.999,
            "pp_file": "O.upf",
            "orb_file": "O_gga_9au_100Ry_1s1p.orb",
            "mag_each": 0.0,
            "natom": 1,
            "atom": [
                {"coord": [10.0, 10.0, 10.0], "m": [0, 0, 0]},
            ],
        })
        atoms = params_stru_to_ase(params_stru)
        # Check by element counts (independent of the ASE formula output order)
        from collections import Counter
        assert Counter(atoms.get_chemical_symbols()) == Counter(["U", "U", "O"])
        assert len(atoms) == 3
        assert atoms.get_chemical_formula() in ("UUO", "OU2")  # ASE order is not fixed

    def test_direct_coords_conversion(self):
        """Direct fractional coordinates -> Cartesian."""
        params_stru = self._make_params_stru()
        params_stru["coord_type"] = "Direct"
        # Convert Cartesian to Direct (0.5, 0.5, shift/cell)
        cell_ang = 15.875317469822937
        z1 = 7.93765873 / cell_ang
        z2 = 10.68765873 / cell_ang
        params_stru["species"][0]["atom"] = [
            {"coord": [0.5, 0.5, z1], "m": [0, 0, 0]},
            {"coord": [0.5, 0.5, z2], "m": [0, 0, 0]},
        ]
        atoms = params_stru_to_ase(params_stru)
        import numpy as np
        np.testing.assert_allclose(atoms.positions[0], [7.93765873]*3, atol=1e-6)
        np.testing.assert_allclose(atoms.positions[1],
                                   [7.93765873, 7.93765873, 10.68765873], atol=1e-6)

    def test_validate_report(self):
        """The validation report should be all ok."""
        report = params_stru_to_ase_validate(self._make_params_stru())
        for name, info in report.items():
            if isinstance(info, dict) and "ok" in info:
                assert info["ok"], f"{name} failed: {info}"
        assert report["n_atoms"]["got"] == 2
        assert report["formula"]["got"] == "U2"
        assert report["cell"]["max_abs_error_angstrom"] < 1e-6
        assert report["coord_type"]["is_direct"] is False

    def test_roundtrip_with_stru_file(self, u_json_config, temp_dir):
        """For a generated STRU file, parse → params_stru_to_ase should be equivalent to read_stru_as_ase."""
        output_path = os.path.join(temp_dir, "U-dimer-2.75-9au", "STRU")
        generate_stru(
            u_json_config, output_path, bond_length=2.75,
            orb_filename="U_gga_9au_100Ry_27s27p26d26f25g.orb",
        )

        # Method 1: read_stru_as_ase
        atoms1 = read_stru_as_ase(output_path)

        # Method 2: parse_stru -> build params_stru -> params_stru_to_ase
        parsed = parse_stru(output_path)
        params_stru = {
            "lat": {"const": parsed["lattice_constant"],
                    "vec": parsed["lattice_vectors"]},
            "species": [{
                "symbol": parsed["element"], "mass": 1.0,
                "pp_file": parsed["pseudo_file"],
                "orb_file": parsed["orbital_file"],
                "mag_each": 0.0, "natom": len(parsed["coordinates"]),
                "atom": [{"coord": c, "m": [0, 0, 0]}
                          for c in parsed["coordinates"]],
            }],
            "coord_type": "Cartesian_angstrom_center_xyz",
        }
        atoms2 = params_stru_to_ase(params_stru)

        import numpy as np
        np.testing.assert_allclose(atoms1.positions, atoms2.positions, atol=1e-8)
        np.testing.assert_allclose(atoms1.cell.array, atoms2.cell.array, atol=1e-8)
        assert atoms1.get_chemical_formula() == atoms2.get_chemical_formula()


# ============================================================================
# 4. Integration test - full workflow
# ============================================================================

class TestIntegration:
    """Full workflow tests."""

    @siab_required
    def test_full_workflow(self, u_json_config, temp_dir):
        """Test generating the NSW, INPUT and STRU files together.

        Simulate the file set that SIAB generates:
        - U_gga_9au_100Ry_*s*p*d*f*g.orb (NSW)
        - U-dimer-1.89-9au/INPUT
        - U-dimer-1.89-9au/STRU
        """
        # 1. Generate NSW
        nsw_path = generate_nsw(
            u_json_config,
            output_dir=os.path.join(temp_dir, "primitive_jy"),
            lmaxmax=4,
        )
        assert os.path.exists(nsw_path)

        # 2. Generate INPUT
        job_folder = os.path.join(temp_dir, "U-dimer-1.89-9au")
        os.makedirs(job_folder, exist_ok=True)
        input_path = generate_incar(
            u_json_config,
            output_path=os.path.join(job_folder, "INPUT"),
        )
        assert os.path.exists(input_path)

        # 3. Generate STRU
        stru_path = generate_stru(
            u_json_config,
            output_path=os.path.join(job_folder, "STRU"),
            bond_length=1.89,
            orb_filename=os.path.basename(nsw_path),
        )
        assert os.path.exists(stru_path)

        # Verify the file structure
        assert os.path.isfile(nsw_path)
        assert os.path.isfile(input_path)
        assert os.path.isfile(stru_path)

        # Verify that the STRU references the NSW orbital
        with open(stru_path) as f:
            stru_content = f.read()
        assert os.path.basename(nsw_path) in stru_content


# ============================================================================
# 5. Error handling tests
# ============================================================================

class TestErrorHandling:
    """Error handling tests."""

    def test_missing_element(self, temp_dir):
        """Test a missing element field."""
        config = {"ecutjy": 100, "bessel_nao_rcut": [9]}
        with pytest.raises(KeyError):
            generate_nsw(config, output_dir=temp_dir)

    def test_missing_ecutjy(self, temp_dir):
        """Test a missing ecutjy field."""
        config = {"element": "U", "bessel_nao_rcut": [9]}
        with pytest.raises(KeyError):
            generate_nsw(config, output_dir=temp_dir)

    def test_invalid_proto(self):
        """Test an invalid proto."""
        with pytest.raises(ValueError):
            generate_atom_coords("invalid", 2.0, 30.0)

    def test_parse_nonexistent_file(self):
        """Test reading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            parse_incar("/nonexistent/path/INPUT")


# ============================================================================
# 6. Performance/smoke tests
# ============================================================================

class TestSmoke:
    """Simple smoke tests - ensure the interfaces do not raise."""

    @siab_required
    def test_smoke_nsw(self, u_json_config, temp_dir):
        """Quick check of the NSW interface."""
        try:
            generate_nsw(u_json_config, output_dir=temp_dir, lmaxmax=2)
            assert True
        except Exception as e:
            pytest.fail(f"NSW generation failed: {e}")

    def test_smoke_incar(self, u_json_config, temp_dir):
        """Quick check of the INCAR interface."""
        try:
            generate_incar(u_json_config, os.path.join(temp_dir, "INPUT"))
            assert True
        except Exception as e:
            pytest.fail(f"INCAR generation failed: {e}")

    def test_smoke_stru(self, u_json_config, temp_dir):
        """Quick check of the STRU interface."""
        try:
            generate_stru(
                u_json_config,
                os.path.join(temp_dir, "STRU"),
                bond_length=1.89,
            )
            assert True
        except Exception as e:
            pytest.fail(f"STRU generation failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
