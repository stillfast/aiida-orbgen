"""
测试 aiida_orbgen.interfaces 的接口

测试三个核心接口：
1. generate_nsw: 生成 NSW 原始轨道
2. generate_incar: 生成 INPUT 文件
3. generate_stru: 生成 STRU 文件

运行方式：
    cd /home/liguozhou/abacus/calculations/orbgen/aiida_orbgen
    pytest tests/test_interfaces.py -v
    # 或
    python -m pytest tests/test_interfaces.py -v
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import pytest

# 添加 src 到 path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# 添加 SIAB 到 path (ABACUS-CSW-NAO 库)
#
# ``append`` 而不是 ``insert(0, ...)``: 这台机器上有两份 SIAB 检出, 其中
# ``/home/liguozhou/install/ABACUS-CSW-NAO`` 未打 ``simpson`` 补丁, 在 scipy >= 1.12
# 的环境里 ``from scipy.integrate import simps`` 直接 ImportError. 插到最前面会
# 遮蔽正常的那份 (workspace 里的 ``.../orbgen/ABACUS-CSW-NAO``, 已改成
# ``from scipy.integrate import simpson as simps``), 并且污染同一进程里之后跑的
# 所有测试. 作为后备路径放在最后, 只在没有可用 SIAB 时才生效.
SIAB_PATH = Path("/home/liguozhou/install/ABACUS-CSW-NAO")
if SIAB_PATH.exists() and str(SIAB_PATH) not in sys.path:
    sys.path.append(str(SIAB_PATH))

# 检测 SIAB 是否可用
try:
    import SIAB  # noqa: F401
    SIAB_AVAILABLE = True
except ImportError:
    SIAB_AVAILABLE = False

# pytest skip 装饰器
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
# 测试配置 (对应 U 元素的 JSON 配置)
# ============================================================================

@pytest.fixture
def u_json_config():
    """U 元素的 JSON 配置 (对应 project/pbe/pbe_orbgen.json)."""
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
    """临时目录 fixture."""
    tmp = tempfile.mkdtemp(prefix="aiida_orbgen_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# 1. NSW 接口测试
# ============================================================================

class TestNswInterface:
    """NSW 接口测试."""

    @siab_required
    def test_compute_nbes_per_l(self):
        """测试 Bessel 函数数量计算."""
        # U 元素: rcut=9, ecut=100, lmaxmax=4
        nbes = compute_nbes_per_l(rcut=9, ecut=100, lmaxmax=4,
                                   primitive_type="reduced")
        assert len(nbes) == 5  # l = 0,1,2,3,4
        # nbes 可以是 numpy.int64, 转换为 Python int
        nbes_int = [int(n) for n in nbes]
        assert all(isinstance(n, int) for n in nbes_int)
        assert all(n >= 0 for n in nbes_int)
        # 实际值取决于 JLZEROS 表
        # 这里只验证数量是合理的
        assert sum(nbes_int) > 0

    @siab_required
    def test_compute_nbes_normalized(self):
        """normalized 类型每个 l 多 1."""
        nbes_reduced = compute_nbes_per_l(
            rcut=9, ecut=100, lmaxmax=2, primitive_type="reduced"
        )
        nbes_normalized = compute_nbes_per_l(
            rcut=9, ecut=100, lmaxmax=2, primitive_type="normalized"
        )
        for n_r, n_n in zip(nbes_reduced, nbes_normalized):
            assert n_n == n_r + 1

    def test_generate_nsw_u(self, u_json_config, temp_dir):
        """测试生成 U 元素的 NSW 轨道."""
        output_dir = os.path.join(temp_dir, "primitive_jy")
        orb_path = generate_nsw(
            u_json_config,
            output_dir=output_dir,
            lmaxmax=4,
        )

        # 验证文件存在
        assert os.path.exists(orb_path)

        # 验证文件名
        basename = os.path.basename(orb_path)
        assert basename.startswith("U_gga_9au_100Ry_")
        assert basename.endswith(".orb")

        # 验证文件非空
        assert os.path.getsize(orb_path) > 0

        # 验证文件内容
        with open(orb_path, "r") as f:
            content = f.read()
        # 至少应包含元素符号和参数
        assert "9" in content  # rcut
        assert "100" in content  # ecut
        assert "0.01" in content  # dr

    @siab_required
    def test_generate_nsw_filename_pattern(self, u_json_config, temp_dir):
        """验证 NSW 文件名格式."""
        orb_path = generate_nsw(
            u_json_config,
            output_dir=temp_dir,
        )
        basename = os.path.basename(orb_path)
        # 期望: U_gga_9au_100Ry_27s27p26d26f25g.orb
        # 实际数量取决于 JLZEROS 表，可能略有不同
        assert basename.startswith("U_gga_9au_100Ry_")
        # 验证 nzeta_str 部分 (匹配数字+字母的组合)
        import re
        m = re.match(r"U_gga_9au_100Ry_((?:\d+[spdfghijklmnopqrstuvwxyz]+)+)\.orb", basename)
        assert m is not None, f"Invalid filename: {basename}"


# ============================================================================
# 2. INCAR 接口测试
# ============================================================================

class TestIncarInterface:
    """INCAR 接口测试."""

    def test_generate_incar_u(self, u_json_config, temp_dir):
        """测试生成 U 元素的 INPUT 文件."""
        output_path = os.path.join(temp_dir, "U-dimer-1.89-9au", "INPUT")
        input_path = generate_incar(u_json_config, output_path)

        assert os.path.exists(input_path)
        assert os.path.isfile(input_path)

        # 验证文件内容
        with open(input_path, "r") as f:
            content = f.read()

        # 必须以 INPUT_PARAMETERS 开头
        assert content.startswith("INPUT_PARAMETERS")

        # 关键参数必须存在
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
        """测试 INPUT 文件读写一致性."""
        output_path = os.path.join(temp_dir, "INPUT")

        # 写入
        generate_incar(u_json_config, output_path)

        # 读回
        params = parse_incar(output_path)

        # 验证关键参数
        assert str(params.get("ecutwfc")) == "150"
        assert params.get("basis_type") == "lcao"
        assert str(params.get("nspin")) == "1"
        assert params.get("smearing_method") == "gauss"
        assert params.get("mixing_type") == "broyden"
        assert "0.4" in str(params.get("mixing_beta", "0"))
        assert "0.02" in str(params.get("smearing_sigma", "0"))
        assert "9" in str(params.get("bessel_nao_rcut", "0"))

    def test_incar_lcao_default(self, temp_dir):
        """测试 LCAO 模式下的默认参数."""
        config = {
            "element": "Si",
            "ecutwfc": 60,
            "ecutjy": 40,
            "nspin": 1,
            "basis_type": "lcao",  # lcao 模式
            "bessel_nao_rcut": [7],
        }
        output_path = os.path.join(temp_dir, "INPUT")
        generate_incar(config, output_path)

        with open(output_path) as f:
            content = f.read()

        # LCAO 模式应自动添加这些参数
        assert "ks_solver" in content
        assert "genelpa" in content
        assert "out_mat_hs" in content
        assert "1 12" in content
        assert "out_wfc_lcao" in content

    def test_incar_no_autoset(self, temp_dir):
        """测试关闭自动填充."""
        config = {"element": "Si", "ecutwfc": 60}
        output_path = os.path.join(temp_dir, "INPUT")
        generate_incar(config, output_path, auto_set=False)

        with open(output_path) as f:
            content = f.read()

        # 不应有默认值
        assert "ks_solver" not in content
        assert "smearing_method" not in content


# ============================================================================
# 3. STRU 接口测试
# ============================================================================

class TestStruInterface:
    """STRU 接口测试."""

    def test_dft_folder_name_dimer(self):
        """测试 dimer 文件夹名."""
        assert dft_folder_name("U", "dimer", 1.89, rcut=9) == "U-dimer-1.89-9au"
        assert dft_folder_name("Si", "dimer", 2.0, rcut=7) == "Si-dimer-2.00-7au"

    def test_dft_folder_name_monomer(self):
        """测试 monomer 文件夹名 (无 pert)."""
        assert dft_folder_name("U", "monomer", 0) == "U-monomer"

    def test_dft_folder_name_no_rcut(self):
        """测试无 rcut 的文件夹名."""
        assert dft_folder_name("U", "dimer", 1.89) == "U-dimer-1.89"

    def test_atom_coords_dimer(self):
        """测试 dimer 原子坐标."""
        coords = generate_atom_coords("dimer", bond_length=1.89, lattice_constant=30.0)
        assert len(coords) == 2
        # 第一个原子在 (shift, shift, shift)
        # 第二个原子在 (shift, shift, bond_length + shift)
        shift = 30.0 / 2 / 1.8897259886
        assert abs(coords[0][0] - shift) < 1e-6
        assert abs(coords[0][1] - shift) < 1e-6
        assert abs(coords[0][2] - shift) < 1e-6
        assert abs(coords[1][2] - 1.89 - shift) < 1e-6

    def test_atom_coords_monomer(self):
        """测试 monomer 原子坐标."""
        coords = generate_atom_coords("monomer", bond_length=0.0, lattice_constant=30.0)
        assert len(coords) == 1

    def test_atom_coords_trimer(self):
        """测试 trimer 原子坐标."""
        coords = generate_atom_coords("trimer", bond_length=2.0, lattice_constant=30.0)
        assert len(coords) == 3

    def test_atom_coords_octahedron(self):
        """测试 octahedron 原子坐标."""
        coords = generate_atom_coords("octahedron", bond_length=2.0, lattice_constant=30.0)
        assert len(coords) == 6

    def test_atom_coords_cube(self):
        """测试 cube 原子坐标."""
        coords = generate_atom_coords("cube", bond_length=2.0, lattice_constant=30.0)
        assert len(coords) == 8

    def test_atom_coords_invalid_proto(self):
        """测试非法 proto."""
        with pytest.raises(ValueError):
            generate_atom_coords("invalid_proto", 2.0, 30.0)

    def test_generate_stru_u(self, u_json_config, temp_dir):
        """测试生成 U 元素的 STRU 文件."""
        output_path = os.path.join(temp_dir, "U-dimer-1.89-9au", "STRU")
        orb_filename = "U_gga_9au_100Ry_27s27p26d26f25g.orb"
        stru_path = generate_stru(
            u_json_config,
            output_path=output_path,
            bond_length=1.89,
            orb_filename=orb_filename,
        )

        assert os.path.exists(stru_path)

        # 验证内容
        with open(stru_path, "r") as f:
            content = f.read()

        # 必须包含的段
        assert "ATOMIC_SPECIES" in content
        assert "U" in content
        assert "U.pbe-n-nc.upf" in content
        assert "NUMERICAL_ORBITAL" in content
        assert orb_filename in content
        assert "LATTICE_CONSTANT" in content
        assert "LATTICE_VECTORS" in content
        assert "ATOMIC_POSITIONS" in content
        assert "Cartesian_angstrom_center_xyz" in content

        # 验证原子坐标: 第一个原子 (shift, shift, shift), 第二个 (shift, shift, bond_length+shift)
        # shift = 30.0 / 2 / 1.8897259886 = 7.93765873
        # 第二个原子的 z 坐标 = 1.89 + 7.93765873 = 9.82765873
        shift = 30.0 / 2 / 1.8897259886
        expected_z = 1.89 + shift
        assert f"{expected_z:.8f}" in content

    def test_generate_stru_monomer(self, u_json_config, temp_dir):
        """测试生成 monomer 的 STRU."""
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
        # monomer 只有 1 个原子
        assert "1       //number of atoms" in content

    def test_generate_stru_nspin2(self, u_json_config, temp_dir):
        """测试 nspin=2 时的 starting magnetization."""
        u_json_config["geoms"][0]["nspin"] = 2
        output_path = os.path.join(temp_dir, "U-dimer-1.89-9au", "STRU")
        generate_stru(u_json_config, output_path, bond_length=1.89)

        with open(output_path, "r") as f:
            content = f.read()
        # nspin=2 时 starting_magnetization = 2.0
        assert "2.00" in content


# ============================================================================
# 3.5 ASE 转换测试
# ============================================================================

class TestAseConversion:
    """read_stru_as_ase + verify_ase_atoms 测试."""

    def test_read_stru_as_ase_dimer(self, u_json_config, temp_dir):
        """读取 dimer 的 STRU 并验证 ASE 转换正确性."""
        from ase.io import read as ase_read
        from ase import Atoms

        # 先生成 STRU
        output_path = os.path.join(temp_dir, "U-dimer-2.75-9au", "STRU")
        generate_stru(
            u_json_config,
            output_path=output_path,
            bond_length=2.75,
            orb_filename="U_gga_9au_100Ry_27s27p26d26f25g.orb",
        )

        # 1. 直接用 ASE 读应当失败
        with pytest.raises(AssertionError):
            ase_read(output_path, format="abacus")

        # 2. 用我们的 read_stru_as_ase 应当成功
        atoms = read_stru_as_ase(output_path)
        assert isinstance(atoms, Atoms)
        assert atoms.get_chemical_formula() == "U2"
        assert len(atoms) == 2
        assert all(atoms.pbc)

        # 3. 验证位置: 第一个原子在 (7.9377, 7.9377, 7.9377) 附近
        import numpy as np
        shift = 30.0 / 2 / 1.8897259886
        np.testing.assert_allclose(
            atoms.positions[0], [shift, shift, shift], atol=1e-6
        )
        # 第二个原子 z = shift + 2.75
        np.testing.assert_allclose(
            atoms.positions[1], [shift, shift, shift + 2.75], atol=1e-6
        )

        # 4. 验证 cell: 15.875 Å (30 Bohr / 1.8897)
        cell_ang = 30.0 / 1.8897259886
        np.testing.assert_allclose(
            atoms.cell.array, np.eye(3) * cell_ang, atol=1e-6
        )

    def test_verify_ase_atoms_dimer(self, u_json_config, temp_dir):
        """verify_ase_atoms 报告应全 ok."""
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
        """读取 monomer STRU: 1 个原子."""
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
    """params_stru dict → ase.Atoms 转换测试."""

    def _make_params_stru(self, bond_length=2.75, nspin=1):
        """构造一个 dimmer 风格的 params_stru."""
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
        """基本转换: U-dimer."""
        atoms = params_stru_to_ase(self._make_params_stru(2.75))
        assert atoms.get_chemical_formula() == "U2"
        assert len(atoms) == 2
        assert all(atoms.pbc)
        import numpy as np
        np.testing.assert_allclose(
            atoms.cell.array, np.eye(3) * 15.875317469822937, atol=1e-8
        )
        # 位置 (Angstrom)
        assert atoms.positions[0, 2] == 7.93765873
        assert abs(atoms.positions[1, 2] - 10.68765873) < 1e-6

    def test_magnetic_moments_collinear(self):
        """共线磁矩: m = [0, 0, 2.0] 保留为 3-vector 形式."""
        params_stru = self._make_params_stru(nspin=2)
        atoms = params_stru_to_ase(params_stru)
        m = atoms.get_initial_magnetic_moments()
        # ASE 完整保留 3-vector (不会自动 collapse 到 z 分量)
        import numpy as np
        np.testing.assert_allclose(m, [[0, 0, 2.0], [0, 0, 2.0]], atol=1e-8)

    def test_masses_preserved(self):
        """质量传递正确."""
        params_stru = self._make_params_stru()
        params_stru["species"][0]["mass"] = 238.03
        atoms = params_stru_to_ase(params_stru)
        import numpy as np
        np.testing.assert_allclose(atoms.get_masses(), [238.03, 238.03])

    def test_multi_species(self):
        """多元素: 比如 U-O."""
        params_stru = self._make_params_stru()
        # 在末尾追加一个 O species
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
        # 用元素计数核对 (不依赖 ASE formula 的输出顺序)
        from collections import Counter
        assert Counter(atoms.get_chemical_symbols()) == Counter(["U", "U", "O"])
        assert len(atoms) == 3
        assert atoms.get_chemical_formula() in ("UUO", "OU2")  # ASE 顺序不固定

    def test_direct_coords_conversion(self):
        """Direct 分数坐标 -> Cartesian."""
        params_stru = self._make_params_stru()
        params_stru["coord_type"] = "Direct"
        # 把 Cartesian 转成 Direct (0.5, 0.5, shift/cell)
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
        """验证报告应全 ok."""
        report = params_stru_to_ase_validate(self._make_params_stru())
        for name, info in report.items():
            if isinstance(info, dict) and "ok" in info:
                assert info["ok"], f"{name} failed: {info}"
        assert report["n_atoms"]["got"] == 2
        assert report["formula"]["got"] == "U2"
        assert report["cell"]["max_abs_error_angstrom"] < 1e-6
        assert report["coord_type"]["is_direct"] is False

    def test_roundtrip_with_stru_file(self, u_json_config, temp_dir):
        """对生成的 STRU 文件, parse → params_stru_to_ase 应与 read_stru_as_ase 等价."""
        output_path = os.path.join(temp_dir, "U-dimer-2.75-9au", "STRU")
        generate_stru(
            u_json_config, output_path, bond_length=2.75,
            orb_filename="U_gga_9au_100Ry_27s27p26d26f25g.orb",
        )

        # 方法 1: read_stru_as_ase
        atoms1 = read_stru_as_ase(output_path)

        # 方法 2: parse_stru -> 构造 params_stru -> params_stru_to_ase
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
# 4. 集成测试 - 完整工作流
# ============================================================================

class TestIntegration:
    """完整工作流测试."""

    @siab_required
    def test_full_workflow(self, u_json_config, temp_dir):
        """测试同时生成 NSW, INPUT, STRU 文件.

        模拟 SIAB 生成:
        - U_gga_9au_100Ry_*s*p*d*f*g.orb (NSW)
        - U-dimer-1.89-9au/INPUT
        - U-dimer-1.89-9au/STRU
        """
        # 1. 生成 NSW
        nsw_path = generate_nsw(
            u_json_config,
            output_dir=os.path.join(temp_dir, "primitive_jy"),
            lmaxmax=4,
        )
        assert os.path.exists(nsw_path)

        # 2. 生成 INPUT
        job_folder = os.path.join(temp_dir, "U-dimer-1.89-9au")
        os.makedirs(job_folder, exist_ok=True)
        input_path = generate_incar(
            u_json_config,
            output_path=os.path.join(job_folder, "INPUT"),
        )
        assert os.path.exists(input_path)

        # 3. 生成 STRU
        stru_path = generate_stru(
            u_json_config,
            output_path=os.path.join(job_folder, "STRU"),
            bond_length=1.89,
            orb_filename=os.path.basename(nsw_path),
        )
        assert os.path.exists(stru_path)

        # 验证文件结构
        assert os.path.isfile(nsw_path)
        assert os.path.isfile(input_path)
        assert os.path.isfile(stru_path)

        # 验证 STRU 引用了 NSW 轨道
        with open(stru_path) as f:
            stru_content = f.read()
        assert os.path.basename(nsw_path) in stru_content


# ============================================================================
# 5. 错误处理测试
# ============================================================================

class TestErrorHandling:
    """错误处理测试."""

    def test_missing_element(self, temp_dir):
        """测试缺少 element 字段."""
        config = {"ecutjy": 100, "bessel_nao_rcut": [9]}
        with pytest.raises(KeyError):
            generate_nsw(config, output_dir=temp_dir)

    def test_missing_ecutjy(self, temp_dir):
        """测试缺少 ecutjy 字段."""
        config = {"element": "U", "bessel_nao_rcut": [9]}
        with pytest.raises(KeyError):
            generate_nsw(config, output_dir=temp_dir)

    def test_invalid_proto(self):
        """测试非法 proto."""
        with pytest.raises(ValueError):
            generate_atom_coords("invalid", 2.0, 30.0)

    def test_parse_nonexistent_file(self):
        """测试读取不存在的文件."""
        with pytest.raises(FileNotFoundError):
            parse_incar("/nonexistent/path/INPUT")


# ============================================================================
# 6. 性能/烟雾测试
# ============================================================================

class TestSmoke:
    """简单烟雾测试 - 确保接口不抛异常."""

    @siab_required
    def test_smoke_nsw(self, u_json_config, temp_dir):
        """快速验证 NSW 接口."""
        try:
            generate_nsw(u_json_config, output_dir=temp_dir, lmaxmax=2)
            assert True
        except Exception as e:
            pytest.fail(f"NSW generation failed: {e}")

    def test_smoke_incar(self, u_json_config, temp_dir):
        """快速验证 INCAR 接口."""
        try:
            generate_incar(u_json_config, os.path.join(temp_dir, "INPUT"))
            assert True
        except Exception as e:
            pytest.fail(f"INCAR generation failed: {e}")

    def test_smoke_stru(self, u_json_config, temp_dir):
        """快速验证 STRU 接口."""
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
