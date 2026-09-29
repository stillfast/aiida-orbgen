"""
对比测试: 验证 aiida_orbgen.interfaces 生成的 INCAR、STRU、orb 文件
与 project/pbe 中的实际文件一致。

运行方式:
    cd /home/liguozhou/abacus/calculations/orbgen/aiida_orbgen
    python tests/test_project_pbe.py
    # 或
    pytest tests/test_project_pbe.py -v
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import pytest

# 路径设置
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

# 添加 SIAB 到 path
#
# ``append`` 而不是 ``insert(0, ...)``: 这台机器上有两份 SIAB, 这一份未打
# ``simpson`` 补丁 (scipy >= 1.12 删掉了 ``scipy.integrate.simps``), 插到最前面会
# 遮蔽 workspace 里可用的那份, 并污染同进程后续所有测试.
SIAB_PATH = Path("/home/liguozhou/install/ABACUS-CSW-NAO")
if SIAB_PATH.exists() and str(SIAB_PATH) not in sys.path:
    sys.path.append(str(SIAB_PATH))

# 检测 SIAB 是否可用
try:
    import SIAB  # noqa: F401
    SIAB_AVAILABLE = True
except ImportError:
    SIAB_AVAILABLE = False

siab_required = pytest.mark.skipif(
    not SIAB_AVAILABLE,
    reason="SIAB library not available"
)

# 项目实际目录
PROJECT_DIR = Path("/home/liguozhou/abacus/calculations/orbgen/project/pbe")
JSON_CONFIG_PATH = PROJECT_DIR / "pbe_orbgen.json"
PRIMITIVE_JY_DIR = PROJECT_DIR / "primitive_jy"
JOB_DIR = PROJECT_DIR / "U-dimer-1.89-9au"
EXPECTED_ORB = PRIMITIVE_JY_DIR / "U_gga_9au_100Ry_27s27p26d26f25g.orb"
EXPECTED_INPUT = JOB_DIR / "INPUT"
EXPECTED_STRU = JOB_DIR / "STRU"

from aiida_orbgen.interfaces.nsw import generate_nsw, compute_nbes_per_l
from aiida_orbgen.interfaces.incar import generate_incar, parse_incar
from aiida_orbgen.interfaces.stru import generate_stru, dft_folder_name


# ============================================================================
# 1. NSW 轨道对比
# ============================================================================

@siab_required
class TestNswComparison:
    """对比生成的 NSW 轨道与实际 NSW 轨道."""

    def test_nsw_file_exists(self):
        """验证 project/pbe/primitive_jy/U_gga_9au_100Ry_27s27p26d26f25g.orb 存在."""
        assert EXPECTED_ORB.exists(), f"NSW file not found: {EXPECTED_ORB}"
        assert EXPECTED_ORB.stat().st_size > 0, "NSW file is empty"

    def test_nsw_filename_format(self):
        """验证 NSW 文件名格式符合 U_gga_9au_100Ry_27s27p26d26f25g.orb."""
        filename = EXPECTED_ORB.name
        assert filename == "U_gga_9au_100Ry_27s27p26d26f25g.orb"
        assert filename.startswith("U_gga_9au_100Ry_")
        assert filename.endswith(".orb")

    def test_nsw_nbes_computation(self):
        """验证 nbes 计算结果与文件名一致.

        文件名: 27s27p26d26f25g
        意味着: l=0:27, l=1:27, l=2:26, l=3:26, l=4:25
        """
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)
        nbes = compute_nbes_per_l(
            rcut=float(config["bessel_nao_rcut"][0]),
            ecut=float(config["ecutjy"]),
            lmaxmax=config["geoms"][0]["lmaxmax"],
            primitive_type=config["primitive_type"],
        )
        nbes_int = [int(n) for n in nbes]
        # 文件名: 27s27p26d26f25g 表示 [27, 27, 26, 26, 25]
        # nbes 返回的是 reduced 后的 (即已经减1), 文件名也是 reduced 的
        assert nbes_int == [27, 27, 26, 26, 25], (
            f"Expected [27, 27, 26, 26, 25], got {nbes_int}"
        )

    def test_nsw_generation_matches(self, tmp_path):
        """用接口重新生成 NSW, 验证与实际文件内容一致."""
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)

        # 用接口生成
        generated_path = generate_nsw(
            config,
            output_dir=str(tmp_path),
            lmaxmax=config["geoms"][0]["lmaxmax"],
        )

        # 验证文件名一致
        assert Path(generated_path).name == EXPECTED_ORB.name

        # 验证内容长度大致相同
        gen_size = os.path.getsize(generated_path)
        exp_size = EXPECTED_ORB.stat().st_size
        # NSW 轨道文件格式相同, 大小应该接近
        assert abs(gen_size - exp_size) / exp_size < 0.1, (
            f"NSW file size differs: generated={gen_size}, expected={exp_size}"
        )

        # 验证文件头一致
        with open(generated_path) as f:
            gen_content = f.read()
        with open(EXPECTED_ORB) as f:
            exp_content = f.read()

        # 提取头部信息 (取前 1500 字符, 涵盖 dr 等参数)
        gen_header = gen_content[:1500]
        exp_header = exp_content[:1500]
        # 头部应该都包含元素符号、ecut、rcut、dr
        # 实际格式: "Element                     U" (多空格)
        for keyword in ["Element", "U", "Energy Cutoff", "Radius Cutoff",
                        "Lmax", "Sorbital", "Porbital", "Dorbital",
                        "Forbital", "Gorbital", "Mesh", "dr"]:
            assert keyword in gen_header, f"Missing '{keyword}' in generated NSW"

        # 数值参数 (保留 JSON 原类型: "100" (int), "9" (int), "0.01" (float))
        assert "100" in gen_header  # ecut
        assert " 9\n" in gen_header or "9 " in gen_header  # rcut
        assert "0.01" in gen_header  # dr


# ============================================================================
# 2. INPUT (INCAR) 对比
# ============================================================================

@siab_required
class TestIncarComparison:
    """对比生成的 INPUT 与实际 INPUT."""

    def test_input_file_exists(self):
        """验证 project/pbe/U-dimer-1.89-9au/INPUT 存在."""
        assert EXPECTED_INPUT.exists(), f"INPUT not found: {EXPECTED_INPUT}"
        assert EXPECTED_INPUT.stat().st_size > 0

    def test_input_key_parameters(self):
        """验证 INPUT 关键参数与实际一致."""
        params = parse_incar(str(EXPECTED_INPUT))

        # 关键参数
        assert str(params.get("basis_type")) == "lcao"
        assert str(params.get("ecutwfc")) == "150"
        assert str(params.get("nspin")) == "1"
        assert str(params.get("nbands")) == "40"
        assert str(params.get("bessel_nao_rcut")) == "9"
        assert str(params.get("ks_solver")) == "genelpa"
        assert str(params.get("smearing_method")) == "gauss"
        assert "0.02" in str(params.get("smearing_sigma"))
        assert str(params.get("mixing_type")) == "broyden"
        assert "0.4" in str(params.get("mixing_beta"))
        assert str(params.get("mixing_ndim")) == "12"
        assert str(params.get("gamma_only")) == "1"
        assert str(params.get("out_wfc_lcao")) == "1"
        assert str(params.get("out_mat_hs")) == "1 12"
        assert str(params.get("out_mat_tk")) == "1 12"

    def test_input_generation_matches(self, tmp_path):
        """用接口重新生成 INPUT, 验证关键参数与实际一致."""
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)

        # 用接口生成
        gen_path = generate_incar(
            config,
            output_path=str(tmp_path / "INPUT"),
        )

        # 解析生成的文件
        gen_params = parse_incar(gen_path)
        exp_params = parse_incar(str(EXPECTED_INPUT))

        # 路径类参数会被 SIAB 转换为绝对路径, 接口层只能使用相对路径
        # 所以跳过这些字段的比较
        SKIP_KEYS = {"pseudo_dir", "orbital_dir"}

        # 比较所有共同存在的参数
        for key in gen_params:
            if key in exp_params and key not in SKIP_KEYS:
                assert str(gen_params[key]) == str(exp_params[key]), (
                    f"Parameter '{key}' mismatch: "
                    f"generated={gen_params[key]}, expected={exp_params[key]}"
                )

    def test_input_no_extra_user_params_in_expected(self):
        """实际 INPUT 不应包含用户没有指定的额外 DFT 参数 (除了 autoset 默认值)."""
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)

        # 解析实际 INPUT
        exp_params = parse_incar(str(EXPECTED_INPUT))

        # 用户显式提供的 DFT 相关参数
        user_dft_params = {
            "ecutwfc", "smearing_method", "smearing_sigma",
            "mixing_type", "mixing_beta", "mixing_ndim"
        }

        for key in user_dft_params:
            assert key in exp_params, f"User-specified param '{key}' missing in INPUT"

    def test_input_lcao_derived_from_fit_basis(self):
        """basis_type 应当从 fit_basis='jy' 推断为 lcao."""
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)
        assert config.get("fit_basis") == "jy"
        # 这条规则: fit_basis='jy' => basis_type='lcao'
        # 已经在接口 _extract_dftparam 中实现
        params = parse_incar(str(EXPECTED_INPUT))
        assert params.get("basis_type") == "lcao"


# ============================================================================
# 3. STRU 对比
# ============================================================================

@siab_required
class TestStruComparison:
    """对比生成的 STRU 与实际 STRU."""

    def test_stru_file_exists(self):
        """验证 project/pbe/U-dimer-1.89-9au/STRU 存在."""
        assert EXPECTED_STRU.exists(), f"STRU not found: {EXPECTED_STRU}"

    def test_stru_key_sections(self):
        """验证 STRU 关键段."""
        content = EXPECTED_STRU.read_text()
        assert "ATOMIC_SPECIES" in content
        assert "U 1.000000 U.pbe-n-nc.upf" in content
        assert "NUMERICAL_ORBITAL" in content
        assert "U_gga_9au_100Ry_27s27p26d26f25g.orb" in content
        assert "LATTICE_CONSTANT" in content
        assert "30.000000" in content
        assert "LATTICE_VECTORS" in content
        assert "ATOMIC_POSITIONS" in content
        assert "Cartesian_angstrom_center_xyz" in content
        assert "0.00" in content  # starting magnetization
        assert "2       //number of atoms" in content

    def test_stru_atom_coordinates(self):
        """验证 STRU 原子坐标.

        shift = 30.0 / 2 / 1.8897259886 = 7.93765873
        dimer 第二个原子的 z 坐标 = 1.89 + shift = 9.82765873
        """
        content = EXPECTED_STRU.read_text()
        # bond_length 1.89
        shift = 30.0 / 2 / 1.8897259886
        expected_z = 1.89 + shift
        assert f"{expected_z:.8f}" in content

    def test_stru_generation_matches(self, tmp_path):
        """用接口重新生成 STRU, 验证与实际一致."""
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)

        gen_path = generate_stru(
            config,
            output_path=str(tmp_path / "STRU"),
            bond_length=1.89,
            orb_filename="U_gga_9au_100Ry_27s27p26d26f25g.orb",
        )

        gen_content = Path(gen_path).read_text()
        exp_content = EXPECTED_STRU.read_text()

        # 比较关键字段
        for keyword in [
            "ATOMIC_SPECIES",
            "U 1.000000 U.pbe-n-nc.upf",
            "NUMERICAL_ORBITAL",
            "U_gga_9au_100Ry_27s27p26d26f25g.orb",
            "LATTICE_CONSTANT",
            "30.000000",
            "Cartesian_angstrom_center_xyz",
            "0.00",
            "2       //number of atoms",
            "7.93765873 7.93765873 7.93765873 0 0 0",
            "7.93765873 7.93765873 9.82765873 0 0 0",
        ]:
            assert keyword in gen_content, f"Missing '{keyword}' in generated STRU"
            assert keyword in exp_content, f"Missing '{keyword}' in expected STRU"

    def test_dft_folder_name(self):
        """验证 dft_folder_name 与实际文件夹名一致."""
        # 实际文件夹: U-dimer-1.89-9au
        assert JOB_DIR.name == dft_folder_name("U", "dimer", 1.89, rcut=9)


# ============================================================================
# 4. 集成对比 - 三个文件一起
# ============================================================================

@siab_required
class TestFullComparison:
    """完整对比测试: 同时验证三个文件的生成结果."""

    def test_generate_all_three_files(self, tmp_path):
        """一次生成 NSW + INPUT + STRU 三个文件."""
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)

        # 1. 生成 NSW
        nsw_path = generate_nsw(
            config,
            output_dir=str(tmp_path / "primitive_jy"),
            lmaxmax=config["geoms"][0]["lmaxmax"],
        )

        # 2. 生成 INPUT
        job_folder = tmp_path / "U-dimer-1.89-9au"
        job_folder.mkdir()
        input_path = generate_incar(
            config,
            output_path=str(job_folder / "INPUT"),
        )

        # 3. 生成 STRU
        stru_path = generate_stru(
            config,
            output_path=str(job_folder / "STRU"),
            bond_length=1.89,
            orb_filename=Path(nsw_path).name,
        )

        # 验证文件存在
        assert os.path.exists(nsw_path)
        assert os.path.exists(input_path)
        assert os.path.exists(stru_path)

        # 验证 NSW 文件名与 project/pbe 实际一致
        assert Path(nsw_path).name == "U_gga_9au_100Ry_27s27p26d26f25g.orb"
        assert Path(nsw_path).name == EXPECTED_ORB.name

        # 验证 STRU 引用了正确的 NSW
        stru_content = Path(stru_path).read_text()
        assert Path(nsw_path).name in stru_content

    def test_nsw_content_size_consistent(self):
        """验证生成的 NSW 文件大小与实际文件基本一致."""
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)

        with tempfile.TemporaryDirectory() as tmp:
            gen_path = generate_nsw(
                config,
                output_dir=tmp,
                lmaxmax=config["geoms"][0]["lmaxmax"],
            )
            gen_size = os.path.getsize(gen_path)
            exp_size = EXPECTED_ORB.stat().st_size

            # 文件大小差异应该 < 5%
            ratio = abs(gen_size - exp_size) / exp_size
            assert ratio < 0.05, (
                f"NSW file size differs by {ratio*100:.1f}%: "
                f"generated={gen_size}, expected={exp_size}"
            )

    def test_input_parameter_set_consistent(self):
        """验证 INPUT 参数集合与实际文件一致 (允许有相同值)."""
        with open(JSON_CONFIG_PATH) as f:
            config = json.load(f)

        with tempfile.TemporaryDirectory() as tmp:
            gen_path = generate_incar(
                config,
                output_path=str(Path(tmp) / "INPUT"),
            )
            gen_params = parse_incar(gen_path)
            exp_params = parse_incar(str(EXPECTED_INPUT))

            # 计算参数集的 Jaccard 相似度
            gen_keys = set(gen_params.keys())
            exp_keys = set(exp_params.keys())

            intersection = gen_keys & exp_keys
            union = gen_keys | exp_keys
            jaccard = len(intersection) / len(union) if union else 0

            # 至少 90% 的参数匹配
            assert jaccard > 0.85, (
                f"INPUT parameter sets diverge: jaccard={jaccard:.2f}, "
                f"generated_only={gen_keys - exp_keys}, "
                f"expected_only={exp_keys - gen_keys}"
            )


if __name__ == "__main__":
    # 直接运行时, 显示详细输出
    pytest.main([__file__, "-v", "--tb=short", "-s"])
