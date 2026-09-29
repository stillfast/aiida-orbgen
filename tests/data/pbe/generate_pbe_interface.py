"""
generate_pbe_interface.py

使用 aiida-orbgen 的接口, 在 project/pbe_interface/ 目录中生成:
- primitive_jy/U_gga_9au_100Ry_27s27p26d26f25g.orb  (NSW 原始轨道)
- U-dimer-1.89-9au/INPUT                          (ABACUS 输入)
- U-dimer-1.89-9au/STRU                           (ABACUS 结构)

然后可以用 diff 等工具与 project/pbe/ 中的实际文件对比。

运行方式:
    cd /home/liguozhou/abacus/calculations/orbgen
    python aiida-orbgen/tests/data/pbe/generate_pbe_interface.py
"""

import json
import os
import sys
from pathlib import Path

# 将 aiida-orbgen/src 添加到 path
ROOT = Path("/home/liguozhou/abacus/calculations/orbgen")
sys.path.insert(0, str(ROOT / "aiida-orbgen" / "src"))
sys.path.insert(0, str(ROOT / "aiida-orbgen"))

# 添加 SIAB (ABACUS-CSW-NAO) 库到 path
SIAB_PATH = Path("/home/liguozhou/install/ABACUS-CSW-NAO")
if SIAB_PATH.exists() and str(SIAB_PATH) not in sys.path:
    sys.path.insert(0, str(SIAB_PATH))

from aiida_orbgen.interfaces.nsw import generate_nsw
from aiida_orbgen.interfaces.incar import generate_incar
from aiida_orbgen.interfaces.stru import generate_stru


def main():
    """主函数: 生成 NSW, INPUT, STRU 文件到 project/pbe_interface/ 目录."""

    # 配置路径
    config_path = ROOT / "aiida-orbgen" / "tests" / "data" / "pbe" / "pbe_orbgen.json"
    output_root = ROOT / "project" / "pbe_interface"
    job_folder = output_root / "U-dimer-1.89-9au"
    primitive_jy_dir = output_root / "primitive_jy"

    # 创建输出目录
    primitive_jy_dir.mkdir(parents=True, exist_ok=True)
    job_folder.mkdir(parents=True, exist_ok=True)

    # 读取 JSON 配置
    with open(config_path) as f:
        config = json.load(f)

    print("=" * 70)
    print("aiida-orbgen 接口文件生成测试")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"输出目录: {output_root}")
    print()

    # 1. 生成 NSW 原始轨道
    print("[1/3] 生成 NSW 原始轨道...")
    nsw_path = generate_nsw(
        config,
        output_dir=str(primitive_jy_dir),
        lmaxmax=config["geoms"][0]["lmaxmax"],
    )
    print(f"      NSW 文件: {nsw_path}")
    nsw_filename = os.path.basename(nsw_path)
    print()

    # 2. 生成 INPUT 文件
    print("[2/3] 生成 INPUT 文件...")
    input_path = generate_incar(
        config,
        output_path=str(job_folder / "INPUT"),
    )
    print(f"      INPUT 文件: {input_path}")
    print()

    # 3. 生成 STRU 文件
    print("[3/3] 生成 STRU 文件...")
    stru_path = generate_stru(
        config,
        output_path=str(job_folder / "STRU"),
        bond_length=1.89,
        orb_filename=nsw_filename,
    )
    print(f"      STRU 文件: {stru_path}")
    print()

    # 输出对比摘要
    print("=" * 70)
    print("生成完成, 可对比以下目录:")
    print("=" * 70)
    print(f"  参考: {ROOT / 'project' / 'pbe'}")
    print(f"  生成: {output_root}")
    print()
    print("对比命令:")
    print(f"  diff -r {ROOT / 'project' / 'pbe' / 'U-dimer-1.89-9au' / 'INPUT'} {job_folder / 'INPUT'}")
    print(f"  diff -r {ROOT / 'project' / 'pbe' / 'U-dimer-1.89-9au' / 'STRU'} {job_folder / 'STRU'}")
    print(f"  diff {ROOT / 'project' / 'pbe' / 'primitive_jy' / nsw_filename} {nsw_path}")
    print()
    print("或直接查看生成的文件:")
    print(f"  ls -la {output_root}")
    print(f"  ls -la {job_folder}")
    print(f"  ls -la {primitive_jy_dir}")


if __name__ == "__main__":
    main()
