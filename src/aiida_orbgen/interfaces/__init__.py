"""
Interfaces for aiida_orbgen

统一的接口层，对外暴露简单的高级 API。
data 层负责 AiiDA 节点转换，workflows 层负责 AiiDA 工作流。
本层提供函数式接口，方便测试和快速调用。

主要 API
--------

NSW (原始轨道)
~~~~~~~~~~~~~~
- :func:`generate_nsw`              生成单个 .orb 原始球 Bessel 轨道
- :func:`compute_nbes_per_l`        计算每个角动量通道的 Bessel 函数个数

INPUT (ABACUS 主输入)
~~~~~~~~~~~~~~~~~~~~~
- :func:`generate_incar`            从 SIAB JSON 生成 ABACUS INPUT
- :func:`parse_incar`               从 INPUT 文件反向解析为 dict

STRU (ABACUS 结构)
~~~~~~~~~~~~~~~~~~
- :func:`generate_stru`             从 SIAB JSON 生成 ABACUS STRU
- :func:`dft_folder_name`           SIAB 标准 DFT 任务文件夹名
- :func:`generate_atom_coords`      计算 SIAB 几何原型的原子坐标
- :func:`parse_stru`                从 STRU 文件反向解析为 dict
- :func:`read_stru_as_ase`          从 STRU 文件读出 ASE ``Atoms`` 对象

Pipeline (一键)
~~~~~~~~~~~~~~~
- :func:`generate_all`              一次生成 NSW + 多个 INPUT/STRU
- :func:`generate_all_from_json`    从 JSON 路径一键生成

底层依赖
--------
所有接口都委托给 ``SIAB`` (ABACUS-CSW-NAO) 完成实际计算，确保默认值
和文件格式与 SIAB 主线一致。
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
