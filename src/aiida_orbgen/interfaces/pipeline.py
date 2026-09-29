"""
Pipeline 接口

一键从 SIAB JSON 配置生成完整的 ABACUS 计算目录:

- ``primitive_jy/{elem}_{xc}_{rcut}au_{ecut}Ry_{nzeta}.orb``  (NSW 原始轨道)
- ``{proto}-{pert}-{rcut}au/INPUT``                          (ABACUS 主输入)
- ``{proto}-{pert}-{rcut}au/STRU``                           (ABACUS 结构)

对应 SIAB 的 ``SIAB.orbgen.main`` 工作流的第一步产物。

键长 (``pertmags``) 完全委托给 SIAB:

- 列表: 按给定键长依次展开
- ``"auto"``: 调用 ``SIAB.abacus.blscan.blgen`` (等价于
  ``SIAB.abacus.api._build_pert(pertmags='auto')``), 走 SIAB 内部的
  ``CellGenerator.get_dimer_bond_length`` 等默认键长表
- ``"scan"``: 与 ``"auto"`` 类似但会结合已有的 DFT 结果做键长筛选
"""

import os
import json
from typing import Optional, Dict, Any, List, Union
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
) -> List[float]:
    """
    决定要展开的键长列表。优先级:

    1. ``override`` 显式传入的 list
    2. ``bond_length`` 单个值 -> 包装为单元素 list
    3. ``geoms[0].pertmags``: 列表直接用; 字符串 ``"auto"``/``"scan"`` 调 SIAB
    4. ``geoms[0].pertmags``: 单个数字 -> 包装为单元素 list
    5. fallback: 调用 SIAB 的 ``_build_pert(pertmags='auto')`` 拿默认键长

    任何"自动"路径都不在接口层硬编码数值。
    """
    if override is not None:
        return [float(p) for p in override]
    if bond_length is not None:
        return [float(bond_length)]

    geoms = json_config.get("geoms", [])
    pertmags_raw = geoms[0].get("pertmags") if geoms else None

    if isinstance(pertmags_raw, str):
        # "auto" / "scan" → 完全交给 SIAB
        return _siab_pertmags(json_config, proto, pertmags_raw)
    if isinstance(pertmags_raw, list):
        return [float(p) for p in pertmags_raw]
    if isinstance(pertmags_raw, (int, float)):
        return [float(pertmags_raw)]

    # 顶层 JSON 没有 geoms 或 pertmags: 走 SIAB 默认
    return _siab_pertmags(json_config, proto, "auto")


def _siab_pertmags(
    json_config: dict,
    proto: str,
    mode: str,
) -> List[float]:
    """
    调用 SIAB 的 ``_build_pert`` / ``blgen`` 来获取自动键长。

    SIAB 支持的两种模式:
    - ``"auto"``: SIAB 查 ``CellGenerator.{proto}_bond_length(elem)`` 表
    - ``"scan"``: 与 ``"auto"`` 类似，但会读取已有 DFT 结果做筛选
      (这里只负责生成初始列表, 筛选留给后续步骤)
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
    一键生成 NSW 轨道 + 多个 INPUT/STRU 任务目录。

    Parameters
    ----------
    json_config : dict
        SIAB JSON 配置字典 (project/pbe/pbe_orbgen.json 风格)
    output_root : str
        输出根目录。所有路径相对于它:
        - NSW:    {output_root}/{nsw_dirname}/{orb_name}
        - INPUT:  {output_root}/{dft_folder}/INPUT
        - STRU:   {output_root}/{dft_folder}/STRU
    bond_length : float, optional
        单个 DFT 任务的键长 (Angstrom)。如果提供, 仅生成一个任务目录。
        与 ``pertmags`` 互斥。
    pertmags : list, optional
        显式键长列表。如果提供, 按给定键长依次展开。
        留空时:
        - JSON ``geoms[0].pertmags`` 是 list → 按列表展开
        - JSON ``geoms[0].pertmags`` 是 ``"auto"``/``"scan"`` → 调 SIAB 默认
    proto : str, optional
        几何原型 (覆盖 JSON 中的设置)
    nspin : int, optional
        自旋极化 (1 或 2)
    lattice_constant : float, optional
        晶格常数 (Bohr, 覆盖 JSON 中的设置)
    lmaxmax : int, optional
        最大角动量 (从 geoms[0].lmaxmax 推断)
    dr : float
        NSW 径向网格步长 (Bohr), default 0.01
    nsw_dirname : str
        NSW 轨道目录名 (default "primitive_jy")

    Returns
    -------
    dict
        - nsw:         NSW 轨道文件绝对路径
        - nsw_filename: NSW 轨道 basename
        - pertmags:    实际使用的键长列表
        - dft:         list of {folder, pert, input, stru}

    Examples
    --------
    >>> import json
    >>> with open("pbe_orbgen.json") as f:
    ...     config = json.load(f)
    >>> # JSON 写的是 "pertmags": "auto", 委托给 SIAB 生成键长
    >>> result = generate_all(config, output_root="./generated")
    >>> # 与 project/pbe/ 一样, 自动生成 U-dimer-1.89-9au ... U-dimer-4.50-9au
    >>> print([d["folder"] for d in result["dft"]])
    ['U-dimer-1.89-9au', 'U-dimer-2.09-9au', 'U-dimer-2.75-9au',
     'U-dimer-3.50-9au', 'U-dimer-4.50-9au']
    """
    # 1. NSW 原始轨道: 只生成一次
    nsw_dir = os.path.join(output_root, nsw_dirname)
    nsw_path = generate_nsw(
        json_config,
        output_dir=nsw_dir,
        lmaxmax=lmaxmax,
        dr=dr,
    )
    orb_filename = os.path.basename(nsw_path)

    # 2. 决定要展开的 DFT 任务列表
    elem = json_config["element"]
    if proto is None and json_config.get("geoms"):
        proto = json_config["geoms"][0].get("proto", "dimer")
    proto = proto or "dimer"

    rcut_raw = json_config.get("bessel_nao_rcut", [9])
    rcut_raw_first = rcut_raw[0] if isinstance(rcut_raw, list) else rcut_raw
    # 保留原类型 (int / float), 避免 9 -> 9.0 改变文件夹名
    if isinstance(rcut_raw_first, float) and rcut_raw_first.is_integer():
        rcut_for_folder: Any = int(rcut_raw_first)
    else:
        rcut_for_folder = rcut_raw_first
    rcut = float(rcut_raw_first)

    pertmags_list = _resolve_pertmags(json_config, proto, pertmags, bond_length)

    # 3. 为每个键长生成 INPUT/STRU
    dft_results = []
    for pert in pertmags_list:
        folder = dft_folder_name(elem, proto, pert, rcut=rcut_for_folder)
        dft_root = os.path.join(output_root, folder)
        os.makedirs(dft_root, exist_ok=True)

        input_path = generate_incar(
            json_config,
            output_path=os.path.join(dft_root, "INPUT"),
        )
        stru_path = generate_stru(
            json_config,
            output_path=os.path.join(dft_root, "STRU"),
            proto=proto,
            bond_length=pert,
            nspin=nspin,
            lattice_constant=lattice_constant,
            orb_filename=orb_filename,
        )
        dft_results.append({
            "folder": folder,
            "pert": pert,
            "input": os.path.abspath(input_path),
            "stru": os.path.abspath(stru_path),
        })

    return {
        "nsw": os.path.abspath(nsw_path),
        "nsw_filename": orb_filename,
        "pertmags": pertmags_list,
        "dft": dft_results,
    }


def generate_all_from_json(
    json_path: str,
    output_root: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    从 JSON 文件路径一键生成 NSW + INPUT + STRU。

    Parameters
    ----------
    json_path : str
        SIAB JSON 配置文件路径
    output_root : str, optional
        输出根目录 (默认: JSON 文件所在目录)
    **kwargs
        透传给 :func:`generate_all`

    Returns
    -------
    dict
        与 :func:`generate_all` 相同
    """
    json_path = os.path.abspath(json_path)
    with open(json_path, "r") as f:
        config = json.load(f)

    if output_root is None:
        output_root = str(Path(json_path).parent)

    return generate_all(config, output_root=output_root, **kwargs)
