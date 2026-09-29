"""
INPUT 接口 (类比 VASP 的 INCAR)

生成 ABACUS INPUT 文件。

对应文件: {folder}/INPUT
例: U-dimer-1.89-9au/INPUT

本模块是对 SIAB 的高层封装，底层调用:
- ``SIAB.abacus.io.autoset``           : 自动设置默认 DFT 参数
- ``SIAB.abacus.io.dftparam_to_text``  : 字典 → INPUT 文本

这样做的好处:
1. 默认值始终与 SIAB 主线保持一致，避免接口与上游漂移。
2. 接口层只负责 "JSON → ABACUS DFT 参数" 的语义映射。
"""

import os
import re
from typing import Optional, Dict, Any

__all__ = ["generate_incar", "parse_incar"]


# 与 SIAB 行为保持一致：fit_basis='jy' 时强制使用 lcao 基组
def _resolve_basis_type(json_config: dict, dftparam: dict) -> str:
    """
    从 fit_basis 推断 basis_type。

    SIAB.driver.main.init 的约定:
        fit_basis == 'jy'  -> basis_type == 'lcao'
        fit_basis == 'pw'  -> basis_type == 'pw'
    """
    if "basis_type" in dftparam:
        return dftparam["basis_type"]
    fit_basis = json_config.get("fit_basis", "jy")
    return "lcao" if fit_basis == "jy" else "pw"


def _resolve_nbands(json_config: dict, dftparam: dict) -> Optional[int]:
    """
    推断 nbands。

    SIAB 行为: nbands 从具体几何 (geoms[i].nbands) 继承。
    若 JSON 顶层没有提供，则从 geoms[0] 中读取。
    """
    if "nbands" in dftparam:
        return dftparam["nbands"]
    geoms = json_config.get("geoms", [])
    if geoms and "nbands" in geoms[0]:
        return geoms[0]["nbands"]
    return None  # 由 SIAB.autoset 填入默认值 'auto'


def _extract_dftparam(json_config: dict) -> dict:
    """
    从 SIAB JSON 配置中提取 ABACUS DFT 输入参数。

    通过白名单 ``SIAB.abacus.io.ABACUS_PARAMS`` 过滤掉非 ABACUS 字段
    (例如 SIAB 专用的 ``element``/``ecutjy``/``bessel_nao_rcut`` 等)。
    """
    from SIAB.abacus.io import ABACUS_PARAMS

    ABACUS_PARAM_SET = set(ABACUS_PARAMS)
    dftparam = {k: v for k, v in json_config.items() if k in ABACUS_PARAM_SET}

    # 特殊推断
    dftparam["basis_type"] = _resolve_basis_type(json_config, dftparam)
    nbands = _resolve_nbands(json_config, dftparam)
    if nbands is not None:
        dftparam["nbands"] = nbands

    # pseudo_dir 在 JSON 中可能是路径形式；保留原值（autoset 会用 './' 作默认）
    if "pseudo_dir" not in dftparam and "pseudo_dir" in json_config:
        dftparam["pseudo_dir"] = json_config["pseudo_dir"]

    return dftparam


def generate_incar(
    json_config: dict,
    output_path: str,
    auto_set: bool = True,
) -> str:
    """
    生成 ABACUS INPUT 文件。

    Parameters
    ----------
    json_config : dict
        SIAB JSON 配置字典 (e.g. project/pbe/pbe_orbgen.json)。
        顶层 ABACUS 相关字段会透传到 INPUT，其余字段被忽略。
    output_path : str
        输出文件路径
    auto_set : bool
        是否自动填充默认参数 (default True，对应 SIAB.autoset)。

    Returns
    -------
    str
        生成的 INPUT 文件的绝对路径

    Examples
    --------
    >>> import json
    >>> with open("pbe_orbgen.json") as f:
    ...     config = json.load(f)
    >>> generate_incar(config, output_path="./U-dimer-1.89-9au/INPUT")
    """
    # 延迟导入 SIAB，方便在没有 SIAB 的环境下 import 该模块
    from SIAB.abacus.io import autoset as _siab_autoset
    from SIAB.abacus.io import dftparam_to_text as _siab_dftparam_to_text

    dftparam = _extract_dftparam(json_config)

    if auto_set:
        dftparam = _siab_autoset(dftparam)
    else:
        # 不补默认：把 bessel_nao_rcut=[9] 这种 list 转成字符串
        # 让 ABACUS 能正确解析
        for k, v in list(dftparam.items()):
            if isinstance(v, list):
                dftparam[k] = " ".join(str(x) for x in v)

    text = _siab_dftparam_to_text(dftparam)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(text)

    return os.path.abspath(output_path)


def parse_incar(filepath: str) -> dict:
    """
    从 INPUT 文件读取参数。

    Parameters
    ----------
    filepath : str
        INPUT 文件路径

    Returns
    -------
    dict
        解析出的参数 (值尽量推断为 int/float/str)

    Examples
    --------
    >>> params = parse_incar("./U-dimer-1.89-9au/INPUT")
    >>> params["ecutwfc"]
    150
    """
    result: Dict[str, Any] = {}
    pattern = re.compile(r"^\s*([\w_]+)\s+([^\#]+)")
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line == "INPUT_PARAMETERS":
                continue
            match = pattern.match(line)
            if match is None:
                continue
            key = match.group(1)
            raw_value = match.group(2).strip()
            # 尝试转 int / float
            try:
                value: Any = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
            result[key] = value
    return result
