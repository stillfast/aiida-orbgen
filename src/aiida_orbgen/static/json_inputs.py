"""
aiida_orbgen.static.json_inputs - 从 JSON 文件加载 abacus/orbgen 配置

提供两个核心函数:

- :func:`load_orbgen_config`  : 加载 ``pbe_orbgen.json`` 风格的 SIAB 配置
- :func:`load_abacus_config`  : 加载 ``abacus.json`` 风格的 ABACUS 配置
- :func:`parse_lmax_rcut_candidates` : 从 orbgen.json 提取 (l_max, r_cut) 候选列表

JSON schema
-----------

**abacus.json** (新文件, 调整 ABACUS INPUT + 决定 basis + tolerance):

abacus.json 现在用 ``abacus`` 键包裹所有 aiida-abacus 相关参数 (与 aiida-abacus
``test_abacus_base.py`` 一致), 结构如下:

.. code-block:: json

    {
      "basis": ["pw", "lcao_nsw"],

      "abacus": {
        "code": "abacus_lts@yeesuan",
        "parameters": {
          "input": {
            "ecutwfc": 100,
            "ks_solver": "scalapack_gvx",
            "nbands": 40,
            "mixing_type": "broyden",
            "mixing_beta": 0.4,
            "scf_thr": 1e-7,
            "smearing_method": "gauss",
            "smearing_sigma": 0.02,
            "nspin": 1
          }
        },
        "metadata": {
          "options": {
            "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 56},
            "max_wallclock_seconds": 600,
            "queue_name": "q_ysuan"
          }
        }
      },

      "tolerance_meV": 4.2,
      "max_l_max": 5,
      "max_r_cut": 12.0
    }

**orbgen.json** (现有 pbe_orbgen.json, 单组合 l_max/r_cut):

.. code-block:: json

    {
      "element": "U",
      "pseudo_dir": "./U.pbe-n-nc.upf",
      "fit_basis": "jy",
      "ecutjy": 100,
      "bessel_nao_rcut": [9],
      "geoms": [{"lmaxmax": 4, ...}],

      "geoms": [
        {"proto": "dimer", "pertmags": "auto", "nbands": 40, "nspin": 1, "lmaxmax": 4}
      ]
    }
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


__all__ = [
    "load_orbgen_config",
    "load_abacus_config",
    "parse_lmax_rcut_candidates",
    "merge_input_overrides",
    "merge_scheduler",
]


# ---------------------------------------------------------------------------
# JSON 加载
# ---------------------------------------------------------------------------


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """从 .json / .yaml 文件加载 dict。"""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML support requires PyYAML") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text) or {}


def load_orbgen_config(path: Union[str, Path]) -> Dict[str, Any]:
    """加载 ``pbe_orbgen.json`` 风格的 SIAB 配置。

    Returns
    -------
    dict
        原始 JSON 字典 (包含 element / pseudo_dir / geoms / ...)

    Raises
    ------
    ValueError
        必填字段缺失
    """
    cfg = load_json(path)
    required = ["element"]
    for k in required:
        if k not in cfg:
            raise ValueError(
                f"orbgen.json missing required key '{k}': {path}"
            )
    return cfg


def load_abacus_config(path: Union[str, Path]) -> Dict[str, Any]:
    """加载 ``abacus.json`` 风格的 ABACUS 配置。

    Returns
    -------
    dict
        包含::

            {
              "basis": ["pw", "lcao_nsw"],
              "input_overrides": {...},
              "tolerance_meV": 4.2,
              "scheduler": {"queue_name": ..., ...}
            }

    Raises
    ------
    ValueError
        basis 字段为空或缺
    """
    cfg = load_json(path)
    basis = cfg.get("basis", ["pw", "lcao_nsw"])
    if not basis:
        raise ValueError(
            f"abacus.json 'basis' must be a non-empty list: {path}"
        )
    valid = {"pw", "lcao_nsw"}
    for b in basis:
        if b not in valid:
            raise ValueError(
                f"abacus.json 'basis' contains unknown: {b!r}, "
                f"expected one of {valid}"
            )
    return cfg


# ---------------------------------------------------------------------------
# 候选参数解析
# ---------------------------------------------------------------------------


def parse_lmax_rcut_candidates(
    orbgen_cfg: Dict[str, Any],
) -> List[Tuple[int, float]]:
    """从 orbgen.json 提取 ``(l_max, r_cut)`` 候选列表。

    当前用法: 单组合, 直接从 ``bessel_nao_rcut`` + ``geoms[0].lmaxmax`` 提取::

        "bessel_nao_rcut": [9],
        "geoms[0].lmaxmax": 4
        → [(4, 9.0)]

    历史规则 (保留, 暂不启用):
        1) 显式 ``basis_candidates`` 列表
        2) 笛卡尔积 ``lmax_candidates × rcut_candidates``
        3) ``bessel_nao_rcut`` + ``lmaxmax`` (当前)

    Returns
    -------
    list of (l_max, r_cut) tuples
    """
    # 1) 显式 list (保留以备扩展)
    explicit = orbgen_cfg.get("basis_candidates")
    if explicit:
        out: List[Tuple[int, float]] = []
        for item in explicit:
            lmax = int(item["lmax"])
            rcut = float(item["rcut"])
            out.append((lmax, rcut))
        return out

    # 2) 笛卡尔积 (保留以备扩展)
    lmax_list = orbgen_cfg.get("lmax_candidates")
    rcut_list = orbgen_cfg.get("rcut_candidates")
    if lmax_list and rcut_list:
        return [
            (int(lm), float(rc))
            for lm in lmax_list
            for rc in rcut_list
        ]

    # 3) 默认: 用 bessel_nao_rcut + lmaxmax (单组合)
    rcut_raw = orbgen_cfg.get("bessel_nao_rcut", [9])
    if isinstance(rcut_raw, (int, float)):
        rcut_list = [float(rcut_raw)]
    else:
        rcut_list = [float(x) for x in rcut_raw]

    geoms = orbgen_cfg.get("geoms", [])
    lmax = 4
    if geoms and "lmaxmax" in geoms[0]:
        lmax = int(geoms[0]["lmaxmax"])

    return [(lmax, rc) for rc in rcut_list]


# ---------------------------------------------------------------------------
# 合并工具
# ---------------------------------------------------------------------------


def merge_input_overrides(
    *sources: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """合并多个 INPUT override 字典, 后面的覆盖前面的。"""
    out: Dict[str, Any] = {}
    for src in sources:
        if not src:
            continue
        out.update(src)
    return out


def merge_scheduler(
    *sources: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """合并多个 scheduler 字典, 后面的覆盖前面的。"""
    return merge_input_overrides(*sources)


# ---------------------------------------------------------------------------
# 默认 abacus.json 模板 (可作为 build_inputs 的回退)
# ---------------------------------------------------------------------------


DEFAULT_ABACUS_CONFIG: Dict[str, Any] = {
    "basis": ["pw", "lcao_nsw"],
    "abacus": {
        "code": "abacus_lts@yeesuan",
        "parameters": {
            "input": {
                "ks_solver": "scalapack_gvx",
            },
        },
        "metadata": {
            "options": {
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 56},
                "max_wallclock_seconds": 6 * 60 * 60,
                "queue_name": "q_ysuan",
                "max_memory_kb": 100 * 1024 * 1024,
                "withmpi": True,
            },
        },
    },
    "tolerance_meV": 4.2,
}


def with_default_abacus(
    abacus_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """把 ``abacus_cfg`` 合并到 ``DEFAULT_ABACUS_CONFIG`` 上 (覆盖默认)。

    abacus.json 现在用 ``abacus`` 键包裹所有 aiida-abacus 参数 (与 aiida-abacus
    ``test_abacus_base.py`` 一致):

    .. code-block:: json

        {
          "basis": ["pw", "lcao_nsw"],
          "abacus": {
            "code": "abacus_lts@yeesuan",
            "parameters": {"input": {"ecutwfc": 100, ...}},
            "metadata": {"options": {...}}
          },
          "tolerance_meV": 4.2,
        }
    """
    if not abacus_cfg:
        return dict(DEFAULT_ABACUS_CONFIG)

    # 获取默认和用户的 abacus 配置
    default_abacus = DEFAULT_ABACUS_CONFIG.get("abacus", {})
    user_abacus = abacus_cfg.get("abacus", {})

    # 合并 parameters.input (后者覆盖前者)
    default_input = default_abacus.get("parameters", {}).get("input", {})
    user_input = user_abacus.get("parameters", {}).get("input", {})

    # 合并 metadata.options
    default_options = default_abacus.get("metadata", {}).get("options", {})
    user_options = user_abacus.get("metadata", {}).get("options", {})

    out: Dict[str, Any] = {
        "basis": abacus_cfg.get("basis", DEFAULT_ABACUS_CONFIG["basis"]),
        "abacus": {
            "code": user_abacus.get("code", default_abacus.get("code", "abacus_lts@yeesuan")),
            "parameters": {
                "input": merge_input_overrides(default_input, user_input),
            },
            "metadata": {
                "options": merge_scheduler(default_options, user_options),
            },
        },
        "tolerance_meV": abacus_cfg.get(
            "tolerance_meV", DEFAULT_ABACUS_CONFIG["tolerance_meV"]
        ),
    }
    # 透传 metadata 的 label 和 description
    if "label" in user_abacus.get("metadata", {}):
        out["abacus"]["metadata"]["label"] = user_abacus["metadata"]["label"]
    if "description" in user_abacus.get("metadata", {}):
        out["abacus"]["metadata"]["description"] = user_abacus["metadata"]["description"]
    
    # 透传 max_l_max / max_r_cut (OrbgenGridSearchWorkChain 用)
    for key in ("max_l_max", "max_r_cut"):
        if key in abacus_cfg:
            out[key] = abacus_cfg[key]
    return out
