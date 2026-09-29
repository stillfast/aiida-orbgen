"""
AbacusDftCalc: 读取 JSON 配置, 提交一个 ``abacus.base`` 任务。

输入格式严格对齐 ``abacus.base`` workchain:

.. code-block:: python

    inputs = {
        "abacus": {
            "code": code,                              # Code
            "parameters": {"input": {...}},            # Dict, 实际 INPUT 在 'input' 子键下
            "structure": StructureData(ase=atoms),     # StructureData
            "metadata": {
                "options": {                           # resources, queue, ...
                    "resources": {...},
                    "max_wallclock_seconds": 600,
                    "queue_name": "...",
                    ...
                },
                "label": "...",
                "description": "...",
            },
        },
        "kpoints": kpoints,                            # KpointsData
        "pseudo_family": "apns-eff-fam",               # str
    }

JSON config 示例
----------------

.. code-block:: json

    {
      "profile": "default",
      "workchain": "abacus.base",
      "code": "abacus_lts@yeesuan",
      "pseudo_family": "apns-eff-fam",

      "input_file": "/abs/INPUT",                      // 二选一
      "input_params": { "ecutwfc": 100, ... },         // 二选一

      "structure_file": "/abs/STRU",                   // STRU 文件

      "kpoints_mesh": [4, 4, 4],
      "kpoints_offset": [0, 0, 0],

      "label": "abacus-scf-test",
      "description": "test",
      "metadata": {
        "options": {
          "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 56, "tot_num_mpiprocs": 56},
          "max_wallclock_seconds": 600,
          "queue_name": "q_ysuan",
          "scheduler_stderr": "error.log",
          "scheduler_stdout": "output.log",
          "max_memory_kb": 50331648,
          "withmpi": true
        }
      },

      "dry_run": true,
      "output_dir": "./dry_run_submissions"
    }

使用
----

.. code-block:: python

    from aiida_orbgen.calculations.abacus import build_inputs_from_config, submit_from_config

    cfg = json.load(open("config.json"))
    inputs = build_inputs_from_config(cfg)

    if cfg.get("dry_run", True):
        # 写出参考 JSON 供检查
        write_dry_run(inputs, cfg["output_dir"])
    else:
        submit_from_config(cfg)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from aiida import orm
from aiida.plugins import WorkflowFactory

from aiida_orbgen.data.input import IncarData
from aiida_orbgen.data.structure import StructureData

__all__ = [
    "build_inputs_from_config",
    "submit_from_config",
    "write_dry_run",
    "load_config",
    "build_abacus_json_from_files",
]


# ---------------------------------------------------------------------------
# JSON 加载
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict:
    """Load JSON (or YAML if available) config."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("YAML config requires PyYAML") from exc
        return yaml.safe_load(text)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Inputs 构造
# ---------------------------------------------------------------------------


def _build_input_dict(cfg: dict) -> Dict[str, Any]:
    """从 cfg 构造 ``parameters = {"input": {...}}`` 形式的 dict。"""
    if "input_file" in cfg:
        return IncarData.from_file(cfg["input_file"]).get_dict()
    if "input_params" in cfg:
        return IncarData.from_dict(cfg["input_params"]).get_dict()
    raise ValueError("Config must contain 'input_file' or 'input_params'.")


def _build_structure_node(cfg: dict) -> StructureData:
    """从 cfg 构造 ase.Atoms (供 AiiDA StructureData 使用)。"""
    if "structure_file" not in cfg:
        raise ValueError("Config must contain 'structure_file' (path to STRU).")
    return StructureData.from_stru(cfg["structure_file"])


def _build_kpoints_node(cfg: dict) -> Optional[orm.KpointsData]:
    """从 cfg 构造 KpointsData, 没有则返回 None (使用 abacus.base 默认值)。"""
    if "kpoints_mesh" in cfg:
        kp = orm.KpointsData()
        kp.set_kpoints_mesh(
            cfg["kpoints_mesh"],
            offset=cfg.get("kpoints_offset", [0, 0, 0]),
        )
        return kp
    if "kpoints" in cfg:
        kp = orm.KpointsData()
        kp.set_kpoints(cfg["kpoints"])
        return kp
    return None


def build_inputs_from_config(cfg: dict) -> Dict[str, Any]:
    """从 JSON config 构造 ``abacus.base`` workchain 的 inputs dict。

    Parameters
    ----------
    cfg : dict
        JSON config 解析后的 dict (见模块顶部示例)。

    Returns
    -------
    dict
        可直接 ``submit(AbacusBaseWorkChain, **inputs)`` 的字典。
        其中 ``abacus.structure`` 是 ``ase.Atoms``, 提交时
        ``_materialize_inputs`` 会包装成 AiiDA ``StructureData``。

    Notes
    -----
    本函数**只构造 inputs**, 不调用 ``orm.load_code`` / 不连接 AiiDA,
    因此可以安全地在 dry-run 模式 (不加载 profile) 下使用。
    真正提交时, 提交函数会用 ``orm.load_code`` 把 ``"code"`` 字符串
    解析为 Code 对象。
    """
    if "code" not in cfg:
        raise ValueError("Config must contain 'code' (AiiDA code label).")
    if "pseudo_family" not in cfg:
        raise ValueError("Config must contain 'pseudo_family'.")

    input_params = _build_input_dict(cfg)
    sd = _build_structure_node(cfg)
    atoms = sd.to_ase()

    label = cfg.get("label", "abacus-scf")
    description = cfg.get("description", "")
    metadata = cfg.get("metadata", {"options": {}})

    abacus_inputs: Dict[str, Any] = {
        "code": cfg["code"],
        "parameters": {"input": input_params},
        "structure": atoms,  # ase.Atoms, 提交时再包装成 StructureData
        "metadata": {
            "options": metadata.get("options", {}),
            "label": label,
            "description": description,
        },
    }

    inputs: Dict[str, Any] = {
        "abacus": abacus_inputs,
        "pseudo_family": cfg["pseudo_family"],
    }

    # K 点 (可选)
    kp = _build_kpoints_node(cfg)
    if kp is not None:
        inputs["kpoints"] = kp

    return inputs


# ---------------------------------------------------------------------------
# 提交 / dry-run
# ---------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    """把 inputs 里的不可序列化对象转成 JSON 可写的形式 (供 dry-run)。"""
    # numpy 标量 -> Python 原生
    if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    # numpy 数组
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    # ase.Atoms — 只存必要字段
    if hasattr(obj, "get_chemical_symbols") and hasattr(obj, "get_cell"):
        return {
            "_type": "ase.Atoms",
            "symbols": list(obj.get_chemical_symbols()),
            "cell_angstrom": obj.get_cell().array.tolist(),
            "positions_angstrom": obj.get_positions().tolist(),
            "pbc": [bool(x) for x in obj.pbc],
        }
    # KpointsData — 序列化 mesh / offset
    if isinstance(obj, orm.KpointsData):
        try:
            mesh, offset = obj.get_kpoints_mesh()
            return {
                "_type": "KpointsData",
                "mesh": list(mesh),
                "offset": list(offset),
            }
        except Exception:
            return {"_type": "KpointsData", "repr": repr(obj)}
    return obj


def write_dry_run(inputs: Dict[str, Any], output_dir: str | Path) -> Path:
    """把 inputs 写到 ``output_dir/dry_run_inputs.json`` (供人工检查)。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dry_run_inputs.json"

    serializable = _to_serializable(inputs)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    return out_path.resolve()


def _materialize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """把 inputs 里的 ``atoms`` / code 字符串实例化成 AiiDA 节点 / Code。"""
    abacus = dict(inputs["abacus"])

    # code: str -> Code
    abacus["code"] = orm.load_code(abacus["code"])

    # structure: ase.Atoms -> StructureData
    if not isinstance(abacus["structure"], orm.StructureData):
        abacus["structure"] = orm.StructureData(ase=abacus["structure"])

    # parameters: dict -> Dict
    if not isinstance(abacus["parameters"], orm.Dict):
        abacus["parameters"] = orm.Dict(dict=abacus["parameters"])

    out = {"abacus": abacus, "pseudo_family": inputs["pseudo_family"]}
    if "kpoints" in inputs:
        out["kpoints"] = inputs["kpoints"]
    return out


def submit_from_config(cfg: dict) -> int:
    """从 config 提交一个 ``abacus.base`` workchain。

    Returns
    -------
    int
        提交后节点的 PK。
    """
    workchain_name = cfg.get("workchain", "abacus.base")
    WC = WorkflowFactory(workchain_name)

    inputs = build_inputs_from_config(cfg)
    inputs = _materialize_inputs(inputs)

    from aiida.engine import submit
    node = submit(WC, **inputs)
    return node.pk


# ---------------------------------------------------------------------------
# 3 层架构的入口: 用 interfaces/ 生成的 INPUT/STRU 文件, 经 data/ 转 AiiDA 节点
# ---------------------------------------------------------------------------


def build_abacus_json_from_files(
    *,
    input_file: str,
    structure_file: str,
    code: str,
    pseudo_family: str,
    label: str = "abacus-scf",
    description: str = "",
    kpoints_mesh: Optional[list] = None,
    kpoints_offset: Optional[list] = None,
    metadata_options: Optional[Dict[str, Any]] = None,
    orbital_path: Optional[str] = None,
    output_dir: str = "./dry_run_submissions",
) -> Dict[str, Any]:
    """用已生成的 INPUT + STRU 文件, 构造 abacus.base JSON config。

    **这是 3 层架构的顶层入口**:

    1. ``interfaces/`` 已写出 ``input_file`` (INPUT) 和 ``structure_file`` (STRU);
    2. 本函数用 ``data.input.IncarData.from_file`` 和
       ``data.structure.StructureData.from_stru`` 把它们读回 AiiDA 数据节点;
    3. 拼装一个 ``abacus.base`` JSON config (可写盘, 可提交)。

    Parameters
    ----------
    input_file : str
        ABACUS INPUT 文件路径 (通常由 ``interfaces.incar.generate_incar`` 生成)。
    structure_file : str
        ABACUS STRU 文件路径 (通常由 ``interfaces.stru.generate_stru`` 生成)。
    code : str
        AiiDA code 标签, 例如 ``abacus_lts@yeesuan``。
    pseudo_family : str
        AiiDA pseudo 族标签。
    label, description : str
        工作流标签 / 描述。
    kpoints_mesh, kpoints_offset : list, optional
        K 点网格。默认 ``[1, 1, 1]`` (Gamma-only, SIAB 习惯)。
    metadata_options : dict, optional
        AiiDA scheduler options (resources, queue_name, ...)。
    orbital_path : str, optional
        轨道文件路径 (记录在 input_params.orbital_dir, 仅 LCAO 需要)。
    output_dir : str
        dry-run JSON 的输出目录 (用于 ``aiida-orbgen run --dry-run``)。

    Returns
    -------
    dict
        ``abacus.base`` workchain 接受的 JSON config。
    """
    from aiida_orbgen.data.input import IncarData
    from aiida_orbgen.data.structure import StructureData

    # 1) 用 data/ 读回 AiiDA 数据节点
    incar = IncarData.from_file(input_file)
    sd = StructureData.from_stru(structure_file)

    # 2) 提取 INPUT 参数字典 (即 abacus.base 需要的 parameters.input)
    input_params = incar.get_dict()
    # LCAO 时手动加 orbital_dir
    if orbital_path and "orbital_dir" not in input_params:
        input_params["orbital_dir"] = orbital_path

    # 3) 拼装 JSON config
    if kpoints_mesh is None:
        kpoints_mesh = [1, 1, 1]
    if kpoints_offset is None:
        kpoints_offset = [0, 0, 0]

    if metadata_options is None:
        metadata_options = {
            "resources": {
                "num_machines": 1,
                "num_mpiprocs_per_machine": 24,
                "tot_num_mpiprocs": 24,
            },
            "max_wallclock_seconds": 600,
            "withmpi": True,
        }

    return {
        "profile": "default",
        "workchain": "abacus.base",
        "code": code,
        "pseudo_family": pseudo_family,
        "input_params": input_params,
        "structure_file": os.path.abspath(structure_file),
        "kpoints_mesh": kpoints_mesh,
        "kpoints_offset": kpoints_offset,
        "label": label,
        "description": description,
        "metadata": {"options": metadata_options},
        "dry_run": True,
        "output_dir": output_dir,
    }
