"""
AbacusDftCalc: read a JSON config and submit one ``abacus.base`` job.

The input format mirrors the ``abacus.base`` workchain exactly:

.. code-block:: python

    inputs = {
        "abacus": {
            "code": code,                              # Code
            "parameters": {"input": {...}},            # Dict; real INPUT under the 'input' key
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

Example JSON config
-------------------

.. code-block:: json

    {
      "profile": "default",
      "workchain": "abacus.base",
      "code": "abacus_lts@yeesuan",
      "pseudo_family": "apns-eff-fam",

      "input_file": "/abs/INPUT",                      // pick one of the two
      "input_params": { "ecutwfc": 100, ... },         // pick one of the two

      "structure_file": "/abs/STRU",                   // STRU file

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

Usage
-----

.. code-block:: python

    from aiida_orbgen.calculations.abacus import build_inputs_from_config, submit_from_config

    cfg = json.load(open("config.json"))
    inputs = build_inputs_from_config(cfg)

    if cfg.get("dry_run", True):
        # write a reference JSON for inspection
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
# JSON loading
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
# Building the inputs
# ---------------------------------------------------------------------------


def _build_input_dict(cfg: dict) -> Dict[str, Any]:
    """Build a ``parameters = {"input": {...}}`` dict from ``cfg``."""
    if "input_file" in cfg:
        return IncarData.from_file(cfg["input_file"]).get_dict()
    if "input_params" in cfg:
        return IncarData.from_dict(cfg["input_params"]).get_dict()
    raise ValueError("Config must contain 'input_file' or 'input_params'.")


def _build_structure_node(cfg: dict) -> StructureData:
    """Build ase.Atoms from ``cfg`` (for use by AiiDA StructureData)."""
    if "structure_file" not in cfg:
        raise ValueError("Config must contain 'structure_file' (path to STRU).")
    return StructureData.from_stru(cfg["structure_file"])


def _build_kpoints_node(cfg: dict) -> Optional[orm.KpointsData]:
    """Build a KpointsData from ``cfg``; return None when absent (abacus.base defaults apply)."""
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
    """Build the inputs dict of the ``abacus.base`` workchain from a JSON config.

    Parameters
    ----------
    cfg : dict
        Dict parsed from the JSON config (see the example at the top of the module).

    Returns
    -------
    dict
        A dict that can be passed straight to ``submit(AbacusBaseWorkChain, **inputs)``.
        Its ``abacus.structure`` entry is ``ase.Atoms``; on submission
        ``_materialize_inputs`` wraps it into an AiiDA ``StructureData``.

    Notes
    -----
    This function **only builds the inputs**: it never calls ``orm.load_code``
    and never connects to AiiDA, so it is safe to use in dry-run mode (without
    loading a profile).  The submission helper is what actually resolves the
    ``"code"`` string into a Code object via ``orm.load_code``.
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
        "structure": atoms,  # ase.Atoms; wrapped into StructureData on submission
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

    # K-points (optional)
    kp = _build_kpoints_node(cfg)
    if kp is not None:
        inputs["kpoints"] = kp

    return inputs


# ---------------------------------------------------------------------------
# Submission / dry-run
# ---------------------------------------------------------------------------


def _to_serializable(obj: Any) -> Any:
    """Convert the non-serializable objects in ``inputs`` into JSON-writable form (for dry-run)."""
    # numpy scalar -> native Python
    if hasattr(obj, "item") and callable(obj.item) and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass

    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    # numpy array
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    # ase.Atoms -- keep only the necessary fields
    if hasattr(obj, "get_chemical_symbols") and hasattr(obj, "get_cell"):
        return {
            "_type": "ase.Atoms",
            "symbols": list(obj.get_chemical_symbols()),
            "cell_angstrom": obj.get_cell().array.tolist(),
            "positions_angstrom": obj.get_positions().tolist(),
            "pbc": [bool(x) for x in obj.pbc],
        }
    # KpointsData -- serialize mesh / offset
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
    """Write ``inputs`` to ``output_dir/dry_run_inputs.json`` (for manual inspection)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "dry_run_inputs.json"

    serializable = _to_serializable(inputs)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    return out_path.resolve()


def _materialize_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Materialize the ``atoms`` / code string in ``inputs`` into AiiDA nodes / Code."""
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
    """Submit one ``abacus.base`` workchain from a config.

    Returns
    -------
    int
        PK of the submitted node.
    """
    workchain_name = cfg.get("workchain", "abacus.base")
    WC = WorkflowFactory(workchain_name)

    inputs = build_inputs_from_config(cfg)
    inputs = _materialize_inputs(inputs)

    from aiida.engine import submit
    node = submit(WC, **inputs)
    return node.pk


# ---------------------------------------------------------------------------
# Three-layer entry point: interfaces/ writes INPUT/STRU, data/ reads them into AiiDA nodes
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
    """Build an abacus.base JSON config from already generated INPUT + STRU files.

    **This is the top-level entry point of the three-layer architecture**:

    1. ``interfaces/`` has already written ``input_file`` (INPUT) and ``structure_file`` (STRU);
    2. this function reads them back into AiiDA data nodes with
       ``data.input.IncarData.from_file`` and ``data.structure.StructureData.from_stru``;
    3. it assembles an ``abacus.base`` JSON config (writable to disk and submittable).

    Parameters
    ----------
    input_file : str
        Path to the ABACUS INPUT file (usually generated by ``interfaces.incar.generate_incar``).
    structure_file : str
        Path to the ABACUS STRU file (usually generated by ``interfaces.stru.generate_stru``).
    code : str
        AiiDA code label, e.g. ``abacus_lts@yeesuan``.
    pseudo_family : str
        AiiDA pseudo family label.
    label, description : str
        Workflow label / description.
    kpoints_mesh, kpoints_offset : list, optional
        K-point mesh.  Defaults to ``[1, 1, 1]`` (Gamma-only, the SIAB convention).
    metadata_options : dict, optional
        AiiDA scheduler options (resources, queue_name, ...).
    orbital_path : str, optional
        Path to the orbital file (recorded in input_params.orbital_dir; only needed for LCAO).
    output_dir : str
        Output directory of the dry-run JSON (used by ``aiida-orbgen run --dry-run``).

    Returns
    -------
    dict
        The JSON config accepted by the ``abacus.base`` workchain.
    """
    from aiida_orbgen.data.input import IncarData
    from aiida_orbgen.data.structure import StructureData

    # 1) read the files back into AiiDA data nodes through data/
    incar = IncarData.from_file(input_file)
    sd = StructureData.from_stru(structure_file)

    # 2) extract the INPUT parameter dict (i.e. the parameters.input abacus.base needs)
    input_params = incar.get_dict()
    # add orbital_dir by hand for LCAO
    if orbital_path and "orbital_dir" not in input_params:
        input_params["orbital_dir"] = orbital_path

    # 3) assemble the JSON config
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
