"""
aiida_orbgen.static.json_inputs - load abacus/orbgen configuration from JSON files

Provides two core functions:

- :func:`load_orbgen_config`  : load a ``pbe_orbgen.json``-style SIAB configuration
- :func:`load_abacus_config`  : load an ``abacus.json``-style ABACUS configuration
- :func:`parse_lmax_rcut_candidates` : extract the (l_max, r_cut) candidate list from orbgen.json

JSON schema
-----------

**abacus.json** (new file, adjusts the ABACUS INPUT + decides basis + tolerance):

abacus.json now wraps all aiida-abacus related parameters under the ``abacus``
key (consistent with aiida-abacus ``test_abacus_base.py``), structured as follows:

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

      "tolerance_meV": 100.0,
      "max_l_max": 5,
      "max_r_cut": 12.0
    }

**orbgen.json** (the existing pbe_orbgen.json, single l_max/r_cut combination):

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
# JSON loading
# ---------------------------------------------------------------------------


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a dict from a .json / .yaml file."""
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
    """Load a ``pbe_orbgen.json``-style SIAB configuration.

    Returns
    -------
    dict
        The raw JSON dictionary (containing element / pseudo_dir / geoms / ...)

    Raises
    ------
    ValueError
        A required field is missing
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
    """Load an ``abacus.json``-style ABACUS configuration.

    Returns
    -------
    dict
        Contains::

            {
              "basis": ["pw", "lcao_nsw"],
              "input_overrides": {...},
              "tolerance_meV": 100.0,
              "scheduler": {"queue_name": ..., ...}
            }

    Raises
    ------
    ValueError
        The basis field is empty or missing
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
# Candidate parameter parsing
# ---------------------------------------------------------------------------


def parse_lmax_rcut_candidates(
    orbgen_cfg: Dict[str, Any],
) -> List[Tuple[int, float]]:
    """Extract the ``(l_max, r_cut)`` candidate list from orbgen.json.

    Current usage: a single combination taken directly from ``bessel_nao_rcut``
    + ``geoms[0].lmaxmax``::

        "bessel_nao_rcut": [9],
        "geoms[0].lmaxmax": 4
        → [(4, 9.0)]

    Historical rules (kept, not enabled for now):
        1) an explicit ``basis_candidates`` list
        2) the Cartesian product ``lmax_candidates × rcut_candidates``
        3) ``bessel_nao_rcut`` + ``lmaxmax`` (current)

    Returns
    -------
    list of (l_max, r_cut) tuples
    """
    # 1) explicit list (kept for future extension)
    explicit = orbgen_cfg.get("basis_candidates")
    if explicit:
        out: List[Tuple[int, float]] = []
        for item in explicit:
            lmax = int(item["lmax"])
            rcut = float(item["rcut"])
            out.append((lmax, rcut))
        return out

    # 2) Cartesian product (kept for future extension)
    lmax_list = orbgen_cfg.get("lmax_candidates")
    rcut_list = orbgen_cfg.get("rcut_candidates")
    if lmax_list and rcut_list:
        return [
            (int(lm), float(rc))
            for lm in lmax_list
            for rc in rcut_list
        ]

    # 3) default: use bessel_nao_rcut + lmaxmax (single combination)
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
# Merge helpers
# ---------------------------------------------------------------------------


def merge_input_overrides(
    *sources: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge several INPUT override dictionaries, later ones win."""
    out: Dict[str, Any] = {}
    for src in sources:
        if not src:
            continue
        out.update(src)
    return out


def merge_scheduler(
    *sources: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge several scheduler dictionaries, later ones win."""
    return merge_input_overrides(*sources)


# ---------------------------------------------------------------------------
# Default abacus.json template (usable as a fallback for build_inputs)
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
    "tolerance_meV": 100.0,
}


def with_default_abacus(
    abacus_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge ``abacus_cfg`` on top of ``DEFAULT_ABACUS_CONFIG`` (user wins).

    abacus.json now wraps all aiida-abacus parameters under the ``abacus`` key
    (consistent with aiida-abacus ``test_abacus_base.py``):

    .. code-block:: json

        {
          "basis": ["pw", "lcao_nsw"],
          "abacus": {
            "code": "abacus_lts@yeesuan",
            "parameters": {"input": {"ecutwfc": 100, ...}},
            "metadata": {"options": {...}}
          },
          "tolerance_meV": 100.0,
        }
    """
    if not abacus_cfg:
        return dict(DEFAULT_ABACUS_CONFIG)

    # Fetch the default and the user-provided abacus configuration
    default_abacus = DEFAULT_ABACUS_CONFIG.get("abacus", {})
    user_abacus = abacus_cfg.get("abacus", {})

    # Merge parameters.input (later wins)
    default_input = default_abacus.get("parameters", {}).get("input", {})
    user_input = user_abacus.get("parameters", {}).get("input", {})

    # Merge metadata.options
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
    # Pass through metadata label and description
    if "label" in user_abacus.get("metadata", {}):
        out["abacus"]["metadata"]["label"] = user_abacus["metadata"]["label"]
    if "description" in user_abacus.get("metadata", {}):
        out["abacus"]["metadata"]["description"] = user_abacus["metadata"]["description"]
    
    # Pass through max_l_max / max_r_cut (used by OrbgenGridSearchWorkChain)
    for key in ("max_l_max", "max_r_cut"):
        if key in abacus_cfg:
            out[key] = abacus_cfg[key]
    return out
