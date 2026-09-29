"""
INPUT interface (the VASP INCAR analogue)

Generate ABACUS INPUT files.

Corresponding file: {folder}/INPUT
Example: U-dimer-1.89-9au/INPUT

This module is a high-level wrapper around SIAB and calls, underneath:
- ``SIAB.abacus.io.autoset``           : fill in the default DFT parameters
- ``SIAB.abacus.io.dftparam_to_text``  : dict → INPUT text

Why this is done:
1. Defaults always stay in sync with the SIAB mainline, so the interface cannot
   drift away from upstream.
2. The interface layer only owns the semantic mapping "JSON → ABACUS DFT
   parameters".
"""

import os
import re
from typing import Optional, Dict, Any

__all__ = ["generate_incar", "parse_incar"]


# Keep SIAB's behaviour: fit_basis='jy' forces an lcao basis set
def _resolve_basis_type(json_config: dict, dftparam: dict) -> str:
    """
    Infer basis_type from fit_basis.

    Convention in SIAB.driver.main.init:
        fit_basis == 'jy'  -> basis_type == 'lcao'
        fit_basis == 'pw'  -> basis_type == 'pw'
    """
    if "basis_type" in dftparam:
        return dftparam["basis_type"]
    fit_basis = json_config.get("fit_basis", "jy")
    return "lcao" if fit_basis == "jy" else "pw"


def _resolve_nbands(json_config: dict, dftparam: dict) -> Optional[int]:
    """
    Infer nbands.

    SIAB behaviour: nbands is inherited from the concrete geometry
    (geoms[i].nbands).
    If the JSON top level does not provide it, read it from geoms[0].
    """
    if "nbands" in dftparam:
        return dftparam["nbands"]
    geoms = json_config.get("geoms", [])
    if geoms and "nbands" in geoms[0]:
        return geoms[0]["nbands"]
    return None  # SIAB.autoset fills in the default value 'auto'


def _extract_dftparam(json_config: dict) -> dict:
    """
    Extract the ABACUS DFT input parameters from a SIAB JSON config.

    The whitelist ``SIAB.abacus.io.ABACUS_PARAMS`` filters out non-ABACUS fields
    (such as the SIAB-only ``element``/``ecutjy``/``bessel_nao_rcut``).
    """
    from SIAB.abacus.io import ABACUS_PARAMS

    ABACUS_PARAM_SET = set(ABACUS_PARAMS)
    dftparam = {k: v for k, v in json_config.items() if k in ABACUS_PARAM_SET}

    # Special-case inference
    dftparam["basis_type"] = _resolve_basis_type(json_config, dftparam)
    nbands = _resolve_nbands(json_config, dftparam)
    if nbands is not None:
        dftparam["nbands"] = nbands

    # pseudo_dir may be a path in the JSON; keep the original value
    # (autoset uses './' as the default)
    if "pseudo_dir" not in dftparam and "pseudo_dir" in json_config:
        dftparam["pseudo_dir"] = json_config["pseudo_dir"]

    return dftparam


def generate_incar(
    json_config: dict,
    output_path: str,
    auto_set: bool = True,
) -> str:
    """
    Generate ABACUS INPUT files.

    Parameters
    ----------
    json_config : dict
        SIAB JSON config dict (e.g. project/pbe/pbe_orbgen.json).
        Top-level ABACUS-related fields are passed through to the INPUT; the
        remaining fields are ignored.
    output_path : str
        Output file path
    auto_set : bool
        Whether to fill in the default parameters automatically (default True,
        equivalent to SIAB.autoset).

    Returns
    -------
    str
        Absolute path of the generated INPUT file

    Examples
    --------
    >>> import json
    >>> with open("pbe_orbgen.json") as f:
    ...     config = json.load(f)
    >>> generate_incar(config, output_path="./U-dimer-1.89-9au/INPUT")
    """
    # Import SIAB lazily so this module can be imported where SIAB is absent
    from SIAB.abacus.io import autoset as _siab_autoset
    from SIAB.abacus.io import dftparam_to_text as _siab_dftparam_to_text

    dftparam = _extract_dftparam(json_config)

    if auto_set:
        dftparam = _siab_autoset(dftparam)
    else:
        # No defaults are filled in here: turn a list such as
        # bessel_nao_rcut=[9] into a string so that ABACUS parses it correctly
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
    Read the parameters from an INPUT file.

    Parameters
    ----------
    filepath : str
        Path of the INPUT file

    Returns
    -------
    dict
        Parsed parameters (values are inferred as int/float/str where possible)

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
            # Try int, then float
            try:
                value: Any = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
            result[key] = value
    return result
