"""Result-node assembly for the orbgen WorkChains.

The WorkChains' outputs are plain ``Dict`` nodes; each one is built by a small
``@calcfunction`` so that the mapping from raw run data to the reported summary
is recorded (and cached) in provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from aiida.engine import calcfunction
from aiida.orm import Dict, List as AiiDA_List  # noqa: F401  (typing parity)

__all__ = [
    "create_energies_dict",
    "create_final_results",
    "create_grid_all_results",
    "create_grid_summary",
]


@calcfunction
def create_energies_dict(d: "Dict|dict|List") -> Dict:
    """Build the ``energies`` Dict a WorkChain returns.
    
    Parameters
    ----------
    d : Dict, dict, or List
        a plain dict, an AiiDA Dict, or an AiiDA List holding energy data
        
    Returns
    -------
    Dict
        the AiiDA Dict node
    """
    # a List means raw data collected from children: compute the deltas first
    if hasattr(d, "__iter__") and not hasattr(d, "get_dict"):
        # d is an AiiDA List: compute the energy differences
        energies_by_basis = {}
        for item in d:
            item_dict = item.get_dict() if hasattr(item, "get_dict") else dict(item)
            basis = item_dict.get("basis_type", "unknown")
            energy = item_dict.get("energy", item_dict.get("E_total"))
            folder = item_dict.get("folder", "unknown")
            pert = item_dict.get("pert")
            if energy is None:
                continue
            energies_by_basis.setdefault(basis, []).append({
                "folder": folder,
                "energy": float(energy),
                "pert": pert,
            })
        
        delta_per_struct = []
        if "pw" in energies_by_basis and "lcao" in energies_by_basis:
            pw_by_folder = {e["folder"]: e for e in energies_by_basis["pw"]}
            for e_lcao in energies_by_basis["lcao"]:
                folder = e_lcao["folder"]
                e_pw_entry = pw_by_folder.get(folder)
                if e_pw_entry is None:
                    continue
                dE = abs(e_lcao["energy"] - e_pw_entry["energy"])
                delta_per_struct.append({
                    "folder": folder,
                    "E_pw": e_pw_entry["energy"],
                    "E_lcao_nsw": e_lcao["energy"],
                    "dE": dE,
                })
        
        delta_max = max((x["dE"] for x in delta_per_struct), default=0.0)
        d_dict = {
            "energies": energies_by_basis,
            "delta_E_per_struct": delta_per_struct,
            "delta_E_max_eV": float(delta_max),
            "delta_E_max_meV": float(delta_max * 1000.0),
        }
    elif hasattr(d, "get_dict"):
        # an AiiDA Dict: unwrap it
        d_dict = d.get_dict()
    else:
        # a plain dict: use as-is
        d_dict = dict(d)
    
    return Dict(dict=d_dict)


@calcfunction
def create_final_results(
    l_max_val,
    r_cut_val,
    results_list,
) -> Dict:
    """Build the ``results`` Dict a WorkChain returns.

    Parameters
    ----------
    l_max_val : int or float
        highest angular momentum
    r_cut_val : int or float
        cutoff radius
    results_list : list
        per-child records (task, basis, pk, exit_status, ok)

    Returns
    -------
    Dict
        the AiiDA Dict node
    """
    # unwrap AiiDA Data values
    if hasattr(l_max_val, "value"):
        l_max_val = l_max_val.value
    if hasattr(r_cut_val, "value"):
        r_cut_val = r_cut_val.value
    # results_list is expected to be an AiiDA List of plain dicts
    # the AiiDA engine wraps a plain list in a List node automatically
    if hasattr(results_list, "get_list"):
        children = list(results_list.get_list())
    elif isinstance(results_list, (list, tuple)):
        children = list(results_list)
    else:
        children = results_list

    return Dict(dict={
        "l_max": int(l_max_val) if isinstance(l_max_val, (int, float)) else l_max_val,
        "r_cut": float(r_cut_val) if isinstance(r_cut_val, (int, float)) else r_cut_val,
        "children": children,
    })


@calcfunction
def create_grid_all_results(
    grid_results_list,
    tolerance_meV,
    search_strategy,
) -> Dict:
    """Build the grid search's ``all_results`` Dict.

    Parameters
    ----------
    grid_results_list : list
        each entry is a {l_max, r_cut, calc_pk, exit_status, ...} dict
    tolerance_meV : float
    search_strategy : str

    Returns
    -------
    Dict
    """
    tol = tolerance_meV.value if hasattr(tolerance_meV, "value") else float(tolerance_meV)
    strat = search_strategy.value if hasattr(search_strategy, "value") else str(search_strategy)
    if hasattr(grid_results_list, "get_list"):
        grid = list(grid_results_list.get_list())
    elif isinstance(grid_results_list, (list, tuple)):
        grid = list(grid_results_list)
    else:
        grid = grid_results_list
    return Dict(dict={
        "grid": grid,
        "tolerance_meV": float(tol),
        "search_strategy": str(strat),
    })


@calcfunction
def create_grid_summary(
    n_grid_points,
    n_tried,
    n_passed,
    n_failed,
    tolerance_meV,
    search_strategy,
    best_l_max=None,
    best_r_cut=None,
    best_delta_per_atom_meV=None,
    best_calc_pk=None,
) -> Dict:
    """Build the grid search's ``grid_summary`` Dict."""
    def get_val(x, conv=None):
        if x is None:
            return None
        if hasattr(x, "value"):
            return x.value
        return conv(x) if conv else x

    return Dict(dict={
        "n_grid_points": int(get_val(n_grid_points, int)),
        "n_tried": int(get_val(n_tried, int)),
        "n_passed": int(get_val(n_passed, int)),
        "n_failed": int(get_val(n_failed, int)),
        "best_l_max": get_val(best_l_max),
        "best_r_cut": get_val(best_r_cut),
        "best_delta_per_atom_meV": (
            float(get_val(best_delta_per_atom_meV, float))
            if get_val(best_delta_per_atom_meV) is not None
            else None
        ),
        "best_calc_pk": get_val(best_calc_pk),
        "tolerance_meV": float(get_val(tolerance_meV, float)),
        "search_strategy": str(get_val(search_strategy)),
    })
