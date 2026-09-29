"""Result-node assembly for the orbgen WorkChains.

The WorkChains' outputs are plain ``Dict`` nodes; each one is built by a small
``@calcfunction`` so that the mapping from raw run data to the reported summary
is recorded (and cached) in provenance.

Nothing here computes physics: the ΔE arithmetic has one implementation, in
:mod:`aiida_orbgen.workflows.energies`.  ``create_energies_dict`` used to carry a
second copy of it (for the case of being handed an ``AiiDA List`` of raw child
records); the only caller passes the dict ``pair_energies()`` produced, so the copy
was removed rather than left as a way for two runs to disagree.
"""

from __future__ import annotations

from aiida.engine import calcfunction
from aiida.orm import Dict

__all__ = [
    "create_energies_dict",
    "create_final_results",
    "create_grid_all_results",
    "create_grid_summary",
]


@calcfunction
def create_energies_dict(d: dict) -> Dict:
    """Build the ``energies`` Dict a WorkChain returns.

    Parameters
    ----------
    d : dict or Dict
        the dict returned by :func:`aiida_orbgen.workflows.energies.pair_energies`,
        either plain or already wrapped in an AiiDA ``Dict``.

    Returns
    -------
    Dict
        the AiiDA Dict node
    """
    if hasattr(d, "get_dict"):     # already an AiiDA Dict
        return Dict(dict=d.get_dict())
    return Dict(dict=dict(d))


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
