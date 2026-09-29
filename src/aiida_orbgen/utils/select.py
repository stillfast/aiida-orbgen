"""Record which grid point a report chose, and make that choice reusable.

The workflow already answers "which (l_max, r_cut) is acceptable?" — the report
prints the ΔE matrices and `select_best_point()` picks the cheapest acceptable
point.  What was missing is the last mile: writing that decision down somewhere a
*next* run can consume, instead of a human copying two numbers into a YAML preset
(where they are easy to get wrong, and leave no trace of why).

`aiida-orbgen select` does that:

1. read the ``output.json`` of a `run`, load the grid-search node, collect the
   summary and choose a point — automatically (first acceptable, grid order) or
   explicitly with ``--l-max/--r-cut`` / ``--calc-pk``;
2. materialise the **validated** SIAB config of that point (via
   :class:`aiida_orbgen.spec.OrbgenSpec`, so an unreachable ``nzeta`` or a
   misplaced ``vloc_aux`` fails here rather than inside SIAB);
3. write a small bundle next to the report:

   ``<dir>/selected.json``          the decision (point, ΔE, tolerance, node, config)
   ``<dir>/orbgen_<point>.json``    the SIAB config, ready for ``report --siab-json``
   ``<dir>/input.selected.json``    an ``input.json`` pinned to that candidate:
                                    the point's config goes inline as
                                    ``static.siab_config``, so ``run`` submits a
                                    single ``orbgen.calc`` with no preset edit

Nothing in ``parameters/`` is rewritten: that tree is user input, and a preset
carries more than the two numbers a grid search decides.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Selection",
    "choose_point",
    "selection_payload",
    "write_selection",
]


@dataclass
class Selection:
    """The chosen grid point, with the numbers that justify it."""

    l_max: int | None
    r_cut: float | None
    calc_pk: int
    node_pk: int
    node_uuid: str
    delta_per_atom_meV: float | None = None
    delta_max_meV: float | None = None
    tolerance_meV: float | None = None
    chosen_by: str = "auto"
    grid_points: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.l_max is not None and self.r_cut is not None

    def describe(self) -> str:
        delta = "n/a" if self.delta_per_atom_meV is None else \
            f"{self.delta_per_atom_meV:.3f} meV/atom"
        return (f"l_max={self.l_max}, r_cut={self.r_cut} "
                f"(OrbgenCalcWorkChain<{self.calc_pk}>, ΔE/atom={delta}, "
                f"chosen by {self.chosen_by})")


def _point_record(point) -> dict[str, Any]:
    return {
        "l_max": point.l_max,
        "r_cut": point.r_cut,
        "calc_pk": point.pk,
        "delta_per_atom_meV": point.delta_max_per_atom_meV,
        "delta_max_meV": point.delta_max_meV,
        "tolerance_meV": point.tolerance_meV,
        "acceptable": (
            point.delta_max_per_atom_meV is not None
            and point.tolerance_meV is not None
            and point.delta_max_per_atom_meV <= point.tolerance_meV
        ),
    }


def choose_point(
    summary,
    *,
    l_max: int | None = None,
    r_cut: float | None = None,
    calc_pk: int | None = None,
):
    """Pick a grid point: an explicit request first, otherwise the reported best.

    Explicit ``l_max``/``r_cut`` (both must be given) or ``calc_pk`` override the
    automatic choice — useful when the tolerance was never met and the user
    knowingly takes one of the computed points anyway.

    The automatic choice is exactly the one the report recommends, i.e.
    :func:`~aiida_orbgen.utils.report.orbgen.select_best_point`: the point the grid
    search itself judged acceptable (``summary.best``, which the WorkChain only sets
    after a tolerance pass), falling back to the lowest ΔE when no point passed.
    Whether the chosen point really met the tolerance is recorded per point in the
    bundle rather than silently assumed.
    """
    from aiida_orbgen.utils.report.orbgen import select_best_point

    grid = list(getattr(summary, "grid", []) or [])
    if not grid:
        return None
    if calc_pk is not None:
        return next((p for p in grid if p.pk == calc_pk), None)
    if l_max is not None or r_cut is not None:
        if l_max is None or r_cut is None:
            raise ValueError("--l-max and --r-cut must be given together")
        wanted = (int(l_max), float(r_cut))
        return next(
            (p for p in grid
             if (p.l_max, float(p.r_cut)) == wanted),
            None,
        )
    return select_best_point(summary)


def selection_payload(
    summary,
    point,
    *,
    chosen_by: str = "auto",
    siab_config: dict | None = None,
    base_input: dict | None = None,
) -> dict[str, Any]:
    """Everything the selection bundle contains (also used by the CLI printer)."""
    selection = Selection(
        l_max=point.l_max,
        r_cut=point.r_cut,
        calc_pk=point.pk,
        node_pk=summary.node_pk,
        node_uuid=summary.node_uuid,
        delta_per_atom_meV=point.delta_max_per_atom_meV,
        delta_max_meV=point.delta_max_meV,
        tolerance_meV=point.tolerance_meV,
        chosen_by=chosen_by,
        grid_points=[_point_record(p) for p in (getattr(summary, "grid", []) or [])],
    )
    payload: dict[str, Any] = {
        "selection": asdict(selection),
        "description": selection.describe(),
        "report_node": {"pk": summary.node_pk, "uuid": summary.node_uuid},
    }
    if siab_config is not None:
        payload["siab_config"] = siab_config
    if base_input is not None:
        payload["input_json"] = _single_candidate_input(
            base_input, selection, siab_config
        )
    return payload


def _single_candidate_input(base_input: dict, selection: Selection,
                            siab_config: dict | None = None) -> dict:
    """An ``input.json`` pinned to this one grid point.

    Derived from the ``input.json`` the run used (so codes, profile, scheduler and
    paths stay identical) with two changes:

    * ``static.siab_config`` gets this point's SIAB config — the same dict written
      as ``orbgen_<point>.json``, already carrying the point's own
      ``bessel_nao_rcut`` / ``lmaxmax``.  ``ConfigLoader`` reads that key *instead
      of* ``parameters.orbgen``, so the loader resolves exactly one candidate and
      picks ``orbgen.calc`` without any preset being edited (and validates the
      inline config like a preset).  Leaving the preset name in place was wrong:
      it made the bundle look runnable while pointing at a preset that does not
      exist.
    * ``static.selected_point`` records where the choice came from — informational
      (``selected.json`` is the full record), kept because an ``input.json`` should
      explain itself.
    """
    out = json.loads(json.dumps(base_input or {}))
    static = out.setdefault("static", {})
    if siab_config is not None:
        static["siab_config"] = json.loads(json.dumps(siab_config))
    static["selected_point"] = {
        "l_max": selection.l_max,
        "r_cut": selection.r_cut,
        "from_node": selection.calc_pk,
    }
    return out


def write_selection(
    out_dir: str | Path,
    payload: dict[str, Any],
    *,
    siab_config: dict | None = None,
    point_name: str = "selected",
) -> dict[str, str]:
    """Write the selection bundle.  Returns a ``{kind: path}`` mapping."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    selection_path = out_dir / "selected.json"
    selection_path.write_text(
        json.dumps({k: v for k, v in payload.items()
                    if k not in ("siab_config", "input_json")},
                   indent=4, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written["selection"] = str(selection_path)

    if siab_config is not None:
        config_path = out_dir / f"orbgen_{point_name}.json"
        config_path.write_text(json.dumps(siab_config, indent=4) + "\n",
                               encoding="utf-8")
        written["siab_config"] = str(config_path)

    if payload.get("input_json"):
        input_path = out_dir / "input.selected.json"
        input_path.write_text(json.dumps(payload["input_json"], indent=4) + "\n",
                              encoding="utf-8")
        written["input_json"] = str(input_path)
    return written
