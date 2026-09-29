"""Shared helpers of the ``aiida-orbgen`` CLI (``run`` / ``report`` / ``check``).

Everything workflow-specific lives in :data:`METHOD_SPECS`, so the CLI code
itself stays thin — mirroring ``aiida_uranium_workflow.cli._common``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiida_orbgen.utils.cal_json import SubmittedJob, default_result_path
from aiida_orbgen.utils.config import (
    ParamBundle,
    PresetEntry,
    WORKFLOW_CALC,
    WORKFLOW_GRIDSEARCH,
)

__all__ = [
    "MethodSpec",
    "METHOD_SPECS",
    "SUPPORTED_METHODS",
    "RunPlan",
    "get_method_spec",
    "resolve_method",
    "plan_runs",
    "build_workchain_inputs",
    "submit_plans",
    "generate_one_report",
    "default_result_path",
    "SubmittedJob",
    "WORKFLOW_CALC",
    "WORKFLOW_GRIDSEARCH",
]


# ---------------------------------------------------------------------------
#  Method registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MethodSpec:
    """Per-method bits the CLI needs (entry point ↔ WorkChain class ↔ key)."""

    name: str                      # "orbgen.calc" / "orbgen.gridsearch"
    entry_point: str               # AiiDA workflow entry point
    class_name: str                # WorkChain class name
    key: str = "orbgen"            # second-level key in output.json


METHOD_SPECS: dict[str, MethodSpec] = {
    WORKFLOW_CALC: MethodSpec(
        name=WORKFLOW_CALC,
        entry_point="orbgen.calc",
        class_name="OrbgenCalcWorkChain",
    ),
    WORKFLOW_GRIDSEARCH: MethodSpec(
        name=WORKFLOW_GRIDSEARCH,
        entry_point="orbgen.gridsearch",
        class_name="OrbgenGridSearchWorkChain",
    ),
}

SUPPORTED_METHODS: tuple[str, ...] = tuple(METHOD_SPECS)


def get_method_spec(method: str) -> MethodSpec:
    try:
        return METHOD_SPECS[method]
    except KeyError as exc:
        raise ValueError(
            f"Unknown workflow '{method}'. Supported: {list(METHOD_SPECS)}"
        ) from exc


def _read_workflow_field(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get("workflow")
    return value if isinstance(value, str) and value else None


def resolve_method(
    *,
    cli_method: str | None = None,
    input_json: str | Path | None = None,
    output_json: str | Path | None = None,
) -> str:
    """Pick the workflow name: CLI flag > output.json > input.json."""
    for candidate, source in (
        (cli_method, "--workflow"),
        (_read_workflow_field(output_json), "output.json['workflow']"),
        (_read_workflow_field(input_json), "input.json['workflow']"),
    ):
        if not candidate:
            continue
        if candidate not in METHOD_SPECS:
            raise ValueError(
                f"{source}={candidate!r} is not a known workflow. "
                f"Supported: {list(METHOD_SPECS)}"
            )
        return candidate
    # No explicit workflow: fall back to the loader's auto-detection.
    return ""


# ---------------------------------------------------------------------------
#  Run planning / submission
# ---------------------------------------------------------------------------


@dataclass
class RunPlan:
    """One WorkChain to submit (one abacus preset × one orbgen preset)."""

    workflow: str
    abacus: PresetEntry
    orbgen: PresetEntry
    abacus_config: dict
    siab_json_path: Path
    output_dir: Path
    candidates: list[tuple[int, float]]
    preset_name: str
    code_label: str | None = None
    search_strategy: str = "exhaustive"

    @property
    def entry_point(self) -> str:
        return get_method_spec(self.workflow).entry_point

    def describe(self) -> str:
        if self.workflow == WORKFLOW_CALC:
            l_max, r_cut = self.candidates[0]
            return f"l_max={l_max}, r_cut={r_cut:g}"
        return (
            f"{len(self.candidates)} candidate(s): "
            + ", ".join(f"(l_max={lm}, r_cut={rc:g})" for lm, rc in self.candidates)
        )


def _preset_label(abacus: PresetEntry, orbgen: PresetEntry, multiple_abacus: bool) -> str:
    return f"{abacus.name}@{orbgen.name}" if multiple_abacus else orbgen.name


def plan_runs(
    bundle: ParamBundle,
    *,
    output_root: str | Path | None = None,
    workflow: str | None = None,
    only: int | None = None,
) -> list[RunPlan]:
    """Resolve every ``(abacus preset × orbgen preset)`` pair into a plan.

    Pure filesystem work: the SIAB config of each orbgen preset is written to
    ``<output_root>/configs/orbgen_<preset>.json`` so what is submitted is
    exactly what can be inspected later.
    """
    if output_root is None:
        output_root = bundle.output_root
    if output_root is None:
        base = bundle.input_json.parent if bundle.input_json else Path.cwd()
        output_root = base / "run"
    output_root = Path(output_root).expanduser().resolve()
    (output_root / "configs").mkdir(parents=True, exist_ok=True)

    workflow = workflow or bundle.workflow
    method = get_method_spec(workflow)
    multiple_abacus = len(bundle.abacus_presets) > 1

    plans: list[RunPlan] = []
    for abacus in bundle.abacus_presets:
        for orbgen in bundle.orbgen_presets:
            candidates = bundle.candidates(orbgen)
            if method.name == WORKFLOW_CALC and len(candidates) > 1:
                candidates = candidates[only or 0: (only or 0) + 1]
            if not candidates:
                raise ValueError(
                    f"orbgen preset '{orbgen.name}' produced no (l_max, r_cut) "
                    f"candidate (check bessel_nao_rcut / lmaxmax / max_r_cut)"
                )

            siab_path = output_root / "configs" / f"orbgen_{orbgen.name}.json"
            siab_path.write_text(
                json.dumps(orbgen.config, indent=4) + "\n", encoding="utf-8"
            )

            if method.name == WORKFLOW_GRIDSEARCH:
                plan_dir = output_root / orbgen.name
            else:
                l_max, r_cut = candidates[0]
                plan_dir = output_root / orbgen.name / f"lmax{l_max}_rcut{str(r_cut).replace('.', 'p')}"
            plan_dir.mkdir(parents=True, exist_ok=True)

            plans.append(RunPlan(
                workflow=method.name,
                abacus=abacus,
                orbgen=orbgen,
                abacus_config=abacus.config,
                siab_json_path=siab_path,
                output_dir=plan_dir,
                candidates=candidates,
                preset_name=_preset_label(abacus, orbgen, multiple_abacus),
                code_label=bundle.code_label or abacus.config.get("abacus", {}).get("code"),
                search_strategy=bundle.search_strategy,
            ))
    return plans


def build_workchain_inputs(plan: RunPlan, *, dry_run: bool = False) -> dict:
    """AiiDA inputs for one plan (loads the code → needs a profile)."""
    from aiida import orm
    from aiida.orm import SinglefileData

    abacus_config = orm.Dict(dict=plan.abacus_config)
    siab_json = SinglefileData(file=str(plan.siab_json_path))
    output_dir = orm.Str(str(plan.output_dir))

    inputs: dict[str, Any] = {
        "siab_json": siab_json,
        "abacus_config": abacus_config,
        "output_dir": output_dir,
        "dry_run": orm.Bool(bool(dry_run)),
    }
    if plan.code_label:
        inputs["code_label"] = orm.Str(str(plan.code_label))
    max_iterations = plan.abacus_config.get("abacus", {}).get("max_iterations")
    if max_iterations is not None:
        inputs["max_iterations"] = orm.Int(int(max_iterations))

    if plan.workflow == WORKFLOW_CALC:
        l_max, r_cut = plan.candidates[0]
        inputs["l_max"] = orm.Int(int(l_max))
        inputs["r_cut"] = orm.Float(float(r_cut))
    else:
        inputs["l_max_candidates"] = orm.List(list=sorted({lm for lm, _ in plan.candidates}))
        inputs["r_cut_candidates"] = orm.List(list=sorted({float(rc) for _, rc in plan.candidates}))
        inputs["search_strategy"] = orm.Str(plan.search_strategy)
    return inputs


def submit_plans(plans: list[RunPlan], *, dry_run: bool = False) -> list[SubmittedJob]:
    """Submit every plan and return the records written to ``output.json``."""
    from aiida.engine import submit
    from aiida.plugins import WorkflowFactory

    jobs: list[SubmittedJob] = []
    for plan in plans:
        inputs = build_workchain_inputs(plan, dry_run=dry_run)
        workchain = WorkflowFactory(plan.entry_point)
        node = submit(workchain, **inputs)
        method = get_method_spec(plan.workflow)
        jobs.append(SubmittedJob(
            backend="abacus",
            key=method.key,
            preset_name=plan.preset_name,
            uuid=str(node.uuid),
            pk=node.pk,
            workflow=plan.workflow,
            details={
                "candidates": [[lm, rc] for lm, rc in plan.candidates],
                "output_dir": str(plan.output_dir),
                "siab_json": str(plan.siab_json_path),
                "code": plan.code_label,
            },
        ))
    return jobs


# ---------------------------------------------------------------------------
#  Report dispatch
# ---------------------------------------------------------------------------


def generate_one_report(
    identifier: int | str,
    output_dir: str | Path,
    *,
    profile: str | None = None,
    **kwargs,
):
    """Load a node, verify it is an orbgen WorkChain, and write its report."""
    from aiida import load_profile
    from aiida.orm import load_node

    from aiida_orbgen.utils.report import generate_one_report as _generate

    load_profile(profile)
    node = load_node(identifier)
    label = getattr(node, "process_label", type(node).__name__)
    known = {spec.class_name for spec in METHOD_SPECS.values()}
    if label not in known:
        return None, (
            f"skipped: id={identifier} unsupported WorkChain type {label!r} "
            f"(expected one of {sorted(known)})"
        )
    result = _generate(identifier, output_dir, profile=profile, **kwargs)
    return result, f"ok -> {result.report_path}"
