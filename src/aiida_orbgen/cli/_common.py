"""Shared helpers of the ``aiida-orbgen`` CLI (``run`` / ``report`` / ``check``).

Everything workflow-specific lives in :data:`METHOD_SPECS`, so the CLI code
itself stays thin — mirroring ``aiida_uranium_workflow.cli._common``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiida_orbgen.utils.cal_json import SubmittedJob, default_result_path
from aiida_orbgen.utils.config import (
    SCAN_KEYS,
    ParamBundle,
    PresetEntry,
    SCAN_WORKFLOWS,
    WORKFLOW_BASIS,
    WORKFLOW_CALC,
    WORKFLOW_ECUTWFC,
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
    "scan_workchain_inputs",
    "reuse_pw_reference",
    "describe_scan",
    "submit_plans",
    "generate_one_report",
    "default_result_path",
    "SubmittedJob",
    "WORKFLOW_CALC",
    "WORKFLOW_GRIDSEARCH",
    "WORKFLOW_ECUTWFC",
    "WORKFLOW_BASIS",
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
    # The two value-selection scans: they answer "which parameters", so their outputs
    # are decisions (`ecutwfc_decision` / `basis_decision`) rather than an orbital.
    WORKFLOW_ECUTWFC: MethodSpec(
        name=WORKFLOW_ECUTWFC,
        entry_point="orbgen.ecutwfc",
        class_name="OrbgenEcutwfcWorkChain",
    ),
    WORKFLOW_BASIS: MethodSpec(
        name=WORKFLOW_BASIS,
        entry_point="orbgen.basis",
        class_name="OrbgenBasisScanWorkChain",
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
    #: ``input.json["scan"]``, already validated and canonicalised
    scan: dict = field(default_factory=dict)

    @property
    def entry_point(self) -> str:
        return get_method_spec(self.workflow).entry_point

    def describe_scan(self) -> str:
        """One line describing this scan (same wording as ``check`` prints)."""
        return describe_scan(self.workflow, self.scan, self.candidates)

    def describe(self) -> str:
        if self.workflow in SCAN_WORKFLOWS:
            return self.describe_scan()
        if self.workflow == WORKFLOW_CALC:
            l_max, r_cut = self.candidates[0]
            return f"l_max={l_max}, r_cut={r_cut:g}"
        return (
            f"{len(self.candidates)} candidate(s): "
            + ", ".join(f"(l_max={lm}, r_cut={rc:g})" for lm, rc in self.candidates)
        )


def describe_scan(workflow: str, scan: dict, candidates: list[tuple[int, float]]) -> str:
    """One line describing a scan, for the plan and for ``output.json``.

    Kept out of :class:`RunPlan` so ``check`` can print the same wording without a
    profile, and so the two scans cannot describe themselves differently.
    """
    l_max, r_cut = candidates[0]
    reference = f"reference l_max={l_max}, r_cut={r_cut:g}"
    if "reference_ecutjy" in scan:
        reference += f", ecutjy={float(scan['reference_ecutjy']):g}"
    if workflow == WORKFLOW_ECUTWFC:
        if "ecutwfc_values" in scan:
            ladder = ", ".join(f"{float(v):g}" for v in scan["ecutwfc_values"])
            ladder_txt = f"ecutwfc ladder [{ladder}] Ry"
        else:
            scale = scan.get("ecutwfc_scale") or [1.0]
            ladder_txt = (f"ecutwfc ladder = baseline x "
                          f"{', '.join(f'{float(v):g}' for v in scale)}")
        if scan.get("with_lcao"):
            ladder_txt += " + one LCAO child per geometry"
        return f"{ladder_txt} ({reference})"
    steps = []
    for key, label in (("ecutjy_values", "ecutjy"), ("l_max_values", "l_max"),
                       ("r_cut_values", "r_cut")):
        values = scan.get(key)
        if values:
            steps.append(f"{label} {'/'.join(f'{float(v):g}' for v in values)}")
    ladder_txt = ("ladder " + " -> ".join(steps)) if steps else "no reduction (reference only)"
    return (f"{ladder_txt} from {reference} "
            f"[strategy={scan.get('strategy', 'ladder')}"
            + (f", atomization gate {float(scan['atomization_tolerance_meV']):g} meV"
               if "atomization_tolerance_meV" in scan else "")
            + (f", reusing the PW reference of PK {scan['pw_reference_pk']}"
               if "pw_reference_pk" in scan else "")
            + "]")


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
            if method.name in SCAN_WORKFLOWS and len(candidates) > 1:
                # A scan *reduces* from one reference point, so which of the preset's
                # candidates is that point cannot be guessed: an orbgen preset with
                # several r_cut values is a grid, not a ladder.
                raise ValueError(
                    f"orbgen preset '{orbgen.name}' produced {len(candidates)} "
                    f"(l_max, r_cut) candidates {candidates}, but {method.name} reduces "
                    f"from exactly one reference point — give the preset a single "
                    f"bessel_nao_rcut/lmaxmax (or use `aiida-orbgen select`)"
                )
            if not candidates:
                raise ValueError(
                    f"orbgen preset '{orbgen.name}' produced no (l_max, r_cut) "
                    f"candidate (check bessel_nao_rcut / lmaxmax / max_r_cut)"
                )

            siab_path = output_root / "configs" / f"orbgen_{orbgen.name}.json"
            siab_path.write_text(
                json.dumps(orbgen.config, indent=4) + "\n", encoding="utf-8"
            )

            if method.name in SCAN_WORKFLOWS:
                # <output_dir>/<preset>/scan_<family>: the SIAB step writes one tree
                # per candidate *inside* it (the labels of the ladder, r11_l4_j125…).
                family = "ecutwfc" if method.name == WORKFLOW_ECUTWFC else "basis"
                plan_dir = output_root / orbgen.name / f"scan_{family}"
            elif method.name == WORKFLOW_GRIDSEARCH:
                plan_dir = output_root / orbgen.name
            else:
                l_max, r_cut = candidates[0]
                # The point directory name has one implementation
                # (interfaces.nsw.point_dir_name -> ``lmax4_rcut10``); this used to
                # spell it ``lmax4_rcut10p0`` here while the workflow and the report
                # used the current spelling, so the announced run root and the tree
                # the run produced did not match.
                from aiida_orbgen.interfaces.nsw import point_dir_name

                plan_dir = output_root / orbgen.name / point_dir_name(l_max, r_cut)
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
                scan=dict(bundle.scan or {}),
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

    if plan.workflow in (WORKFLOW_CALC, WORKFLOW_ECUTWFC, WORKFLOW_BASIS):
        l_max, r_cut = plan.candidates[0]
        inputs["l_max"] = orm.Int(int(l_max))
        inputs["r_cut"] = orm.Float(float(r_cut))
    else:
        inputs["l_max_candidates"] = orm.List(list=sorted({lm for lm, _ in plan.candidates}))
        inputs["r_cut_candidates"] = orm.List(list=sorted({float(rc) for _, rc in plan.candidates}))
        inputs["search_strategy"] = orm.Str(plan.search_strategy)

    if plan.workflow in SCAN_WORKFLOWS:
        inputs.update(scan_workchain_inputs(plan))
    return inputs


def scan_workchain_inputs(plan: RunPlan) -> dict[str, Any]:
    """The ``input.json["scan"]`` parameters, as AiiDA inputs.

    Every key is passed through unchanged — that is the point of the section: the
    workflow knows *how* to run the ladder, ``input.json`` says *which* ladder.  Only
    one key is resolved here: ``pw_reference_pk`` becomes the ``pw_reference`` Dict of
    that scan, so reusing a computed PW reference runs no PW child at all.
    """
    from aiida import orm

    scan = dict(plan.scan or {})
    if "pw_reference" in scan and "pw_reference_pk" in scan:
        raise ValueError(
            "scan.pw_reference and scan.pw_reference_pk are mutually exclusive: give "
            "the energies directly, or the scan to take them from"
        )
    pk = scan.pop("pw_reference_pk", None)
    inputs: dict[str, Any] = {}
    for key, value in scan.items():
        # The node class follows the *schema*, not the Python type of the JSON value:
        # `"atomization_tolerance_meV": 50` is a JSON integer but the WorkChain declares
        # a Float, and dispatching on `isinstance(value, int)` submitted an orm.Int —
        # which AiiDA then rejects at submission time with a bare port error.
        expected = SCAN_KEYS[key][1]
        if expected is bool:
            inputs[key] = orm.Bool(bool(value))
        elif expected is list:
            inputs[key] = orm.List(list=value)
        elif expected is str:
            inputs[key] = orm.Str(str(value))
        elif expected is dict:
            inputs[key] = orm.Dict(dict=value)
        elif expected == (int, float):
            inputs[key] = orm.Float(float(value))
        else:  # pragma: no cover - canonical_scan_config rejects these already
            raise TypeError(f"scan.{key}: unsupported value {value!r}")
    if pk is not None:
        inputs["pw_reference"] = reuse_pw_reference(int(pk))
    return inputs


def reuse_pw_reference(pk: int):
    """The ``pw_reference`` Dict of an earlier ``orbgen.ecutwfc`` run.

    ``orbgen.ecutwfc`` records ``{"ecutwfc": …, "geometries": {geometry: {energy,
    n_atoms}}}`` at its converged cutoff for exactly this purpose: with it the basis scan runs **no** PW child, which
    is the expensive half of a basis comparison.  It is resolved here rather than in the
    offline planner because it needs a profile, and a wrong PK fails loudly instead of
    submitting a scan that silently recomputes PW.
    """
    from aiida import orm

    try:
        node = orm.load_node(int(pk))
    except Exception as exc:  # noqa: BLE001 - a bad PK is a user error
        raise ValueError(f"scan.pw_reference_pk={pk}: cannot load that node: {exc}")
    outputs = getattr(node, "outputs", None)
    decision = None
    if outputs is not None and "ecutwfc_decision" in outputs:
        decision = outputs.ecutwfc_decision.get_dict()
    if not decision:
        raise ValueError(
            f"scan.pw_reference_pk={pk}: that node is not an `orbgen.ecutwfc` run (no "
            f"ecutwfc_decision output); run the cutoff scan first, or give "
            f"scan.pw_reference directly"
        )
    reference = decision.get("pw_reference")
    if not reference or "geometries" not in reference:
        raise ValueError(
            f"scan.pw_reference_pk={pk}: its decision carries no pw_reference block "
            f"(an older checkout wrote a different shape) — give scan.pw_reference "
            f"directly"
        )
    return orm.Dict(dict=reference)


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
