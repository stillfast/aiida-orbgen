#!/usr/bin/env python3
"""``aiida-orbgen`` command line entry point.

Sub-commands
------------

``run``     — submit the orbgen WorkChain(s) described by an ``input.json``
              and write an ``output.json`` identifier map next to it.
``report``  — read that ``output.json`` and write ``report.md`` **plus the
              orbital files** (primitive NSW ``.orb``, and — unless disabled —
              the final CSW-NAO orbital produced by SIAB's spillage
              minimisation) into ``-o DIR``.
``check``   — validate an ``input.json`` offline and print the execution plan.
``select``  — record which grid point a report chose, as a reusable bundle.
``fetch-dft`` — download/rebuild the reference DFT tree from AiiDA provenance.

The legacy ``submit-siab {calc,gridsearch}`` sub-command was removed on
2026-09-29 (see ``_removed-20260929/README.md``): it built workchain inputs of
its own, skipping ``apply_input_overrides`` and every ``validate_*`` check.

Parameter presets live in ``src/aiida_orbgen/parameters/``::

    parameters/
    ├── metadata.yml          # scheduler-options presets (static.metadata)
    ├── abacus/
    │   ├── abacus.yml        # default preset file of the "abacus" slot
    │   └── test.yml          # parameters.abacus = {"test": "test"}
    └── orbgen/
        ├── orbgen.yml        # default preset file of the "orbgen" slot
        └── test.yml          # parameters.orbgen = {"test": "test"}

Typical usage::

    # 1. offline validation: resolves presets, candidates, codes, options
    aiida-orbgen check  -i input.json

    # 2. submit, write output.json next to input.json
    aiida-orbgen run    -i input.json

    # 3. report.md + orbital files in the current directory
    aiida-orbgen report -i output.json -o ./

See ``README.md`` for the full ``input.json`` and preset reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aiida_orbgen.cli._common import (
    METHOD_SPECS,
    generate_one_report,
    plan_runs,
    resolve_method,
    submit_plans,
)
from aiida_orbgen.utils.cal_json import (
    collect_job_entries,
    default_result_path,
    read_output_json,
    write_cal_json,
)
from aiida_orbgen.utils.config import SCAN_WORKFLOWS, ConfigLoader

__all__ = [
    "main",
    "build_parser",
    "cmd_run",
    "cmd_report",
    "cmd_fetch_dft",
    "cmd_select",
    "cmd_check",
]


# ---------------------------------------------------------------------------
#  helpers
# ---------------------------------------------------------------------------


def _print_plan(bundle, plans) -> None:
    """Human-readable summary of what ``run`` is about to submit."""
    print(
        f"[plan] workflow={bundle.workflow}"
        f"{'' if bundle.workflow_explicit else ' (auto-detected)'} "
        f"| profile={bundle.profile or '<default>'} "
        f"| code={bundle.code_label}"
    )
    for warning in getattr(bundle, "warnings", []) or []:
        print(f"[plan] WARNING: {warning}")
    print(
        f"[plan] metadata preset={bundle.metadata_name or '<none>'} "
        f"| pseudo_path={bundle.pseudo_path or '<from preset>'}"
    )
    static = bundle.static or {}
    if static.get("output_dir"):
        print(f"[plan] output_dir (from input.json) = {static['output_dir']}")
    else:
        print(
            "[plan] SIAB run root = <input_dir>/run "
            "(override with static.output_dir in input.json or --output-root)"
        )
    options = (bundle.metadata or {}).get("options") or {}
    if options:
        print(
            "[plan] scheduler options: "
            + ", ".join(f"{key}={value}" for key, value in options.items())
        )
    print(f"[plan] {len(plans)} WorkChain(s) to submit:")
    for index, plan in enumerate(plans, start=1):
        print(
            f"  {index}. {plan.workflow} preset={plan.preset_name} "
            f"({plan.entry_point}) — {plan.describe()}"
        )
        print(f"       abacus params : {plan.abacus.source}")
        print(f"       orbgen preset : {plan.orbgen.source}")
        print(f"       siab_json     : {plan.siab_json_path}")
        print(f"       output_dir    : {plan.output_dir}")
        if plan.workflow == METHOD_SPECS["orbgen.gridsearch"].name:
            print(f"       search_strategy: {plan.search_strategy}")
        if plan.workflow in SCAN_WORKFLOWS:
            print(f"       scan          : {plan.describe_scan()}")
        tolerance = plan.abacus_config.get("tolerance_meV")
        if tolerance is not None:
            print(f"       tolerance_meV : {tolerance}")


# ---------------------------------------------------------------------------
#  sub-command: run
# ---------------------------------------------------------------------------


def cmd_run(args) -> int:
    input_json = Path(args.input_json).resolve()
    if not input_json.is_file():
        print(f"Error: input.json not found: {input_json}", file=sys.stderr)
        return 1

    try:
        bundle = ConfigLoader(input_json).load_all()
    except Exception as exc:  # noqa: BLE001 — surface every config error
        print(f"Error loading {input_json}: {exc}", file=sys.stderr)
        return 1

    if args.profile:
        bundle.profile = args.profile
    if args.search_strategy:
        bundle.search_strategy = args.search_strategy

    workflow = args.workflow or bundle.workflow
    try:
        plans = plan_runs(
            bundle,
            output_root=args.output_root,
            workflow=workflow,
            only=args.only,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error building the submission plan: {exc}", file=sys.stderr)
        return 1

    _print_plan(bundle, plans)

    output_path = (
        Path(args.output).resolve() if args.output else default_result_path(input_json)
    )

    if args.dry_run:
        print("[run] --dry-run: nothing submitted; no output.json written.")
        print(f"[run] resolved SIAB config(s): {plans[0].siab_json_path}")
        return 0

    if not bundle.profile:
        print(
            "Error: no AiiDA profile — set input.json['profile'] or pass -p/--profile",
            file=sys.stderr,
        )
        return 1

    try:
        from aiida import load_profile

        load_profile(bundle.profile)
        jobs = submit_plans(plans)
    except Exception as exc:  # noqa: BLE001
        print(f"Error submitting: {exc}", file=sys.stderr)
        return 1

    for job in jobs:
        print(
            f"[run] submitted {job.workflow} preset={job.preset_name} "
            f"PK={job.pk} UUID={job.uuid}"
        )
    write_cal_json(
        jobs,
        output_path=output_path,
        workflow=workflow,
        input_json=input_json,
    )
    print(f"[run] wrote {output_path}")
    print("[run] check with: verdi process list")
    return 0


# ---------------------------------------------------------------------------
#  sub-command: report
# ---------------------------------------------------------------------------


def cmd_report(args) -> int:
    output_json = Path(args.input_json).resolve()
    try:
        data = read_output_json(output_json)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if data.get("workflow") in SCAN_WORKFLOWS:
        # `report` builds the final CSW-NAO orbital from a reference DFT tree; a scan
        # has no tree of its own -- it *chooses* the parameters that run will use.
        print(
            f"[report] {output_json} holds a {data['workflow']} run: it produced a "
            f"decision, not an orbital flat. Read it with `verdi process report <PK>` "
            f"or `python tools/read_scan.py <PK>`, then feed the chosen values back "
            f"into the input.json of an `orbgen.calc` run.",
            file=sys.stderr,
        )
        return 1

    entries = collect_job_entries(data)
    if not entries:
        print(f"Error: no WorkChain identifier found in {output_json}", file=sys.stderr)
        return 1

    try:
        method = resolve_method(
            cli_method=args.workflow,
            output_json=output_json,
            input_json=data.get("input_json"),
        ) or str(data.get("workflow") or "")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Resolve everything against the invoking shell's cwd: the spillage step
    # runs ``orbgen`` with the DFT root as its cwd, so a relative ``-o ./``
    # would otherwise be interpreted there instead of here.
    out_dir = (
        Path(args.output_dir).expanduser() if args.output_dir else Path.cwd()
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dft_root = Path(args.dft_root).expanduser().resolve() if args.dft_root else None
    siab_json = Path(args.siab_json).expanduser().resolve() if args.siab_json else None
    if siab_json is not None and not siab_json.is_file():
        print(f"Error: --siab-json not found: {siab_json}", file=sys.stderr)
        return 1
    if dft_root is not None and not dft_root.is_dir():
        print(f"Error: --dft-root not a directory: {dft_root}", file=sys.stderr)
        return 1

    # ``input.json`` (recorded in output.json) is what lets ``report`` re-read
    # the presets and find ``static.dft_root``; both are optional.
    input_json = data.get("input_json")
    if args.input_config:
        input_json = str(Path(args.input_config).expanduser().resolve())

    try:
        from aiida import load_profile

        load_profile(args.profile)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: cannot load an AiiDA profile: {exc}", file=sys.stderr)
        return 1

    print(
        f"[report] {len(entries)} node(s) from {output_json} "
        f"| workflow={method or '<unknown>'} | out_dir={out_dir}"
    )

    # ``report.md`` for a single node; one directory per node otherwise (so two
    # nodes never share ``primitive/`` or an ``lmax*_rcut*`` orbital directory).
    single = len(entries) == 1
    failures = 0
    for label, identifier in entries:
        node_dir = out_dir if single else out_dir / str(identifier)[:8]
        report_name = "report.md" if single else f"report_{identifier[:8]}.md"
        result, status = generate_one_report(
            identifier,
            node_dir,
            profile=args.profile,
            report_name=report_name,
            export_orbitals=not args.no_orbitals,
            include_upf=args.with_upf,
            run_final_orbital=not args.no_final_orbital,
            dft_root=dft_root,
            run_missing="all" if args.force_final_orbital else "monomer",
            orbgen_command=args.orbgen_command,
            final_orbital_timeout=args.final_orbital_timeout,
            include_process_logs=not args.no_process_logs,
            final_orbital_pk=args.calc_pk,
            siab_config=siab_json,
            input_json=input_json,
            use_stored_config=args.use_stored_config,
            assemble_dft=args.assemble_dft,
            redo_final_orbital=args.redo_final_orbital,
            dry_run=args.dry_run,
        )
        if result is None:
            print(f"  {label}: {status}", file=sys.stderr)
            failures += 1
            continue
        print(f"  {label} [{result.node_uuid[:8]}]: {status}")
        if result.orbital_files:
            print(f"      primitive : {len(result.orbital_files)} .orb -> "
                  f"{result.orbital_files[0].path.parent}")
        bad_orbitals = [
            entry for entry in (result.final_orbitals or [])
            if entry.get("status") in ("failed", "timeout", "empty", "invalid")
            or entry.get("validation_failed")
        ]
        for entry in bad_orbitals:
            print(f"      ⚠ final orbital (l_max={entry.get('l_max')}, "
                  f"r_cut={entry.get('r_cut')}): {entry.get('status')}"
                  + (f" — {entry.get('validation_failed')}" if entry.get("validation_failed") else ""),
                  file=sys.stderr)
        for warning in result.warnings:
            print(f"      warning   : {warning}", file=sys.stderr)
        for final in result.final_orbitals:
            r_cut = final.get("r_cut")
            r_cut = (int(r_cut) if isinstance(r_cut, (int, float))
                     and float(r_cut).is_integer() else r_cut)
            print(f"      final orb : l_max={final.get('l_max')}, "
                  f"r_cut={r_cut} — {final.get('status')}"
                  + (f" ({final.get('message')})" if final.get("message") else ""))
            if final.get("dir"):
                print(f"      directory : {final['dir']}")
            label = ("existing  " if final.get("status") == "already-present"
                     else "produced  ")
            for path in final.get("files") or []:
                print(f"      {label}: {Path(path).name}")

    print(f"[report] {'ok' if not failures else f'{failures} node(s) failed'}")
    return 0 if not failures else 1


# ---------------------------------------------------------------------------
#  sub-command: fetch-dft
# ---------------------------------------------------------------------------


def cmd_fetch_dft(args) -> int:
    """Prepare a spillage reference tree from the AiiDA remote folders."""
    output_json = Path(args.input_json).resolve()
    try:
        data = read_output_json(output_json)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    entries = collect_job_entries(data)
    if not entries:
        print(f"Error: no WorkChain identifier found in {output_json}", file=sys.stderr)
        return 1

    from aiida import load_profile
    from aiida.orm import load_node

    from aiida_orbgen.utils.report.assemble import assemble_reference
    from aiida_orbgen.utils.report.orbgen import collect_summary

    load_profile(args.profile)

    input_json = data.get("input_json")
    if args.input_config:
        input_json = str(Path(args.input_config).expanduser().resolve())
    root = Path(args.dft_root).expanduser().resolve() if args.dft_root else None
    if root is None:
        from aiida_orbgen.utils.report.orbitals import dft_root_from_input_json

        root = dft_root_from_input_json(input_json)
    if root is None:
        print("Error: no DFT root — pass --dft-root or set static.dft_root",
              file=sys.stderr)
        return 1

    failures = 0
    for label, identifier in entries:
        node = load_node(identifier)
        summary = collect_summary(node)
        points = list(summary.grid)
        if args.calc_pk:
            points = [p for p in points if p.pk == args.calc_pk]
            if not points:
                print(f"Error: {args.calc_pk} is not a grid point of {label}",
                      file=sys.stderr)
                return 1
        for point in points:
            calc_node = load_node(point.pk)
            print(f"[fetch-dft] {label} l_max={point.l_max} r_cut={point.r_cut:g} "
                  f"-> {root}")
            result = assemble_reference(calc_node, root)
            print(f"            {result.get('summary') or result.get('message')}")
            for folder, entry in sorted((result.get("folders") or {}).items()):
                detail = entry.get("status", "?")
                if entry.get("dev_eV") is not None:
                    detail += (f" bands={entry.get('bands')}"
                               f" ΔE={entry['dev_eV']:.1e} eV")
                if entry.get("message"):
                    detail += f" — {entry['message']}"
                print(f"              {folder}: {detail}")
            if not result.get("ok", True):
                failures += 1

    if failures:
        print(f"[fetch-dft] {failures} grid point(s) incomplete")
        return 1
    print("[fetch-dft] reference tree ready (missing geometries, e.g. the "
          "monomer, are computed by SIAB with --force-final-orbital)")
    return 0


# ---------------------------------------------------------------------------
#  sub-command: check
# ---------------------------------------------------------------------------


def cmd_select(args) -> int:
    """Record which grid point a report chose (and make it reusable)."""
    from aiida import load_profile
    from aiida.orm import load_node

    from aiida_orbgen.utils.cal_json import collect_job_entries, read_output_json
    from aiida_orbgen.utils.report.orbgen import collect_summary
    from aiida_orbgen.utils.select import (
        choose_point,
        selection_payload,
        write_selection,
    )

    load_profile(args.profile)
    data = read_output_json(args.input_json)
    entries = collect_job_entries(data)
    if not entries:
        print(f"No WorkChain identifier in {args.input_json}", file=sys.stderr)
        return 1

    written_any = False
    for label, identifier in entries:
        node = load_node(identifier)
        summary = collect_summary(node)
        try:
            point = choose_point(
                summary,
                l_max=args.l_max,
                r_cut=args.r_cut,
                calc_pk=args.calc_pk,
            )
        except ValueError as exc:
            print(f"  {label}: {exc}", file=sys.stderr)
            return 1
        if point is None:
            print(f"  {label}: no grid point matched the request", file=sys.stderr)
            continue

        chosen_by = "auto"
        if args.calc_pk is not None:
            chosen_by = f"--calc-pk {args.calc_pk}"
        elif args.l_max is not None:
            chosen_by = f"--l-max {args.l_max} --r-cut {args.r_cut}"

        # SIAB config of the chosen point: explicit file > the node's stored copy,
        # with the grid point applied (same code path as `report`).
        siab_config = None
        try:
            if args.siab_json is not None:
                with open(args.siab_json, "r", encoding="utf-8") as handle:
                    from aiida_orbgen.interfaces.nsw import apply_grid_point

                    siab_config = apply_grid_point(
                        json.load(handle), point.l_max, point.r_cut
                    )
            else:
                from aiida_orbgen.utils.report.orbitals import _build_siab_config

                siab_config = _build_siab_config(
                    load_node(point.pk), point.l_max, point.r_cut
                )
        except Exception as exc:  # noqa: BLE001 — the bundle is still useful
            print(f"  {label}: cannot materialise the SIAB config: {exc}",
                  file=sys.stderr)

        if siab_config is not None:
            # Validating here is the point of the exercise: a point whose scheme
            # the primitive basis cannot provide would otherwise be discovered
            # inside SIAB, after another reference DFT.
            from aiida_orbgen.utils.config import validate_siab_config

            try:
                for warning in validate_siab_config(
                    siab_config, source="selected config"
                ):
                    print(f"      warning: {warning}", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001
                print(f"  {label}: selected config is invalid: {exc}", file=sys.stderr)
                return 1

        base_input = None
        if args.input_config is not None and Path(args.input_config).is_file():
            base_input = json.loads(Path(args.input_config).read_text())
        payload = selection_payload(
            summary, point, chosen_by=chosen_by,
            siab_config=siab_config, base_input=base_input,
        )
        name = f"lmax{point.l_max}_rcut{point.r_cut:g}"
        written = write_selection(
            args.output_dir, payload, siab_config=siab_config, point_name=name,
        )
        written_any = True

        print(f"  {label} [{summary.node_uuid[:8]}]: {payload['description']}")
        for kind, path in written.items():
            print(f"      {kind:11s}: {path}")
        if "input_json" in written:
            # input.selected.json carries the point's config inline
            # (static.siab_config), so it submits exactly this point — no preset edit
            print("      run        : aiida-orbgen check -i "
                  f"{written['input_json']}  &&  aiida-orbgen run -i "
                  f"{written['input_json']}")
        if siab_config is not None:
            print("      next       : aiida-orbgen report -i output.json "
                  f"--siab-json {written['siab_config']} --calc-pk {point.pk}")
    return 0 if written_any else 1


def cmd_check(args) -> int:
    """Offline validation of an ``input.json`` (no AiiDA profile needed)."""
    input_json = Path(args.input_json).resolve()
    if not input_json.is_file():
        print(f"Error: input.json not found: {input_json}", file=sys.stderr)
        return 1
    try:
        bundle = ConfigLoader(input_json).load_all()
        plans = plan_runs(bundle, output_root=args.output_root, workflow=args.workflow)
    except Exception as exc:  # noqa: BLE001
        print(f"[check] invalid input: {exc}", file=sys.stderr)
        return 1
    _print_plan(bundle, plans)
    print("[check] configuration OK — nothing submitted.")
    return 0


# ---------------------------------------------------------------------------
#  parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aiida-orbgen",
        description=(
            "AiiDA plugin for ABACUS CSW-NAO orbital generation.\n"
            "Sub-commands: run, report, select, fetch-dft, check."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="subcmd", required=True)

    # ── run ──────────────────────────────────────────────────────────────
    p_run = sub.add_parser(
        "run",
        help="Submit the orbgen WorkChain(s) described by an input.json.",
        description=(
            "Read input.json + the parameters/ presets, submit one WorkChain per "
            "(abacus preset × orbgen preset), and write output.json next to the input "
            "file.  Which WorkChain is decided by input.json['workflow'], else by the "
            "candidate grid (one point -> OrbgenCalcWorkChain, several -> "
            "OrbgenGridSearchWorkChain), else by the 'scan' section "
            "(ecutwfc_values -> OrbgenEcutwfcWorkChain; one ladder of "
            "ecutjy/l_max/r_cut -> OrbgenBasisScanWorkChain).  Every parameter of "
            "'scan' comes from input.json; the plugin only holds the code."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_run.add_argument("-i", "--input", dest="input_json", required=True,
                       help="Path to input.json.")
    p_run.add_argument("-p", "--profile", default=None,
                       help="AiiDA profile (overrides input.json['profile']).")
    p_run.add_argument("-o", "--output", dest="output", default=None,
                       metavar="PATH",
                       help="Where to write output.json (default: <input_dir>/output.json).")
    p_run.add_argument("--output-root", dest="output_root", default=None, type=Path,
                       help="SIAB run root (default: <input_dir>/run). "
                            "Overrides static.output_dir.")
    p_run.add_argument("--workflow", choices=sorted(METHOD_SPECS), default=None,
                       help="Force a workflow; default: from input.json or auto-detected.")
    p_run.add_argument("--search-strategy", dest="search_strategy", default=None,
                       choices=("iterative", "exhaustive"),
                       help="Grid-search strategy (default: exhaustive).")
    p_run.add_argument("--only", type=int, default=None,
                       help="With orbgen.calc: use candidate #N (0-indexed) instead of the first.")
    p_run.add_argument("--dry-run", action="store_true",
                       help="Resolve the configuration and print the plan; submit nothing.")
    p_run.set_defaults(func=cmd_run)

    # ── report ───────────────────────────────────────────────────────────
    p_report = sub.add_parser(
        "report",
        help="Write report.md + orbital files for the runs in an output.json.",
        description=(
            "Load every WorkChain identifier of output.json and, for each, "
            "write report.md (status, ΔE matrices, per-dimer energies, basis "
            "inventory) plus the orbital files into --output-dir: the primitive "
            "NSW .orb of every grid point, and — unless --no-final-orbital — the "
            "final CSW-NAO orbital generated by SIAB's spillage minimisation for "
            "the best point."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_report.add_argument("-i", "--input", dest="input_json", required=True,
                          help="Path to the output.json written by `aiida-orbgen run`.")
    p_report.add_argument("-o", "--output-dir", dest="output_dir", default=".", type=Path,
                          help="Directory for report.md and the orbital files (default: .).")
    p_report.add_argument("-p", "--profile", default=None, help="AiiDA profile.")
    p_report.add_argument("--workflow", choices=sorted(METHOD_SPECS), default=None,
                          help="Force a workflow (default: output.json['workflow']).")
    p_report.add_argument("--no-orbitals", action="store_true",
                          help="Do not export the primitive .orb files.")
    p_report.add_argument("--with-upf", action="store_true",
                          help="Also export the UPF next to each .orb.")
    p_report.add_argument("--no-final-orbital", action="store_true",
                          help="Skip the SIAB spillage minimisation (final CSW-NAO orbital).")
    p_report.add_argument("--dft-root", dest="dft_root", default=None, type=Path,
                          help="Reference DFT tree. Default: "
                               "static.dft_roots[<point>] -> static.dft_root -> the "
                               "point's output_dir; a directory of per-point runs is "
                               "searched one level down.")
    p_report.add_argument("--redo-final-orbital", dest="redo_final_orbital",
                          action="store_true",
                          help="Recompute the spillage even when the point's "
                               "directory already holds up-to-date orbital files.")
    p_report.add_argument("--assemble-dft", dest="assemble_dft", action="store_true",
                          default=True,
                          help="Download missing reference DFT data from the AiiDA "
                               "remote folders (and rebuild LCAO wavefunctions if "
                               "needed) before the spillage step. Default: on.")
    p_report.add_argument("--no-assemble-dft", dest="assemble_dft",
                          action="store_false",
                          help="Never touch the cluster: use only local data.")
    p_report.add_argument("--force-final-orbital", action="store_true",
                          help="Allow SIAB to compute *any* missing reference DFT, "
                               "not just the monomer (which is computed "
                               "automatically when it is the only gap).")
    p_report.add_argument("--orbgen-command", dest="orbgen_command", default=None,
                          help="Path/name of the SIAB `orbgen` executable (default: orbgen).")
    p_report.add_argument("--final-orbital-timeout", dest="final_orbital_timeout",
                          type=int, default=3600,
                          help="Timeout in seconds for the spillage step (default: 3600).")
    p_report.add_argument("--calc-pk", dest="calc_pk", type=int, default=None,
                          help="Only this OrbgenCalcWorkChain (default: every grid "
                               "point of the run gets its own lmax*_rcut*/ directory).")
    p_report.add_argument("--siab-json", dest="siab_json", default=None, type=Path,
                          help="Use this SIAB/orbgen JSON for the spillage step "
                               "instead of the config derived from the node / the "
                               "current presets (bessel_nao_rcut and "
                               "geoms[0].lmaxmax are still overridden per point).")
    p_report.add_argument("--input-config", dest="input_config", default=None,
                          type=Path,
                          help="input.json to re-read presets and static.dft_root "
                               "from (default: the path recorded in output.json).")
    p_report.add_argument("--use-stored-config", dest="use_stored_config",
                          action="store_true",
                          help="Never re-read the parameters/ presets: use the "
                               "siab_json stored on the node as-is, even when it "
                               "is stale or invalid.")
    p_report.add_argument("--no-process-logs", action="store_true",
                          help="Omit the `verdi process report` dumps from report.md.")
    p_report.add_argument("--dry-run", action="store_true",
                          help="Prepare everything but do not execute the spillage step.")
    p_report.set_defaults(func=cmd_report)

    # ── fetch-dft ────────────────────────────────────────────────────────
    p_fetch = sub.add_parser(
        "fetch-dft",
        help="Assemble a spillage reference tree from the AiiDA remote folders.",
        description=(
            "For every LCAO child of the selected grid point, download "
            "OUT.<suffix>/{data-0-H/S/T, istate.info, running_scf.log, kpoints, "
            "INPUT} plus the workdir INPUT/STRU and rebuild "
            "WFC_NAO_GAMMA1.txt from the generalised eigenproblem H C = S C eps "
            "(validated against istate.info). No DFT is submitted: geometries "
            "AiiDA never ran (the monomer) are left for SIAB."
        ),
    )
    p_fetch.add_argument("-i", "--input", dest="input_json", required=True,
                         help="Path to the output.json written by `aiida-orbgen run`.")
    p_fetch.add_argument("-p", "--profile", default=None, help="AiiDA profile.")
    p_fetch.add_argument("--dft-root", dest="dft_root", default=None, type=Path,
                         help="Target reference tree (default: static.dft_root "
                              "from the input.json recorded in output.json).")
    p_fetch.add_argument("--input-config", dest="input_config", default=None,
                         type=Path, help="input.json to read static.dft_root from.")
    p_fetch.add_argument("--calc-pk", dest="calc_pk", type=int, default=None,
                         help="Only that grid point (default: all of them).")
    p_fetch.set_defaults(func=cmd_fetch_dft)

    # ── check ────────────────────────────────────────────────────────────
    p_check = sub.add_parser(
        "check",
        help="Validate an input.json offline and print the execution plan.",
        description=(
            "Parse input.json against the parameters/ preset tree, resolve the "
            "workflow, the (l_max, r_cut) candidates, the scheduler options and "
            "the codes, and print exactly what `run` would submit. Creates the "
            "materialised SIAB config(s) under the run root; never submits."
        ),
    )
    p_check.add_argument("-i", "--input", dest="input_json", required=True,
                         help="Path to input.json.")
    p_check.add_argument("--output-root", dest="output_root", default=None, type=Path,
                         help="SIAB run root (default: <input_dir>/run).")
    p_check.add_argument("--workflow", choices=sorted(METHOD_SPECS), default=None,
                         help="Override the workflow.")
    p_check.set_defaults(func=cmd_check)

    # ── select ───────────────────────────────────────────────────────────
    p_select = sub.add_parser(
        "select",
        help="Record which grid point a report chose (selected.json + config).",
        description=(
            "Choose one grid point of a finished grid search — the cheapest "
            "acceptable one by default, or an explicit --l-max/--r-cut / "
            "--calc-pk — and write a small bundle: selected.json (the decision), "
            "orbgen_<point>.json (its validated SIAB config, usable with "
            "`report --siab-json`) and input.selected.json (an input.json that "
            "carries the point's config inline as static.siab_config, so `run` "
            "submits exactly this one orbgen.calc). Parameters/ presets are never "
            "rewritten."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_select.add_argument("-i", "--input", dest="input_json", required=True,
                          help="Path to the output.json written by `aiida-orbgen run`.")
    p_select.add_argument("-o", "--output-dir", dest="output_dir", default=".", type=Path,
                          help="Where to write the selection bundle (default: .).")
    p_select.add_argument("-p", "--profile", default=None, help="AiiDA profile.")
    p_select.add_argument("--l-max", dest="l_max", type=int, default=None,
                          help="Chosen highest angular momentum (with --r-cut).")
    p_select.add_argument("--r-cut", dest="r_cut", type=float, default=None,
                          help="Chosen cutoff radius in a.u. (with --l-max).")
    p_select.add_argument("--calc-pk", dest="calc_pk", type=int, default=None,
                          help="Chosen OrbgenCalcWorkChain instead of the automatic pick.")
    p_select.add_argument("--siab-json", dest="siab_json", default=None, type=Path,
                          help="SIAB config to record instead of the one stored on the node.")
    p_select.add_argument("--input-config", dest="input_config", default=None, type=Path,
                          help="input.json of the finished run; its copy "
                               "(input.selected.json) re-runs the chosen point as-is.")
    p_select.set_defaults(func=cmd_select)

    # The legacy ``submit-siab`` sub-command (``cli/submit_siab.py``) was
    # removed on 2026-09-29: it assembled its own workchain inputs without
    # ``apply_input_overrides`` (leaking AiiDA-managed keys into the ABACUS
    # INPUT and omitting ``out_wfc_lcao``), bypassed every ``validate_*`` check,
    # and duplicated what ``run`` already does.  Use ``run`` / ``report`` /
    # ``check``.

    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
