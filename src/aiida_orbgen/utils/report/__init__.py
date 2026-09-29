"""Report helpers for ``aiida-orbgen report``.

* :mod:`aiida_orbgen.utils.report.orbgen`    — provenance → Markdown
* :mod:`aiida_orbgen.utils.report.orbitals`  — provenance → orbital files
* :func:`generate_one_report`                — the end-to-end pipeline used
  by the CLI (collect → export → optional final orbital → write ``report.md``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from aiida_orbgen.utils.report.orbgen import (
    GridPoint,
    OrbgenRunSummary,
    collect_summary,
    generate_report,
    render_report,
    select_best_point,
)
from aiida_orbgen.utils.report.orbitals import (
    OrbitalFile,
    choose_point_for_dft_root,
    dft_root_from_input_json,
    export_primitive_orbitals,
    generate_final_orbital,
)

__all__ = [
    "GridPoint",
    "OrbgenRunSummary",
    "OrbitalFile",
    "ReportResult",
    "collect_summary",
    "render_report",
    "generate_report",
    "select_best_point",
    "export_primitive_orbitals",
    "generate_final_orbital",
    "choose_point_for_dft_root",
    "dft_root_from_input_json",
    "generate_one_report",
]


@dataclass
class ReportResult:
    """Outcome of :func:`generate_one_report`."""

    node_pk: int
    node_uuid: str
    status: str
    report_path: Path | None = None
    orbital_files: list[OrbitalFile] = field(default_factory=list)
    final_orbitals: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status.startswith("ok")


def generate_one_report(
    node_identifier: int | str,
    output_dir: str | Path,
    *,
    profile: str | None = None,
    report_name: str = "report.md",
    export_orbitals: bool = True,
    include_upf: bool = False,
    run_final_orbital: bool = False,
    dft_root: str | Path | None = None,
    run_missing: str = "monomer",
    orbgen_command: str | None = None,
    final_orbital_timeout: int | None = 3600,
    include_process_logs: bool = True,
    primitive_subdir: str | None = "primitive",
    final_orbital_pk: int | None = None,
    siab_config: str | Path | None = None,
    input_json: str | Path | None = None,
    use_stored_config: bool = False,
    assemble_dft: bool = False,
    redo_final_orbital: bool = False,
    dry_run: bool = False,
) -> ReportResult:
    """Generate ``report.md`` (+ orbital files) for one WorkChain identifier.

    Steps: load the node → collect the summary → export the primitive ``.orb``
    files (into ``<output_dir>/primitive/``, so the final CSW-NAO orbital can
    keep its canonical name at the top level) → optionally run the SIAB
    spillage minimisation for the best ``(l_max, r_cut)`` → render and write
    the Markdown report.

    ``final_orbital_pk`` pins the spillage step to one specific
    ``OrbgenCalcWorkChain`` instead of the automatically chosen best point
    (useful when the reference DFT data only exist for one grid point).
    """
    from aiida import load_profile
    from aiida.orm import load_node

    load_profile(profile)
    node = load_node(node_identifier)
    # Absolute: the spillage step runs ``orbgen`` with the DFT root as cwd.
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_summary(node)

    orbital_files: list[OrbitalFile] = []
    warnings: list[str] = []
    if export_orbitals and summary.grid:
        export_dir = output_dir / primitive_subdir if primitive_subdir else output_dir
        orbital_files, warnings = export_primitive_orbitals(
            summary, export_dir, include_upf=include_upf
        )

    # One spillage run per grid point, so a bare `report -i output.json -o ./`
    # finishes every point it can (assembling missing reference data and
    # computing a missing monomer on the way).
    final_orbitals: list[dict] = []
    if run_final_orbital and summary.grid:
        points = list(summary.grid)
        if final_orbital_pk is not None:
            pinned = collect_summary(load_node(int(final_orbital_pk)))
            if not pinned.grid:
                warnings.append(
                    f"final orbital: node {final_orbital_pk} is not an "
                    f"OrbgenCalcWorkChain — using every grid point instead"
                )
            else:
                points = pinned.grid
        for point in points:
            outcome = generate_final_orbital(
                summary,
                output_dir,
                dft_root=dft_root,
                run_missing=run_missing,
                dry_run=dry_run,
                timeout=final_orbital_timeout,
                orbgen_command=orbgen_command,
                point=point,
                siab_config=siab_config,
                input_json=input_json,
                use_stored_config=use_stored_config,
                assemble=assemble_dft,
                redo=redo_final_orbital,
            )
            final_orbitals.append(outcome)
            if outcome.get("status") in ("failed", "timeout", "empty"):
                warnings.append(
                    f"l_max={point.l_max}, r_cut={point.r_cut}: "
                    + str(outcome.get("message") or outcome.get("status"))
                )

    report_path = output_dir / report_name
    orbital_dirs = sorted(
        path for path in output_dir.glob("lmax*_rcut*") if path.is_dir()
    )
    text = render_report(
        summary,
        orbital_files=orbital_files,
        final_orbitals=final_orbitals,
        orbital_dirs=orbital_dirs,
        include_process_logs=include_process_logs,
    )
    report_path.write_text(text, encoding="utf-8")

    return ReportResult(
        node_pk=node.pk,
        node_uuid=str(node.uuid),
        status=f"ok -> {report_path}",
        report_path=report_path,
        orbital_files=orbital_files,
        final_orbitals=final_orbitals,
        warnings=warnings,
    )
