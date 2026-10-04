"""Pure arithmetic behind the two value-selection workflows.

Nothing here touches AiiDA: the workchains in :mod:`aiida_orbgen.workflows.ecutwfc`
and :mod:`aiida_orbgen.workflows.basis` collect the numbers and call these functions,
which keeps the decision rules testable without a database or a scheduler.

The two decisions:

``pw_convergence``
    choose the plane-wave cutoff ``ecutwfc``: raise it until the *reference* total
    energy stops moving (neighbouring steps below ``tolerance_meV`` per atom).

``basis_table`` / ``pick_cheapest``
    choose ``(r_cut, l_max, ecutjy)`` of the primitive NSW basis: the criterion is the
    NSW-vs-PW energy difference on the reference geometries -- the paper's
    :math:`\\epsilon_{\\mathrm{NSW}}^{\\mathrm{PW}}` -- and among the candidates that meet
    the tolerance the cheapest one wins.

Both functions return plain dictionaries so they can be stored in an AiiDA ``Dict``
node and turned into a report without re-deriving anything.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
#  ecutwfc
# ---------------------------------------------------------------------------


def pw_convergence(
    curve: Mapping[float, Mapping[str, Any]],
    tolerance_meV: float,
) -> dict[str, Any]:
    """Turn PW total energies into an ``ecutwfc`` convergence decision.

    Parameters
    ----------
    curve
        ``{ecutwfc: {geometry: {"energy": E_eV, "n_atoms": N}}}`` (one entry per cutoff
        and reference geometry).
    tolerance_meV
        the per-atom change between *neighbouring* cutoffs that counts as converged.

    Returns
    -------
    dict
        ``{"values": [...], "steps": [...], "chosen": v, "converged": bool,
        "per_geometry": {...}}`` where ``steps[i]`` describes the move from
        ``values[i]`` to ``values[i+1]``: the maximum per-atom energy change over the
        geometries, and whether it is inside the tolerance.  ``chosen`` is the smallest
        cutoff from which *every* further step is inside the tolerance; if no such
        cutoff exists the largest value is returned with ``converged = False``.
    """
    values = sorted(float(v) for v in curve)
    if not values:
        return {"values": [], "steps": [], "chosen": None, "converged": False,
                "per_geometry": {}}

    geometries = sorted({g for entry in curve.values() for g in entry})
    per_geometry: dict[str, list[dict[str, Any]]] = {g: [] for g in geometries}
    for value in values:
        entry = curve.get(value) or curve.get(int(value)) or {}
        for geom in geometries:
            item = entry.get(geom)
            if not item:
                per_geometry[geom].append({"ecutwfc": value, "energy": None,
                                           "energy_per_atom": None})
                continue
            n_atoms = int(item.get("n_atoms") or 1)
            energy = float(item["energy"])
            per_geometry[geom].append({"ecutwfc": value, "energy": energy,
                                       "energy_per_atom": energy / n_atoms})

    steps: list[dict[str, Any]] = []
    for lower, upper in zip(values, values[1:]):
        per_geom = {}
        for geom in geometries:
            e_lo, e_hi = None, None
            for row in per_geometry[geom]:
                if row["ecutwfc"] == lower:
                    e_lo = row["energy_per_atom"]
                if row["ecutwfc"] == upper:
                    e_hi = row["energy_per_atom"]
            if e_lo is None or e_hi is None:
                continue
            per_geom[geom] = (e_hi - e_lo) * 1000.0            # meV/atom
        worst = max((abs(v) for v in per_geom.values()), default=None)
        steps.append({
            "from": lower,
            "to": upper,
            "per_geometry_meV": per_geom,
            "max_meV": worst,
            "within_tolerance": (worst is not None and worst <= tolerance_meV),
        })

    chosen = values[-1]
    converged = False
    for index, value in enumerate(values):
        rest = steps[index:]
        if rest and all(step["within_tolerance"] for step in rest):
            chosen, converged = value, True
            break
    return {
        "values": values,
        "steps": steps,
        "chosen": chosen,
        "converged": converged,
        "tolerance_meV": tolerance_meV,
        "per_geometry": per_geometry,
    }


# ---------------------------------------------------------------------------
#  (r_cut, l_max, ecutjy)
# ---------------------------------------------------------------------------


def candidate_label(candidate: Mapping[str, Any]) -> str:
    """``r11_l4_j125`` for one ``{"r_cut": 11, "l_max": 4, "ecutjy": 125}`` mapping."""
    return (f"r{candidate['r_cut']:g}_l{int(candidate['l_max'])}"
            f"_j{candidate['ecutjy']:g}")


def plan_candidates(
    *,
    r_cut_values: Sequence[float] = (),
    l_max_values: Sequence[int] = (),
    ecutjy_values: Sequence[float] = (),
    reference: Mapping[str, Any] | None = None,
    strategy: str = "ladder",
) -> list[dict[str, Any]]:
    """Build the candidate list of a basis scan.

    ``ladder`` (default and cheapest) walks outwards from ``reference``: it varies one
    super-parameter at a time -- ``ecutjy`` first (the dominant one), then ``l_max``,
    then ``r_cut`` -- and only ever *reduces* a value, so every candidate is cheaper
    than the reference.  ``exhaustive`` takes the full cartesian product instead.
    """
    r_cut_values = [float(v) for v in r_cut_values]
    l_max_values = [int(v) for v in l_max_values]
    ecutjy_values = [float(v) for v in ecutjy_values]
    if strategy == "exhaustive":
        return [{"r_cut": r, "l_max": l, "ecutjy": j}
                for r in r_cut_values or [float(reference["r_cut"])]
                for l in l_max_values or [int(reference["l_max"])]
                for j in ecutjy_values or [float(reference["ecutjy"])]]

    if strategy != "ladder":
        raise ValueError(f"unknown strategy {strategy!r} (use 'ladder' or 'exhaustive')")
    if not reference:
        raise ValueError("the 'ladder' strategy needs a reference point")

    ref = {"r_cut": float(reference["r_cut"]), "l_max": int(reference["l_max"]),
           "ecutjy": float(reference["ecutjy"])}
    out = [dict(ref)]
    seen = {candidate_label(ref)}
    for key, values in (("ecutjy", ecutjy_values), ("l_max", l_max_values),
                        ("r_cut", r_cut_values)):
        for value in values:
            item = dict(ref)
            item[key] = int(value) if key == "l_max" else float(value)
            label = candidate_label(item)
            if label in seen:
                continue
            seen.add(label)
            out.append(item)
    return out


def candidate_cost(nchi: float | None, r_cut: float | None) -> float | None:
    """The static cost proxy of one candidate: ``nchi**2 * r_cut**3`` (scaled).

    The two-centre table build of a LCAO run grows roughly with ``nchi**2`` (pairs of
    radial functions) and with the number of neighbours, which grows with ``r_cut**3``,
    so this ranks candidates the way their run time does -- and unlike measured
    ``seconds`` it is available *before* anything runs, which is what makes an
    "evaluate the cheapest first" order possible.
    """
    if not nchi or not r_cut:
        return None
    return float(nchi) ** 2 * float(r_cut) ** 3 / 1.0e6


def plan_evaluation_order(
    candidates: Sequence[Mapping[str, Any]],
    *,
    costs: Mapping[str, float | None] | None = None,
    reference_label: str | None = None,
    baseline_first: bool = False,
) -> list[dict[str, Any]]:
    """Order a ladder for evaluation: cheapest first, the reference last.

    The reference point is the *most expensive* candidate of the ladder, and a scan that
    stops at the first acceptable candidate would stop there every time -- never finding
    the reduction it exists for.  It therefore goes last and never counts as the
    reduction; the cheaper candidates are walked in increasing cost order, so the first
    one that passes **is** the cheapest one that passes.  (The reference is not
    automatically *accurate* -- it is the basis the scan starts from, nothing more -- so
    a ladder whose reference is not good enough ends with no candidate inside the
    tolerance at all.)

    ``baseline_first`` puts the reference first instead: the atomization gate compares
    every candidate against the reference row, so with that gate the baseline has to
    exist before any verdict can be given.  ``reference_label`` defaults to the first
    candidate (``plan_candidates`` puts the reference there).

    ``costs`` maps ``candidate_label -> cost``; without it the candidates are ordered by
    their parameters, which is the same ranking for a one-parameter-at-a-time ladder.
    """
    candidates = list(candidates)
    if not candidates:
        return []
    if reference_label is None:
        reference_label = candidate_label(candidates[0])

    def cost_of(candidate: Mapping[str, Any]) -> tuple[float, str]:
        label = candidate_label(candidate)
        value = (costs or {}).get(label)
        if value is None:
            ordering = (float(candidate.get("r_cut", 0)), int(candidate.get("l_max", 0)),
                        float(candidate.get("ecutjy", 0)))
            return (float("inf"), repr(ordering))
        return (float(value), label)

    reference = [c for c in candidates if candidate_label(c) == reference_label]
    rest = sorted((c for c in candidates if candidate_label(c) != reference_label),
                  key=cost_of)
    if baseline_first:
        return reference + rest
    return rest + reference


def basis_table(
    rows: Mapping[str, Mapping[str, Any]],
    tolerance_meV: float,
    *,
    reference: str | None = None,
) -> dict[str, Any]:
    """Turn the ``(NSW, PW)`` energies of every candidate into the decision table.

    Parameters
    ----------
    rows
        ``{candidate_label: {"candidate": {...}, "geometries": {name: {"e_nsw": eV,
        "e_pw": eV, "n_atoms": N}}, "seconds": s, "nchi": n, ...}}``
    tolerance_meV
        the per-atom ``|E_nsw - E_pw|`` a candidate has to meet.
    reference
        label of the candidate that plays the role of the reference point (used to
        report the *relative* change, which is what the atomization-energy criterion
        looks at).

    Returns
    -------
    dict
        ``{"rows": [...], "best": label | None, "within_tolerance": [...],
        "reference": label}``; each row carries the max/min ``dE/atom`` over the
        geometries, the atomization-energy difference against the reference when a
        monomer is present, and the cost proxy ``nchi**2 * r_cut**3``.
    """
    table = []
    for label, entry in rows.items():
        candidate = entry.get("candidate") or {}
        geometries = entry.get("geometries") or {}
        deltas = {}
        for name, item in geometries.items():
            e_nsw, e_pw = item.get("e_nsw"), item.get("e_pw")
            if e_nsw is None or e_pw is None:
                continue
            n_atoms = int(item.get("n_atoms") or 1)
            deltas[name] = (float(e_nsw) - float(e_pw)) / n_atoms * 1000.0   # meV/atom
        dimer = {k: v for k, v in deltas.items() if "monomer" not in k}
        atomization = _atomization_difference(geometries)
        nchi = entry.get("nchi")
        r_cut = candidate.get("r_cut")
        cost = candidate_cost(nchi, r_cut)
        table.append({
            "label": label,
            "candidate": dict(candidate),
            "dE_per_atom_meV": deltas,
            "dE_max_meV": max(dimer.values()) if dimer else None,
            "dE_max_abs_meV": max((abs(v) for v in dimer.values()), default=None),
            "atomization_meV": atomization,
            "tolerance_ok": bool(dimer) and max(abs(v) for v in dimer.values()) <= tolerance_meV,
            "seconds": entry.get("seconds"),
            "nchi": nchi,
            "cost": cost,
        })

    if reference is None and table:
        reference = max(
            table,
            key=lambda row: (row["candidate"].get("r_cut", 0),
                             row["candidate"].get("l_max", 0),
                             row["candidate"].get("ecutjy", 0)),
        )["label"]
    ref_atomization = None
    if reference:
        for row in table:
            if row["label"] == reference:
                ref_atomization = _atomization_difference(
                    (rows.get(reference) or {}).get("geometries") or {})
    for row in table:
        if row["atomization_meV"] is not None and ref_atomization is not None:
            row["atomization_vs_reference_meV"] = row["atomization_meV"] - ref_atomization
        else:
            row["atomization_vs_reference_meV"] = None

    order = {"seconds": lambda r: r["seconds"] if r["seconds"] is not None else float("inf"),
             "cost": lambda r: r["cost"] if r["cost"] is not None else float("inf")}
    return {
        "rows": table,
        "reference": reference,
        "within_tolerance": [r["label"] for r in table if r["tolerance_ok"]],
        "tolerance_meV": tolerance_meV,
        "sort_keys": list(order),
    }


def _atomization_difference(geometries: Mapping[str, Any]) -> float | None:
    """``[(E_dimer - 2 E_monomer)_NSW - (…)_PW]`` in meV, or ``None`` without a monomer.

    An energy *difference* of this kind is what the defect-energy criterion looks at:
    a large part of the basis-set error cancels between the two bases.
    """
    monomer = None
    for name, item in geometries.items():
        if "monomer" in name and item.get("e_nsw") is not None and item.get("e_pw") is not None:
            monomer = item
            break
    if monomer is None:
        return None
    out = {}
    for name, item in geometries.items():
        if "monomer" in name:
            continue
        if item.get("e_nsw") is None or item.get("e_pw") is None:
            continue
        nsw = item["e_nsw"] - 2.0 * monomer["e_nsw"]
        pw = item["e_pw"] - 2.0 * monomer["e_pw"]
        out[name] = (nsw - pw) * 1000.0
    if not out:
        return None
    return max(out.values(), key=abs)


def describe_row(row: Mapping[str, Any], tolerance_meV: float | None = None) -> str:
    """One report line for a row of :func:`basis_table`.

    Kept here (not in the WorkChain) so the wording of the decision -- which geometry
    moved how much, what it cost -- is testable without a database, and so that the
    two scans that print it cannot drift apart.
    """
    label = row.get("label")
    worst = row.get("dE_max_abs_meV")
    mark = "✓" if row.get("tolerance_ok") else "✗"
    limit = tolerance_meV if tolerance_meV is not None else row.get("tolerance_meV")
    worst_txt = "n/a" if worst is None else f"{float(worst):.2f}"
    dE = row.get("dE_per_atom_meV") or {}
    per_geometry = ", ".join(
        f"{name} {float(value):+.1f}" for name, value in sorted(dE.items())
    )
    cost = row.get("seconds")
    cost_txt = f"{float(cost):.0f}s" if cost is not None else f"nchi={row.get('nchi')}"
    atom = row.get("atomization_vs_reference_meV")
    atom_txt = "" if atom is None else f", dA={float(atom):+.1f} meV"
    line = f"{label}: max |dE|/atom = {worst_txt} meV {mark}"
    if limit is not None:
        line += f" (tolerance {float(limit):g} meV)"
    if per_geometry:
        line += f" [{per_geometry}]"
    return line + f" ({cost_txt}{atom_txt})"


def no_basis_message(
    table: Mapping[str, Any],
    tolerance_meV: float,
    *,
    reference: str | None = None,
    atomization_tolerance_meV: float | None = None,
) -> str:
    """Why nothing was chosen, and what to change.

    Three different situations end a scan with no winner, and they need different
    answers: the ladder never made it (some candidates were not evaluated), the
    reference point itself is not accurate enough (the ladder cannot reduce from it),
    or the tolerance/gate is simply stricter than this basis family can be.
    """
    rows = list(table.get("rows") or [])
    if not rows:
        return ("no candidate could be evaluated: check the failed children of the "
                "previous steps")
    passed = [row["label"] for row in rows if row.get("tolerance_ok")]
    worst = min(row["dE_max_abs_meV"] for row in rows
                if row.get("dE_max_abs_meV") is not None) if any(
        row.get("dE_max_abs_meV") is not None for row in rows) else None
    reference_row = next((row for row in rows if row["label"] == reference), None)
    parts = [f"no candidate is inside {tolerance_meV:g} meV/atom"]
    if reference_row is not None and not reference_row.get("tolerance_ok"):
        parts.append(
            f"and the reference point ({reference}) is not either "
            f"({reference_row.get('dE_max_abs_meV')} meV/atom): the ladder cannot reduce "
            f"from a basis that is itself not accurate enough -- start from a larger "
            f"reference point (r_cut / l_max / ecutjy)"
        )
    elif worst is not None:
        parts.append(
            f"the best candidate is {float(worst):.2f} meV/atom: widen the ladder, or "
            f"relax tolerance_meV"
        )
    if passed and atomization_tolerance_meV is not None:
        parts.append(
            f"{len(passed)} candidate(s) met tolerance_meV but were rejected by the "
            f"atomization gate ({atomization_tolerance_meV:g} meV): relax it if the "
            f"defect-like criterion is not what you are testing"
        )
    return "; ".join(parts)


def pick_cheapest(
    table: Mapping[str, Any],
    *,
    by: str = "seconds",
    require_atomization: bool = False,
    atomization_tolerance_meV: float | None = None,
    reference: str | None = None,
    reference_is_fallback: bool = True,
) -> dict[str, Any] | None:
    """The cheapest candidate that meets the tolerance (``seconds`` or the cost proxy).

    Parameters
    ----------
    by
        ``"seconds"`` (measured resource use), ``"cost"`` or ``"nchi"`` (static proxies).
        A candidate whose ``seconds`` are unknown (everything came out of AiiDA's cache)
        is ranked by the ``cost`` proxy instead of by a meaningless few seconds.
    require_atomization, atomization_tolerance_meV
        additionally require the atomization-energy difference against the reference to
        stay inside ``atomization_tolerance_meV`` -- the stricter, defect-like criterion.
    reference, reference_is_fallback
        The reference point is the candidate the ladder starts from, i.e. the *most
        expensive* one; it is the answer only when nothing else is usable.  It was
        looked up by cost before, which went wrong as soon as its children came from the
        cache: it then appeared to be the cheapest candidate and was returned as the
        winner next to a warning that no cheaper candidate had passed -- while one had.
    """
    rows = list(table.get("rows") or [])

    def window(key: str, row: Mapping[str, Any]) -> float:
        value = row.get(key)
        if value is not None:
            return float(value)
        if key == "seconds" and row.get("cost") is not None:
            return float(row["cost"])
        return float("inf")

    if by not in ("seconds", "cost", "nchi"):
        raise ValueError(f"unknown sort key {by!r}; use one of ['cost', 'nchi', 'seconds']")
    usable = [r for r in rows if r["tolerance_ok"]]
    if require_atomization and atomization_tolerance_meV is not None:
        usable = [r for r in usable
                  if r["atomization_vs_reference_meV"] is not None
                  and abs(r["atomization_vs_reference_meV"]) <= atomization_tolerance_meV]
    if reference and reference_is_fallback:
        others = [r for r in usable if r["label"] != reference]
        if others:
            usable = others
    if not usable:
        return None
    usable.sort(key=lambda r: (window(by, r), r["dE_max_abs_meV"] or 0.0))
    return usable[0]


__all__ = [
    "basis_table",
    "candidate_cost",
    "candidate_label",
    "describe_row",
    "no_basis_message",
    "pick_cheapest",
    "plan_candidates",
    "plan_evaluation_order",
    "pw_convergence",
]
