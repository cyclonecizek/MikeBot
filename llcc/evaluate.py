"""The evaluator.

Pure function of (WorldSnapshot, t). Three stages:

  1. Section 4.3 pre-pass builds assessment units from the physical-connection
     graph, honouring the 4.3a-d carve-outs that keep certain pairings
     independent and the 4.3e fallback for three or more types.
  2. Every encoded requirement is walked against every applicable unit.
  3. Results reduce to a mission verdict by taking the worst state, with the
     Appendix A manifest forcing indeterminate on any unencoded requirement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum

from .expr import Context, Trace
from .requirements import ENCODED, MANIFEST, Requirement, missing_ids
from .tvl import Tri, not_, or_
from .world import CloudObject, CloudType, WorldSnapshot


class State(Enum):
    GO = 0
    NO_GO_UNTIL = 1
    INDETERMINATE = 2
    NO_GO = 3

    def __str__(self) -> str:
        return {"GO": "GO", "NO_GO_UNTIL": "NO-GO until",
                "INDETERMINATE": "indeterminate", "NO_GO": "NO-GO"}[self.name]


@dataclass
class Result:
    requirement_id: str
    section: str
    title: str
    unit_id: str
    state: State
    release: datetime | None
    trigger: Trace
    exception: Trace

    @property
    def blocking(self) -> bool:
        return self.state is not State.GO


_INDEPENDENT_PAIRS = [
    ({CloudType.ATTACHED_ANVIL}, {CloudType.CUMULUS, CloudType.SMOKE_CUMULUS}),
    ({CloudType.DETACHED_ANVIL}, {CloudType.CUMULUS, CloudType.SMOKE_CUMULUS}),
    ({CloudType.THICK_LAYER}, {CloudType.CUMULUS}),
]

_TYPE_SEVERITY = [
    CloudType.CUMULUS,
    CloudType.SMOKE_CUMULUS,
    CloudType.DEBRIS,
    CloudType.ATTACHED_ANVIL,
    CloudType.DETACHED_ANVIL,
    CloudType.THICK_LAYER,
    CloudType.CIRRIFORM,
]


def _min(values):
    vals = [v for v in values if v is not None]
    return min(vals) if vals else None


def _max(values):
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _all_true(values):
    if any(v is None for v in values):
        return None
    return all(values)


def _any_true(values):
    if any(v is True for v in values):
        return True
    if any(v is None for v in values):
        return None
    return False


def _combine(members: list[CloudObject], unit_id: str) -> CloudObject:
    """Conservative reduction over a physically connected cluster (LLCCR 35).

    Distances take the minimum, tops take the coldest, reflectivities take
    the maximum, 'entirely colder than' predicates must hold for every
    member, and hazard flags are disjunctive.
    """
    types = [m.cloud_type for m in members if m.cloud_type != CloudType.UNCLASSIFIED]
    if types:
        chosen = min(types, key=lambda t: _TYPE_SEVERITY.index(t)
                     if t in _TYPE_SEVERITY else 99)
    else:
        chosen = CloudType.UNCLASSIFIED

    merged = replace(
        members[0],
        id=unit_id,
        cloud_type=chosen,
        parent_ids=sorted({p for m in members for p in m.parent_ids} | {m.id for m in members}),
        unknown_provenance=any(m.unknown_provenance for m in members),
        observability=min(m.observability for m in members),
        slant_min_nmi=_min(m.slant_min_nmi for m in members),
        horiz_min_nmi=_min(m.horiz_min_nmi for m in members),
        intersects_corridor=_any_true([m.intersects_corridor for m in members]),
        top_temp_c=_min(m.top_temp_c for m in members),
        coldest_top_temp_c=_min(m.coldest_top_temp_c for m in members),
        parent_coldest_top_temp_c=_min(m.parent_coldest_top_temp_c for m in members),
        portion_within_5nmi_all_below_0c=_all_true(
            [m.portion_within_5nmi_all_below_0c for m in members]),
        portion_within_10nmi_all_below_0c=_all_true(
            [m.portion_within_10nmi_all_below_0c for m in members]),
        entirely_colder_than_15c=_all_true([m.entirely_colder_than_15c for m in members]),
        contains_liquid_water=_any_true([m.contains_liquid_water for m in members]),
        ever_connected_to_convection=_any_true(
            [m.ever_connected_to_convection for m in members]),
        mrr_max_within_1nmi_dbz=_max(m.mrr_max_within_1nmi_dbz for m in members),
        mrr_max_in_corridor_dbz=_max(m.mrr_max_in_corridor_dbz for m in members),
        mrr_valid=_all_true([m.mrr_valid for m in members]),
        layer_thickness_m=_max(m.layer_thickness_m for m in members),
        layer_spans_0_to_minus20=_any_true([m.layer_spans_0_to_minus20 for m in members]),
        producing_precip=_any_true([m.producing_precip for m in members]),
        moderate_precip_within_5nmi=_any_true(
            [m.moderate_precip_within_5nmi for m in members]),
        bright_band_within_5nmi=_any_true([m.bright_band_within_5nmi for m in members]),
    )
    return merged


def assessment_units(snapshot: WorldSnapshot) -> list[CloudObject]:
    """Section 4.3 pre-pass."""
    units: list[CloudObject] = []
    for cluster in snapshot.connected_clusters():
        cluster = {i for i in cluster if i != "DOMAIN"}
        if not cluster:
            continue
        members = [snapshot.objects[i] for i in sorted(cluster)]
        if len(members) == 1:
            units.append(members[0])
            continue

        types = {m.cloud_type for m in members}

        promoted = []
        for m in members:
            # 4.3b: a detached anvil connected to a cumulus that has ever had a
            # top colder than -10 C is assessed as an attached anvil.
            if (m.cloud_type == CloudType.DETACHED_ANVIL
                    and any(o.cloud_type in (CloudType.CUMULUS, CloudType.SMOKE_CUMULUS)
                            and (o.coldest_top_temp_c is not None
                                 and o.coldest_top_temp_c <= -10.0)
                            for o in members)):
                promoted.append(replace(m, cloud_type=CloudType.ATTACHED_ANVIL))
            else:
                promoted.append(m)

        independent = any(
            types <= (a | b) and types & a and types & b
            for a, b in _INDEPENDENT_PAIRS
        )
        if independent and len(types) == 2:
            units.extend(promoted)
        else:
            # Default LLCCR 35, and the 4.3e three-or-more-types fallback.
            units.append(_combine(promoted, "cluster:" + "+".join(sorted(cluster))))
    return units


def evaluate_one(req: Requirement, unit: CloudObject,
                 snapshot: WorldSnapshot, now: datetime) -> Result | None:
    ctx = Context(snapshot=snapshot, obj=unit, now=now)

    scope_value = Tri.TRUE
    if req.scope is not None:
        scope = req.scope.eval(ctx)
        # LLCCR 1: where it is unclear which criteria apply, apply them all.
        # Only a definitely-false scope removes a requirement from play.
        if scope.value is Tri.FALSE:
            return None
        scope_value = scope.value

    trigger = req.trigger.eval(ctx)
    exception = req.exception.eval(ctx)
    permitted = or_(not_(trigger.value), exception.value)

    if permitted is Tri.TRUE:
        return Result(req.id, req.section, req.title, unit.id,
                      State.GO, None, trigger, exception)

    candidates = req.trigger.release_times(ctx) + req.exception.release_times(ctx)
    release = min(candidates) if candidates else None

    # A blocking result under an unresolved scope still blocks, but it is
    # reported as indeterminate rather than as a definite violation: we do
    # not know that this requirement applies to this object.
    if permitted is Tri.UNKNOWN or scope_value is Tri.UNKNOWN:
        return Result(req.id, req.section, req.title, unit.id,
                      State.INDETERMINATE, release, trigger, exception)

    state = State.NO_GO_UNTIL if release else State.NO_GO
    return Result(req.id, req.section, req.title, unit.id,
                  state, release, trigger, exception)


@dataclass
class Verdict:
    time: datetime
    state: State
    results: list[Result]
    earliest_change: datetime | None
    manifest_gaps: list[str]

    @property
    def blocking(self) -> list[Result]:
        return [r for r in self.results if r.blocking]


def evaluate(snapshot: WorldSnapshot, now: datetime | None = None,
             assume_manifest_complete: bool = False) -> Verdict:
    now = now or snapshot.time
    units = assessment_units(snapshot)

    domain = snapshot.objects.get("DOMAIN")
    results: list[Result] = []

    for req in ENCODED:
        targets = [domain] if not req.per_object else units
        for unit in targets:
            if unit is None:
                continue
            res = evaluate_one(req, unit, snapshot, now)
            if res is not None:
                results.append(res)

    gaps = missing_ids()
    state = max((r.state for r in results), default=State.GO, key=lambda s: s.value)

    if gaps and not assume_manifest_complete:
        state = max(state, State.INDETERMINATE, key=lambda s: s.value)

    blocking_releases = [r.release for r in results if r.blocking and r.release]
    earliest = max(blocking_releases) if blocking_releases else None

    return Verdict(now, state, results, earliest, gaps)


def format_verdict(verdict: Verdict, verbose: bool = False) -> str:
    lines = [f"{verdict.time:%Y-%m-%d %H:%M:%S}Z   {verdict.state}"]
    if verdict.earliest_change:
        lines.append(f"  earliest re-evaluation with a chance to clear: "
                     f"{verdict.earliest_change:%H:%M:%S}Z")
    if verdict.manifest_gaps:
        lines.append(f"  manifest: {len(verdict.manifest_gaps)} of {len(MANIFEST)} "
                     f"requirements unencoded -> global indeterminate")
    for r in sorted(verdict.blocking, key=lambda r: (int(r.requirement_id.split()[1]), r.unit_id)):
        tail = f" {r.release:%H:%M:%S}Z" if r.release else ""
        lines.append(f"  {r.requirement_id:<9} {r.section:<9} {str(r.state)}{tail}"
                     f"   [{r.unit_id}] {r.title}")
        if verbose:
            lines.append("    trigger:")
            lines.append(_indent(r.trigger.render(), 6))
            lines.append("    exception:")
            lines.append(_indent(r.exception.render(), 6))
    return "\n".join(lines)


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())
