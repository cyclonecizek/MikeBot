"""Launch Weather Officer reclassification overrides.

An override is an *input document*, never an edit to the object record. The
evaluator stays a pure function of (snapshot, t); overrides are applied by a
pass that produces a new snapshot, so the same afternoon can always be
replayed both ways and the human's contribution isolated exactly.

Three things this layer enforces:

  Companion facts. A type label is not free-standing. Calling something a
  detached anvil asserts a detachment time and a parent that reached -10 C,
  because LLCCR 15a, 16b and 17 all depend on them. If the officer cannot
  supply those, the object degrades to unclassified rather than being
  granted the label, which forces the conservative branch.

  Lineage binding. The override was a judgment about an object the tracker
  called X. If the tracker now calls it something else, or the object is
  gone, the basis has changed and the override needs re-affirmation.

  Direction. Reclassification swaps which requirements apply, so it can
  relax as easily as restrict. A relaxing override is exactly what LLCCR 3
  guards against, so it requires a justification and both verdicts are
  recorded. It is not forbidden -- the officer knows things the radar does
  not -- but it is never silent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

from .world import CloudObject, CloudType, Event, EventKind, WorldSnapshot

DEFAULT_TTL = timedelta(hours=1)

# Facts a type label asserts, which the tracker will not have supplied when
# it labelled the object something else.
REQUIRED_FACTS: dict[str, tuple[str, ...]] = {
    CloudType.DETACHED_ANVIL: ("detachment_time", "parent_coldest_top_temp_c"),
    CloudType.ATTACHED_ANVIL: ("parent_coldest_top_temp_c",),
    CloudType.DEBRIS: ("period_start_time",),
    CloudType.THICK_LAYER: ("layer_thickness_m",),
    CloudType.CUMULUS: ("top_temp_c",),
    CloudType.SMOKE_CUMULUS: ("top_temp_c",),
    CloudType.CIRRIFORM: ("top_temp_c",),
    CloudType.UNCLASSIFIED: (),
}

_FACT_TO_ATTR = {
    "parent_coldest_top_temp_c": "parent_coldest_top_temp_c",
    "layer_thickness_m": "layer_thickness_m",
    "top_temp_c": "top_temp_c",
}


class Status:
    APPLIED = "applied"
    DEGRADED = "degraded"          # companion facts missing -> unclassified
    EXPIRED = "expired"
    STALE = "stale"                # tracker label changed under the override
    ORPHANED = "orphaned"          # object no longer present


@dataclass
class Override:
    object_id: str
    cloud_type: str
    issued_at: datetime
    issued_by: str
    original_type: str
    justification: str = ""
    facts: dict = field(default_factory=dict)
    ttl_seconds: float = DEFAULT_TTL.total_seconds()
    affirmed_at: datetime | None = None

    @property
    def effective_from(self) -> datetime:
        return self.affirmed_at or self.issued_at

    def expires_at(self) -> datetime:
        return self.effective_from + timedelta(seconds=self.ttl_seconds)

    def missing_facts(self, obj: CloudObject) -> list[str]:
        out = []
        for name in REQUIRED_FACTS.get(self.cloud_type, ()):
            if self.facts.get(name) is not None:
                continue
            attr = _FACT_TO_ATTR.get(name)
            if attr and getattr(obj, attr, None) is not None:
                continue
            out.append(name)
        return out


@dataclass
class Application:
    override: Override
    status: str
    detail: str = ""
    applied_type: str | None = None


def _dt(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def load_overrides(path: str | Path) -> list[Override]:
    raw = json.loads(Path(path).read_text())
    return [
        Override(
            object_id=o["object_id"],
            cloud_type=o["cloud_type"],
            issued_at=_dt(o["issued_at"]),
            issued_by=o["issued_by"],
            original_type=o.get("original_type", CloudType.UNCLASSIFIED),
            justification=o.get("justification", ""),
            facts=o.get("facts", {}),
            ttl_seconds=float(o.get("ttl_seconds", DEFAULT_TTL.total_seconds())),
            affirmed_at=_dt(o.get("affirmed_at")),
        )
        for o in raw.get("overrides", [])
    ]


def apply_overrides(snapshot: WorldSnapshot, overrides: list[Override],
                    now: datetime | None = None
                    ) -> tuple[WorldSnapshot, list[Application]]:
    """Produce a new snapshot with overrides applied, plus a record of what
    happened to each one. Never mutates the input."""
    now = now or snapshot.time
    objects = dict(snapshot.objects)
    events = list(snapshot.events)
    applications: list[Application] = []

    for ov in overrides:
        if ov.effective_from > now:
            continue

        obj = objects.get(ov.object_id)
        if obj is None:
            applications.append(Application(ov, Status.ORPHANED,
                                            "object not present in this snapshot"))
            continue

        if now > ov.expires_at():
            applications.append(Application(
                ov, Status.EXPIRED,
                f"lapsed {ov.expires_at():%H:%M:%S}Z, needs re-affirmation"))
            continue

        if obj.cloud_type != ov.original_type:
            applications.append(Application(
                ov, Status.STALE,
                f"tracker now labels this {obj.cloud_type}, not {ov.original_type}"))
            continue

        missing = ov.missing_facts(obj)
        if missing:
            objects[ov.object_id] = replace(
                obj, cloud_type=CloudType.UNCLASSIFIED,
                overridden_by=ov.issued_by, override_original_type=obj.cloud_type)
            applications.append(Application(
                ov, Status.DEGRADED,
                "missing " + ", ".join(missing) + "; degraded to unclassified",
                CloudType.UNCLASSIFIED))
            continue

        updates: dict = {"cloud_type": ov.cloud_type,
                         "overridden_by": ov.issued_by,
                         "override_original_type": obj.cloud_type}
        for name, attr in _FACT_TO_ATTR.items():
            if ov.facts.get(name) is not None:
                updates[attr] = ov.facts[name]
        objects[ov.object_id] = replace(obj, **updates)

        # A detachment time is an assertion about the event log, so it has to
        # reach the log or the LLCCR 15a and 4.1.6.1 clocks will not see it.
        if ov.facts.get("detachment_time"):
            events.append(Event(EventKind.DETACHMENT, _dt(ov.facts["detachment_time"]),
                                ov.object_id, f"override:{ov.issued_by}"))
        if ov.facts.get("period_start_time"):
            events.append(Event(EventKind.PARENT_TOP_COLLAPSE,
                                _dt(ov.facts["period_start_time"]),
                                ov.object_id, f"override:{ov.issued_by}"))

        applications.append(Application(ov, Status.APPLIED,
                                        f"{ov.original_type} -> {ov.cloud_type}",
                                        ov.cloud_type))

    new_snapshot = replace(snapshot, objects=objects,
                           events=sorted(events, key=lambda e: e.time))
    return new_snapshot, applications


@dataclass
class Diff:
    """Effect of the override set on the verdict, in both directions."""

    baseline_state: str
    overridden_state: str
    direction: str                      # restricts | relaxes | unchanged | mixed
    changes: list[dict] = field(default_factory=list)
    relaxations: list[dict] = field(default_factory=list)
    unjustified: list[str] = field(default_factory=list)


def diff_verdicts(baseline, overridden, overrides: list[Override]) -> Diff:
    """Compare the verdict with and without overrides.

    State ordering is the State enum's own: GO < NO_GO_UNTIL < INDETERMINATE
    < NO_GO. A change to a higher value restricts, a change to a lower value
    relaxes.
    """
    def key(result):
        return (result.requirement_id, result.unit_id)

    before = {key(r): r for r in baseline.results}
    after = {key(r): r for r in overridden.results}

    # A requirement that stops or starts applying is ranked as GO, because
    # "does not apply" and "applies and is satisfied" are equally permissive.
    # Only a move up or down the blocking scale counts as a direction.
    def rank(result):
        return result.state.value if result else 0

    changes, relaxations, restrictions = [], [], []
    for k in sorted(set(before) | set(after)):
        b, a = before.get(k), after.get(k)
        b_state = b.state.name if b else "n/a"
        a_state = a.state.name if a else "n/a"
        if b_state == a_state:
            continue
        b_rank, a_rank = rank(b), rank(a)
        if a_rank > b_rank:
            direction = "restricts"
        elif a_rank < b_rank:
            direction = "relaxes"
        else:
            direction = "neutral"
        entry = {"requirement_id": k[0], "unit_id": k[1],
                 "before": b_state, "after": a_state, "direction": direction}
        changes.append(entry)
        if direction == "relaxes":
            relaxations.append(entry)
        elif direction == "restricts":
            restrictions.append(entry)

    if relaxations and restrictions:
        direction = "mixed"
    elif relaxations:
        direction = "relaxes"
    elif restrictions:
        direction = "restricts"
    else:
        direction = "unchanged"

    unjustified = sorted({
        ov.object_id for ov in overrides if not ov.justification.strip()
    }) if relaxations else []

    return Diff(
        baseline_state=baseline.state.name,
        overridden_state=overridden.state.name,
        direction=direction,
        changes=changes,
        relaxations=relaxations,
        unjustified=unjustified,
    )


def application_to_dict(app: Application) -> dict:
    ov = app.override
    return {
        "object_id": ov.object_id,
        "requested_type": ov.cloud_type,
        "original_type": ov.original_type,
        "applied_type": app.applied_type,
        "status": app.status,
        "detail": app.detail,
        "issued_by": ov.issued_by,
        "issued_at": ov.issued_at.isoformat(),
        "affirmed_at": ov.affirmed_at.isoformat() if ov.affirmed_at else None,
        "expires_at": ov.expires_at().isoformat(),
        "justification": ov.justification,
        "facts": {k: (v.isoformat() if isinstance(v, datetime) else v)
                  for k, v in ov.facts.items()},
    }


def diff_to_dict(diff: Diff) -> dict:
    return {
        "baseline_state": diff.baseline_state,
        "overridden_state": diff.overridden_state,
        "direction": diff.direction,
        "changes": diff.changes,
        "relaxations": diff.relaxations,
        "unjustified_relaxations": diff.unjustified,
    }
