"""Serialisers for the backend-to-renderer contract.

Two documents:

  snapshot.json  what the world looked like  (geometry engine output)
  verdict.json   what the evaluator concluded (evaluator output)

The snapshot schema is deliberately the same shape `replay.py` already
consumes, so a live backend and an archived scenario are interchangeable
inputs. Pin SCHEMA_VERSION and bump it on any breaking change; the renderer
refuses documents it does not recognise.

Nothing here evaluates anything. The verdict is produced server-side once
and published as a durable artifact, so the display renders a decision
rather than re-deriving one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path

from .evaluate import Result, Verdict
from .requirements import rule_name
from .expr import Trace
from .world import CloudObject, WorldSnapshot

SCHEMA_VERSION = "llcc-1"

# Standoff each object actually demands, given its present state -- not the
# widest any applicable requirement could ever demand. A warm shallow cumulus
# is governed only by the through-cloud rule in 4.1.3.1 and has no standoff at
# all; the same cloud with a -25 C top demands 10 nmi under 4.1.3.3. Drawing
# a fixed ring per type would overstate the first and understate nothing,
# which sounds safe but teaches the operator to disbelieve the ring.


_SKIP_FIELDS = {"series"}


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def object_to_dict(obj: CloudObject) -> dict:
    out = {}
    for f in fields(obj):
        if f.name in _SKIP_FIELDS:
            continue
        value = getattr(obj, f.name)
        out[f.name] = _iso(value) if isinstance(value, datetime) else value
    return out


def snapshot_to_dict(snapshot: WorldSnapshot, pad: str = "",
                     azimuth_deg: float | None = None,
                     radar=None, origin: dict | None = None,
                     overlay: dict | None = None,
                     classifier: dict | None = None) -> dict:
    doc = {
        "schema": SCHEMA_VERSION,
        "kind": "snapshot",
        "time": _iso(snapshot.time),
        "pad": pad,
        "azimuth_deg": azimuth_deg,
        # Grid origin in geographic coordinates, so the display can place map
        # tiles under the local ENU frame. Defaults to LC-39A.
        "origin": origin or {"lat": 28.6083, "lon": -80.6041},
        "sites": [
            {"name": "LC-39A", "lat": 28.6083, "lon": -80.6041, "kind": "pad"},
            {"name": "LC-39B", "lat": 28.6272, "lon": -80.6208, "kind": "pad"},
            {"name": "SLC-40", "lat": 28.5619, "lon": -80.5772, "kind": "pad"},
            {"name": "SLC-41", "lat": 28.5833, "lon": -80.5833, "kind": "pad"},
            {"name": "KMLB", "lat": 28.1131, "lon": -80.6544, "kind": "radar"},
        ],
        "field_mills_available": snapshot.field_mills_available,
        "disturbed_weather": snapshot.disturbed_weather,
        "vehicle": (
            {
                "name": snapshot.vehicle.name,
                "triboelectric_exemption": snapshot.vehicle.triboelectric_exemption,
                "triboelectric_basis": snapshot.vehicle.triboelectric_basis,
            }
            if snapshot.vehicle else None
        ),
        "profile": {
            "uncertainty_m": snapshot.profile.uncertainty_m,
            "levels": [list(level) for level in snapshot.profile.levels],
            "isotherms": {
                str(int(t)): snapshot.profile.isotherm_altitude(t)
                for t in (5, 0, -5, -10, -15, -20)
            },
        },
        "objects": {oid: object_to_dict(o) for oid, o in snapshot.objects.items()},
        "connections": [list(pair) for pair in snapshot.connections],
        "events": [
            {
                "kind": e.kind,
                "time": _iso(e.time),
                "object_id": e.object_id,
                "source": e.source,
                "detail": e.detail,
            }
            for e in snapshot.events
        ],
    }
    if radar is not None:
        doc["radar"] = radar.to_dict()
    if overlay:
        doc["overlay"] = overlay
    if classifier:
        doc["classifier"] = classifier
    return doc


def trace_to_dict(trace: Trace) -> dict:
    return {
        "label": trace.label,
        "value": str(trace.value),
        "detail": trace.detail,
        "children": [trace_to_dict(c) for c in trace.children],
    }


def result_to_dict(result: Result) -> dict:
    return {
        "requirement_id": result.requirement_id,
        "rule_name": rule_name(result.section),
        "section": result.section,
        "title": result.title,
        "unit_id": result.unit_id,
        "state": result.state.name,
        "release": _iso(result.release),
        "trigger": trace_to_dict(result.trigger),
        "exception": trace_to_dict(result.exception),
    }


def _minutes_since(snapshot, object_id: str, kind: str, now) -> float | None:
    when = snapshot.last_event_time(object_id, kind)
    return None if when is None else (now - when).total_seconds() / 60.0


def required_standoff_nmi(obj, snapshot, now) -> float:
    """The standoff this object demands right now, in nautical miles.

    Zero means the object is governed only by a through-cloud rule: it
    matters if the flight path goes into it, and not otherwise. An unknown
    type or an unknown cloud top returns the widest standoff in the
    relevant section, because those are the cases where nothing has been
    ruled out.
    """
    from .world import CloudType

    kind = obj.cloud_type
    top = obj.top_temp_c

    if kind in (CloudType.CUMULUS, CloudType.SMOKE_CUMULUS):
        # 4.1.3.2 and 4.1.3.3. Below -10 C only the through-cloud rule in
        # 4.1.3.1 applies, so there is no standoff to draw.
        if top is None:
            return 10.0
        if top <= -20.0:
            return 10.0
        if top <= -10.0:
            return 5.0
        return 0.0

    if kind == CloudType.ATTACHED_ANVIL:
        parent = obj.parent_coldest_top_temp_c
        if parent is not None and parent > -10.0:
            return 0.0                       # out of scope for 4.1.4
        since = _minutes_since(snapshot, obj.id, "lightning", now)
        if since is not None and since <= 30.0:
            return 10.0                      # 4.1.4.3
        if since is not None and since <= 180.0:
            return 5.0                       # 4.1.4.2
        return 3.0                           # 4.1.4.1 applies regardless

    if kind == CloudType.DETACHED_ANVIL:
        since = _minutes_since(snapshot, obj.id, "lightning", now)
        if since is not None and since <= 30.0:
            return 10.0                      # 4.1.5.3
        return 3.0                           # 4.1.5.2

    if kind == CloudType.DEBRIS:
        # 4.1.6.3 applies only during the three-hour period of 4.1.6.1.
        starts = [
            _minutes_since(snapshot, obj.id, k, now)
            for k in ("detachment", "parent_top_collapse", "lightning")
        ]
        elapsed = [m for m in starts if m is not None]
        if elapsed and min(elapsed) <= 180.0:
            return 3.0
        return 0.0

    if kind in (CloudType.THICK_LAYER, CloudType.CIRRIFORM):
        # 4.1.8 is a through-cloud rule. The 5 nmi in 4.1.8.1a says which
        # part of the layer counts toward thickness, not how far to stand off.
        return 0.0

    return 10.0                              # unclassified: nothing ruled out


def unit_buffers(verdict: Verdict, snapshot=None) -> dict[str, dict]:
    """Per-unit display buffer: the widest standoff any applicable
    requirement implicates, and the worst state among them.

    Every unit in scope gets an entry, not only the blocking ones. An
    officer judging distance needs to see how far a cloud has to be before
    it stops mattering, and a standoff that is currently satisfied is
    exactly the one worth watching as the cloud drifts.
    """
    out: dict[str, dict] = {}
    for r in verdict.results:
        entry = out.setdefault(r.unit_id, {"buffer_nmi": 0.0, "state": "GO",
                                           "requirement_ids": []})
        if r.blocking:
            entry["requirement_ids"].append(r.requirement_id)
        order = ["GO", "NO_GO_UNTIL", "INDETERMINATE", "NO_GO"]
        if order.index(r.state.name) > order.index(entry["state"]):
            entry["state"] = r.state.name

    if snapshot is None:
        return out

    # Section 4.3 assessment units can be clusters, whose id names several
    # objects rather than one. The display draws per object, so a cluster's
    # verdict has to reach each of its members -- otherwise a physically
    # connected group renders with no standoff at all.
    expanded: dict[str, dict] = {}
    for unit_id, entry in out.items():
        members = ([m for m in unit_id[len("cluster:"):].split("+")]
                   if unit_id.startswith("cluster:") else [unit_id])
        for member in members:
            obj = snapshot.objects.get(member)
            if obj is None:
                expanded.setdefault(member, dict(entry))
                continue
            merged = expanded.setdefault(member, {
                "buffer_nmi": 0.0, "state": "GO", "requirement_ids": [],
                "unit_id": unit_id})
            merged["requirement_ids"] = sorted(
                set(merged["requirement_ids"]) | set(entry["requirement_ids"]))
            order = ["GO", "NO_GO_UNTIL", "INDETERMINATE", "NO_GO"]
            if order.index(entry["state"]) > order.index(merged["state"]):
                merged["state"] = entry["state"]
            merged["buffer_nmi"] = max(
                merged["buffer_nmi"],
                required_standoff_nmi(obj, snapshot, verdict.time))
    return expanded


def verdict_to_dict(verdict: Verdict, applications=None, diff=None,
                    snapshot=None) -> dict:
    from .overrides import application_to_dict, diff_to_dict

    doc = {
        "schema": SCHEMA_VERSION,
        "kind": "verdict",
        "time": _iso(verdict.time),
        "state": verdict.state.name,
        "earliest_change": _iso(verdict.earliest_change),
        "manifest_gaps": verdict.manifest_gaps,
        "units": unit_buffers(verdict, snapshot),
        "results": [result_to_dict(r) for r in verdict.results],
    }
    if applications is not None:
        doc["overrides"] = [application_to_dict(a) for a in applications]
    if diff is not None:
        doc["override_diff"] = diff_to_dict(diff)
    return doc


def write_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
