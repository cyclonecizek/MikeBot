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
from .expr import Trace
from .world import CloudObject, WorldSnapshot

SCHEMA_VERSION = "llcc-1"

# Standoff used to draw each object's exclusion buffer, in nautical miles.
# This is a display concern, not a criterion: it is the outer edge of the
# distance tier the requirement covers, so the operator can see at a glance
# how far the cloud must be to stop mattering.
BUFFER_NMI: dict[str, float] = {
    "LLCCR 5": 10.0,
    "LLCCR 6": 10.0,
    "LLCCR 9": 0.0,
    "LLCCR 10": 5.0,
    "LLCCR 11": 10.0,
    "LLCCR 12": 3.0,
    "LLCCR 13": 5.0,
    "LLCCR 14": 10.0,
    "LLCCR 15": 0.0,
    "LLCCR 16": 3.0,
    "LLCCR 17": 10.0,
    "LLCCR 18": 0.0,
    "LLCCR 19": 0.0,
    "LLCCR 20": 3.0,
    "LLCCR 21": 5.0,
    "LLCCR 22": 5.0,
    "LLCCR 25": 0.0,
    "LLCCR 27": 0.0,
}

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
                     radar=None) -> dict:
    doc = {
        "schema": SCHEMA_VERSION,
        "kind": "snapshot",
        "time": _iso(snapshot.time),
        "pad": pad,
        "azimuth_deg": azimuth_deg,
        "field_mills_available": snapshot.field_mills_available,
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
        "section": result.section,
        "title": result.title,
        "unit_id": result.unit_id,
        "state": result.state.name,
        "release": _iso(result.release),
        "buffer_nmi": BUFFER_NMI.get(result.requirement_id),
        "trigger": trace_to_dict(result.trigger),
        "exception": trace_to_dict(result.exception),
    }


def unit_buffers(verdict: Verdict) -> dict[str, dict]:
    """Per-unit display buffer: the widest standoff any blocking requirement
    implicates, and the worst state driving it."""
    out: dict[str, dict] = {}
    for r in verdict.results:
        if not r.blocking:
            continue
        buf = BUFFER_NMI.get(r.requirement_id, 0.0) or 0.0
        entry = out.setdefault(r.unit_id, {"buffer_nmi": 0.0, "state": "GO",
                                           "requirement_ids": []})
        entry["buffer_nmi"] = max(entry["buffer_nmi"], buf)
        entry["requirement_ids"].append(r.requirement_id)
        order = ["GO", "NO_GO_UNTIL", "INDETERMINATE", "NO_GO"]
        if order.index(r.state.name) > order.index(entry["state"]):
            entry["state"] = r.state.name
    return out


def verdict_to_dict(verdict: Verdict, applications=None, diff=None) -> dict:
    from .overrides import application_to_dict, diff_to_dict

    doc = {
        "schema": SCHEMA_VERSION,
        "kind": "verdict",
        "time": _iso(verdict.time),
        "state": verdict.state.name,
        "earliest_change": _iso(verdict.earliest_change),
        "manifest_gaps": verdict.manifest_gaps,
        "units": unit_buffers(verdict),
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
