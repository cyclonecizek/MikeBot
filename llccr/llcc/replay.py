"""Replay harness.

Loads a scenario (a thermal profile, an event log, and a sequence of frames)
and evaluates every frame. Because the evaluator is a pure function of
(snapshot, t), replaying an archived case is just feeding it frames in order.
This is the only real validation available: take an afternoon the 45th
Weather Squadron scrubbed, replay it, and check both the verdict and the
reason.

Events are filtered to those that have occurred by the frame time, so a
scenario file holds the whole timeline and each frame sees only its past.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

from .evaluate import Verdict, evaluate
from .overrides import Override, apply_overrides, diff_verdicts
from .world import (CloudObject, Event, Series, ThermalProfile, VehicleConfig,
                    WorldSnapshot)


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text)


@dataclass
class Frame:
    time: datetime
    overrides: dict[str, dict]
    connections: list[tuple[str, str]]


@dataclass
class Scenario:
    name: str
    pad: str
    azimuth_deg: float
    profile: ThermalProfile
    base_objects: dict[str, dict]
    frames: list[Frame]
    events: list[Event]
    series: dict[str, dict[str, Series]]
    field_mills_available: bool = False
    vehicle: VehicleConfig | None = None
    overrides: list = field(default_factory=list)

    def snapshot(self, frame: Frame) -> WorldSnapshot:
        objects: dict[str, CloudObject] = {}
        for oid, base in self.base_objects.items():
            spec = dict(base)
            spec.update(frame.overrides.get(oid, {}))
            if spec.pop("absent", False):
                continue
            spec.pop("first_seen", None)
            obj = CloudObject(id=oid, **spec)
            obj.series = self.series.get(oid, {})
            objects[oid] = obj
        events = [e for e in self.events if e.time <= frame.time]
        return WorldSnapshot(
            time=frame.time,
            objects=objects,
            events=events,
            profile=self.profile,
            connections=frame.connections,
            field_mills_available=self.field_mills_available,
            vehicle=self.vehicle,
        )

    def run(self, assume_manifest_complete: bool = False,
            with_overrides: bool = True) -> list[Verdict]:
        out = []
        for f in self.frames:
            snap = self.snapshot(f)
            if with_overrides and self.overrides:
                snap, _ = apply_overrides(snap, self.overrides, f.time)
            out.append(evaluate(snap, f.time,
                                assume_manifest_complete=assume_manifest_complete))
        return out

    def run_pair(self, frame, assume_manifest_complete: bool = False):
        """Baseline and overridden verdicts for one frame, plus the diff."""
        snap = self.snapshot(frame)
        baseline = evaluate(snap, frame.time,
                            assume_manifest_complete=assume_manifest_complete)
        if not self.overrides:
            return snap, baseline, baseline, [], diff_verdicts(baseline, baseline, [])
        snap2, applications = apply_overrides(snap, self.overrides, frame.time)
        overridden = evaluate(snap2, frame.time,
                              assume_manifest_complete=assume_manifest_complete)
        diff = diff_verdicts(baseline, overridden, self.overrides)
        return snap2, baseline, overridden, applications, diff


def load_scenario(path: str | Path) -> Scenario:
    raw = json.loads(Path(path).read_text())

    profile = ThermalProfile(
        levels=[(float(z), float(t)) for z, t in raw["profile"]["levels"]],
        uncertainty_m=float(raw["profile"].get("uncertainty_m", 300.0)),
    )

    events = [
        Event(
            kind=e["kind"],
            time=_dt(e["time"]),
            object_id=e["object_id"],
            source=e.get("source", ""),
            detail=e.get("detail", {}),
        )
        for e in raw.get("events", [])
    ]

    series: dict[str, dict[str, Series]] = {}
    for oid, named in raw.get("series", {}).items():
        series[oid] = {
            name: Series(
                samples=[(_dt(t), float(v)) for t, v in spec["samples"]],
                max_gap=timedelta(seconds=spec.get("max_gap_s", 120)),
            )
            for name, spec in named.items()
        }

    frames = [
        Frame(
            time=_dt(f["time"]),
            overrides=f.get("objects", {}),
            connections=[tuple(pair) for pair in f.get("connections", [])],
        )
        for f in raw["frames"]
    ]

    return Scenario(
        name=raw["name"],
        pad=raw.get("pad", ""),
        azimuth_deg=float(raw.get("azimuth_deg", 0.0)),
        profile=profile,
        base_objects=raw.get("objects", {}),
        frames=sorted(frames, key=lambda f: f.time),
        events=sorted(events, key=lambda e: e.time),
        series=series,
        field_mills_available=raw.get("field_mills_available", False),
        vehicle=VehicleConfig(**raw["vehicle"]) if raw.get("vehicle") else None,
        overrides=[
            Override(
                object_id=o["object_id"], cloud_type=o["cloud_type"],
                issued_at=_dt(o["issued_at"]), issued_by=o["issued_by"],
                original_type=o.get("original_type", "unclassified"),
                justification=o.get("justification", ""),
                facts=o.get("facts", {}),
                ttl_seconds=float(o.get("ttl_seconds", 3600.0)),
                affirmed_at=_dt(o["affirmed_at"]) if o.get("affirmed_at") else None,
            )
            for o in raw.get("overrides", [])
        ],
    )


def expectations_report(scenario: Scenario, raw_path: str | Path) -> list[str]:
    """Compare each frame's blocking requirement ids against the scenario's
    declared expectations. This is what turns a replay into a regression test."""
    raw = json.loads(Path(raw_path).read_text())
    verdicts = scenario.run(assume_manifest_complete=True)
    lines: list[str] = []
    for frame, verdict in zip(scenario.frames, verdicts):
        expected = raw.get("expect", {}).get(frame.time.isoformat())
        if expected is None:
            continue
        actual = sorted({r.requirement_id for r in verdict.blocking})
        ok = actual == sorted(expected)
        lines.append(f"  {'pass' if ok else 'FAIL'}  {frame.time:%H:%M:%S}Z  "
                     f"expected={sorted(expected)} actual={actual}")
    return lines
