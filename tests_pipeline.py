"""End-to-end pipeline tests.

A synthetic sequence goes in as reflectivity; verdicts come out. This is the
test that proves the layers fit together -- each one is checked in isolation
elsewhere, but only here does a scan actually become a verdict.

Requires numpy, scipy and scikit-image.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from llcc.evaluate import State, evaluate
from llcc.geometry import Grid, build_corridor
from llcc.pipeline import Pipeline, PipelineConfig
from llcc.serialize import snapshot_to_dict, verdict_to_dict
from llcc.world import CloudType, EventKind, ThermalProfile, VehicleConfig

FAILURES: list[str] = []
GRID = Grid(nx=64, ny=64, nz=32, dx=1000.0, dy=1000.0, dz=500.0)
T0 = datetime(2026, 9, 4, 21, 0, 0)
SCAN = timedelta(minutes=4, seconds=30)

PROFILE = ThermalProfile(levels=[
    (0, 29.0), (1000, 22.0), (2000, 15.5), (3000, 9.0), (3600, 5.0),
    (4600, 0.0), (5600, -5.0), (6600, -10.0), (7600, -15.0), (8700, -20.0),
    (10000, -33.0), (12000, -52.0), (14000, -68.0), (16000, -80.0),
])
VEHICLE = VehicleConfig(name="ref", triboelectric_exemption="4.1.10.2b",
                        triboelectric_basis="ESD analysis")


def check(name: str, condition: bool) -> None:
    if not condition:
        FAILURES.append(name)
    print(f"  {'pass' if condition else 'FAIL'}  {name}")


def blob(f, ci, cj, peak, radius=6.0, base=2, top=18):
    i, j, k = np.ogrid[:GRID.nx, :GRID.ny, :GRID.nz]
    mid, span = (base + top) / 2, max(1.0, (top - base) / 2)
    d = np.sqrt(((i - ci) / radius) ** 2 + ((j - cj) / radius) ** 2
                + ((k - mid) / span) ** 2)
    np.maximum(f, peak * np.exp(-1.4 * d * d) - 2.0, out=f)
    return f


def scene(centres):
    f = np.full(GRID.shape, -30.0)
    for spec in centres:
        f = blob(f, *spec)
    return f


CORRIDOR = build_corridor(
    GRID, [(0, 0.0, 0), (5, 3.2, 450), (11, 7.0, 640), (20, 12.6, 900)],
    azimuth_deg=45.0, radius_m=1500.0)


def fresh(classify=False, mills=False) -> Pipeline:
    return Pipeline(PipelineConfig(
        grid=GRID, corridor=CORRIDOR, profile=PROFILE, vehicle=VEHICLE,
        field_mills_available=mills, classify=classify))


print("a scan becomes a snapshot")
pipe = fresh()
snap = pipe.ingest(scene([(40, 40, 48)]), T0)
check("snapshot carries objects plus the domain pseudo-object",
      len(snap.objects) == 2 and "DOMAIN" in snap.objects)
obj = [o for k, o in snap.objects.items() if k != "DOMAIN"][0]
check("distances are populated from the geometry engine",
      obj.slant_min_nmi is not None and obj.horiz_min_nmi is not None)
check("horizontal distance never exceeds slant distance",
      obj.horiz_min_nmi <= obj.slant_min_nmi + 1e-6)
check("top temperature is derived from the thermal profile",
      obj.top_temp_c is not None and obj.top_temp_c < 29.0)
check("display geometry is populated for the renderer",
      obj.range_nmi is not None and obj.top_alt_km is not None
      and obj.bearing_deg is not None)
check("worst-case classifier labels everything cumulus",
      obj.cloud_type == CloudType.CUMULUS)

print("\nthe snapshot evaluates")
verdict = evaluate(snap, T0, assume_manifest_complete=True)
check("evaluation produces results", len(verdict.results) > 0)
check("a verdict state is reached", verdict.state in set(State))
check("the tribo exemption is honoured end to end",
      all(r.state is State.GO for r in verdict.results
          if r.requirement_id == "LLCCR 27"))

print("\nthe snapshot serialises against the display contract")
doc = snapshot_to_dict(snap, pad="LC-39A", azimuth_deg=45.0)
vdoc = verdict_to_dict(verdict)
for key in ("time", "objects", "profile", "events", "connections"):
    check(f"snapshot document has {key}", key in doc)
check("isotherm altitudes are exported", doc["profile"]["isotherms"]["0"] is not None)
check("verdict document has per-requirement results", len(vdoc["results"]) > 0)

print("\nproximity drives the verdict")
near = fresh().ingest(scene([(34, 34, 48)]), T0)
far = fresh().ingest(scene([(4, 60, 48)]), T0)
n_near = [o for k, o in near.objects.items() if k != "DOMAIN"][0].slant_min_nmi
n_far = [o for k, o in far.objects.items() if k != "DOMAIN"][0].slant_min_nmi
check("a cell near the corridor is closer than one far from it", n_near < n_far)
blocking_near = len(evaluate(near, T0, assume_manifest_complete=True).blocking)
blocking_far = len(evaluate(far, T0, assume_manifest_complete=True).blocking)
check("the nearer cell blocks at least as much", blocking_near >= blocking_far)

print("\nlightning flows through to the clocks")
pipe = fresh()
snap = pipe.ingest(scene([(40, 40, 48)]), T0, strikes=[(0.0, 8000.0)])
flashes = [e for e in snap.events if e.kind == EventKind.LIGHTNING]
check("a strike is recorded as an event", len(flashes) == 1)
check("the strike carries its distance to the flight path",
      flashes[0].detail.get("distance_nmi") is not None)
check("the strike is attributed to an object",
      flashes[0].object_id != "unattributed")
check("the flashed object becomes a thunderstorm",
      snap.objects[flashes[0].object_id].is_thunderstorm is True)
v = evaluate(snap, T0, assume_manifest_complete=True)
held = [r for r in v.blocking if r.requirement_id == "LLCCR 5"]
check("LLCCR 5 holds after a strike within 10 nmi",
      bool(held) and held[0].release == T0 + timedelta(minutes=30))
later = evaluate(snap, T0 + timedelta(minutes=31), assume_manifest_complete=True)
check("the hold clears once 30 minutes pass",
      all(r.requirement_id != "LLCCR 5" for r in later.blocking))

print("\na split reaches the event log as a detachment")
pipe = fresh()
seq = [scene([(20, 20, 48)]),
       scene([(20, 20, 48)]),
       scene([(16, 20, 48), (26, 20, 40)])]
snaps = [pipe.ingest(f, T0 + n * SCAN) for n, f in enumerate(seq)]
detach = [e for e in snaps[-1].events if e.kind == EventKind.DETACHMENT]
check("the tracker's split becomes a detachment event", len(detach) >= 1)
check("the detachment is timestamped at the frame it occurred",
      bool(detach) and detach[0].time == T0 + 2 * SCAN)
child = snaps[-1].objects.get(detach[0].object_id)
check("the detached child records its parent in lineage",
      child is not None and bool(child.parent_ids))

print("\nunobservable volume is not treated as clear")
obs = np.ones(GRID.shape, dtype=bool)
obs[:, 40:, :] = False
hidden = fresh().ingest(scene([(20, 55, 48)]), T0, observable=obs)
check("echo inside a blind region yields no objects",
      len([k for k in hidden.objects if k != "DOMAIN"]) == 0)
seen = fresh().ingest(scene([(20, 55, 48)]), T0)
check("the same echo yields an object when observable",
      len([k for k in seen.objects if k != "DOMAIN"]) == 1)

print("\nrepeated ingest is stable")
pipe = fresh()
ids = []
for n in range(4):
    s = pipe.ingest(scene([(40, 40, 48)]), T0 + n * SCAN)
    ids.append(sorted(k for k in s.objects if k != "DOMAIN"))
check("a stationary cell keeps one identity across scans",
      all(x == ids[0] for x in ids) and len(ids[0]) == 1)
check("the event log does not grow without cause",
      len([e for e in pipe.events if e.kind == EventKind.LIGHTNING]) == 0)

print(f"\n{'all pipeline tests passed' if not FAILURES else str(len(FAILURES)) + ' FAILURES: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
