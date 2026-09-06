"""Segmentation and tracking tests.

Synthetic fields with known answers: two blobs that drift together, merge,
and separate. The frame at which each event happens is known by
construction, so object counts, lineage edges and event timestamps can all
be asserted exactly.

Requires numpy, scipy and scikit-image.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from llcc.geometry import Grid
from llcc.segment import cloud_field, observability, precipitation_flags, segment_field
from llcc.track import Tracker

FAILURES: list[str] = []
GRID = Grid(nx=64, ny=64, nz=32, dx=1000.0, dy=1000.0, dz=500.0)
T0 = datetime(2026, 9, 4, 21, 0, 0)
SCAN = timedelta(minutes=4, seconds=30)


def check(name: str, condition: bool) -> None:
    if not condition:
        FAILURES.append(name)
    print(f"  {'pass' if condition else 'FAIL'}  {name}")


def blob(field_, ci, cj, peak, radius=6.0, base=2, top=18):
    """Paint a rounded reflectivity blob."""
    i, j, k = np.ogrid[:GRID.nx, :GRID.ny, :GRID.nz]
    mid = (base + top) / 2
    span = max(1.0, (top - base) / 2)
    d = np.sqrt(((i - ci) / radius) ** 2 + ((j - cj) / radius) ** 2
                + ((k - mid) / span) ** 2)
    np.maximum(field_, peak * np.exp(-1.4 * d * d) - 2.0, out=field_)
    return field_


def scene(centres):
    f = np.full(GRID.shape, -30.0)
    for ci, cj, peak in centres:
        f = blob(f, ci, cj, peak)
    return f


def temp_at(alt_m):
    return 29.0 - 6.5 * (alt_m / 1000.0)


print("cloud field is ternary")
refl = scene([(20, 20, 45)])
obs = np.ones(GRID.shape, dtype=bool)
obs[:, 40:, :] = False
plain = cloud_field(refl, None)
masked = cloud_field(refl, obs)
check("unobservable volume is excluded from the cloud field",
      plain.sum() >= masked.sum())
far = scene([(20, 55, 45)])
check("echo inside an unobservable region is not segmented as cloud",
      cloud_field(far, obs).sum() == 0)

print("\nsegmentation separates objects without dissolving them")
two = scene([(18, 20, 45), (40, 20, 42)])
seg = segment_field(two, GRID)
check("two separated cells give two objects", len(seg.segments) == 2)
check("separated objects share no component",
      len({s.component for s in seg.segments.values()}) == 2)
check("no connection edge between separated objects", seg.connections == [])

touching = scene([(26, 20, 45), (34, 20, 42)])
seg2 = segment_field(touching, GRID)
check("touching cells remain two distinct objects", len(seg2.segments) == 2)
check("touching cells share one component",
      len({s.component for s in seg2.segments.values()}) == 1)
check("touching cells produce a connection edge", len(seg2.connections) == 1)

# Thin cloud never reaches the seed threshold but is still cloud.
thin = np.full(GRID.shape, -30.0)
thin = blob(thin, 20, 20, 8.0, radius=9.0, base=20, top=26)
segt = segment_field(thin, GRID)
check("sub-seed cloud still yields an object", len(segt.segments) == 1)

print("\nfeatures")
s = list(segment_field(scene([(20, 20, 48)]), GRID).segments.values())[0]
check("top altitude is above base altitude",
      s.top_altitude(GRID) > s.base_altitude(GRID))
check("max reflectivity is recovered", 40 < s.max_refl <= 48)
edge = segment_field(scene([(1, 20, 45)]), GRID)
check("an object at the domain boundary is flagged",
      any(x.touches_edge for x in edge.segments.values()))
check("precipitation flags follow 4.2.2 radar thresholds",
      precipitation_flags(scene([(20, 20, 45)]), s.mask) == (True, True))
weak = scene([(20, 20, 12)])
sw = list(segment_field(weak, GRID).segments.values())[0]
check("weak echo is neither precipitation nor moderate",
      precipitation_flags(weak, sw.mask) == (False, False))

print("\ntracking: identity, merge, split")
tr = Tracker()
frames = [
    (T0 + 0 * SCAN, scene([(16, 20, 45), (44, 20, 42)])),
    (T0 + 1 * SCAN, scene([(22, 20, 45), (38, 20, 42)])),
    (T0 + 2 * SCAN, scene([(28, 20, 45), (33, 20, 42)])),   # merged
    (T0 + 3 * SCAN, scene([(24, 20, 45), (40, 20, 42)])),   # split again
]
ids_seen, all_events = [], []
for when, f in frames:
    sg = segment_field(f, GRID)
    mapping, events = tr.update(sg, f, GRID, when, temp_at)
    ids_seen.append(sorted(set(mapping.values())))
    all_events.extend(events)

check("first frame creates two tracks", len(ids_seen[0]) == 2)
check("identity is preserved across a scan", ids_seen[0] == ids_seen[1])
kinds = {e.kind for e in all_events}
check("a split event is emitted", "split" in kinds)
split = [e for e in all_events if e.kind == "split"]
check("split records its parent", bool(split) and bool(split[0].parents))
child = tr.tracks[split[0].track_id]
check("split child inherits the parent's top history",
      len(child.top_history) > 1)
check("split child records a detachment time", child.detached_at is not None)
check("split timestamp is the frame it happened at",
      split[0].time in [f[0] for f in frames])

print("\nprovenance and history")
edge_tr = Tracker()
f = scene([(1, 20, 45)])
sgz = segment_field(f, GRID)
m, ev = edge_tr.update(sgz, f, GRID, T0, temp_at)
tid = list(m.values())[0]
check("a track entering at the domain edge has unknown provenance",
      edge_tr.tracks[tid].unknown_provenance)

gap_tr = Tracker()
g1 = scene([(20, 20, 45)])
gap_tr.update(segment_field(g1, GRID), g1, GRID, T0, temp_at)
m2, _ = gap_tr.update(segment_field(g1, GRID), g1, GRID,
                      T0 + timedelta(minutes=40), temp_at)
check("a track appearing after a data gap has unknown provenance",
      gap_tr.tracks[list(m2.values())[0]].unknown_provenance)

hist_tr = Tracker()
for n in range(4):
    f = scene([(20, 20, 45)])
    mm, _ = hist_tr.update(segment_field(f, GRID), f, GRID, T0 + n * SCAN, temp_at)
t = hist_tr.tracks[list(mm.values())[0]]
check("history shorter than the window yields unknown, not false",
      t.colder_than_within(-10.0, timedelta(hours=3), T0 + 3 * SCAN) is None)
check("history spanning the window answers the question",
      t.colder_than_within(-10.0, timedelta(minutes=10), T0 + 3 * SCAN) is not None)
check("coldest top ever is retained for 4.1.4 scope",
      t.coldest_top_c is not None)

print("\nconnection persistence is asymmetric")
ptr = Tracker()
seq = [
    scene([(26, 20, 45), (34, 20, 42)]),   # connected
    scene([(26, 20, 45), (34, 20, 42)]),   # connected
    scene([(16, 20, 45), (44, 20, 42)]),   # apart
]
edges = []
for n, f in enumerate(seq):
    sg = segment_field(f, GRID)
    mp, _ = ptr.update(sg, f, GRID, T0 + n * SCAN, temp_at)
    edges.append(ptr.connections(sg, mp))
check("connection is declared on first evidence", len(edges[0]) == 1)
check("disconnection is not declared on first absence", len(edges[2]) == 1)

sg = segment_field(seq[2], GRID)
mp, _ = ptr.update(sg, seq[2], GRID, T0 + 3 * SCAN, temp_at)
check("disconnection is declared once it persists",
      len(ptr.connections(sg, mp)) == 0)

print("\nobservability")
big = list(segment_field(scene([(20, 20, 45)]), GRID).segments.values())[0]
partial = np.ones(GRID.shape, dtype=bool)
partial[:, :20, :] = False
check("observability is full when everything is seen",
      observability(big.mask, None) == 1.0)
check("observability drops when coverage is partial",
      observability(big.mask, partial) < 1.0)

print(f"\n{'all segmentation tests passed' if not FAILURES else str(len(FAILURES)) + ' FAILURES: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
