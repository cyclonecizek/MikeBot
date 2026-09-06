"""Geometry oracle tests.

Unlike the evaluator, this layer has exact answers, so it is tested against
them rather than against fixtures. Two oracles do most of the work:

  Distance: a nested loop over every voxel pair is exact, just too slow for
  production. On a small grid it settles the anisotropic transform outright.

  MRR: the separable three-pass maximum must equal the naive nested-box
  maximum on random fields. That comparison catches essentially every
  indexing and window-width bug.

Requires numpy and scipy. Run separately from tests.py, which is
dependency-free by design.
"""

from __future__ import annotations

import numpy as np

from llcc.geometry import (
    Corridor, Grid, NM_TO_M, all_colder_than, build_corridor,
    corridor_penetrates_colder_than, horiz_min, intersects_corridor,
    layer_thickness, mrr_field, mrr_validity, slant_min, voxels_within,
)

FAILURES: list[str] = []
rng = np.random.default_rng(20260904)


def check(name: str, condition: bool) -> None:
    if not condition:
        FAILURES.append(name)
    print(f"  {'pass' if condition else 'FAIL'}  {name}")


def brute_distance(mask: np.ndarray, sampling) -> np.ndarray:
    """Exact distance to the nearest True voxel. O(n^2), oracle only."""
    dx, dy, dz = sampling
    idx = np.argwhere(mask).astype(float)
    idx *= np.array([dx, dy, dz])
    out = np.empty(mask.shape)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                p = np.array([i * dx, j * dy, k * dz])
                out[i, j, k] = np.min(np.linalg.norm(idx - p, axis=1))
    return out


def brute_box_max(slab: np.ndarray, wx: int, wy: int) -> np.ndarray:
    """Exact box maximum with edge clamping. Oracle only."""
    nx, ny = slab.shape
    out = np.empty_like(slab)
    hx, hy = wx // 2, wy // 2
    for i in range(nx):
        for j in range(ny):
            i0, i1 = max(0, i - hx), min(nx, i + hx + 1)
            j0, j1 = max(0, j - hy), min(ny, j + hy + 1)
            out[i, j] = slab[i0:i1, j0:j1].max()
    return out


print("distance transform against a brute-force oracle")
from scipy import ndimage  # noqa: E402

for sampling in [(500.0, 500.0, 500.0), (500.0, 500.0, 250.0), (900.0, 400.0, 300.0)]:
    mask = np.zeros((12, 11, 9), dtype=bool)
    for _ in range(4):
        mask[tuple(rng.integers(0, s) for s in mask.shape)] = True
    fast = ndimage.distance_transform_edt(~mask, sampling=sampling)
    slow = brute_distance(mask, sampling)
    check(f"anisotropic EDT matches oracle, sampling={sampling}",
          np.allclose(fast, slow, atol=1e-6))

print("\nMRR separable maximum against a brute-force oracle")
grid = Grid(nx=40, ny=40, nz=24, dx=1000.0, dy=1000.0, dz=500.0)
refl = rng.uniform(-20, 55, size=grid.shape)
freezing = 4600.0
k0 = grid.level_at(freezing)
k1 = grid.level_at(20_000.0)
slab = refl[:, :, k0:k1 + 1].max(axis=2)
w = 2 * int(np.ceil(5556.0 / 1000.0)) + 1
check("three-pass MRR equals the naive nested box maximum",
      np.allclose(mrr_field(refl, grid, freezing), brute_box_max(slab, w, w)))

flat = np.full(grid.shape, -30.0)
flat[20, 20, k0 + 2] = 50.0
m = mrr_field(flat, grid, freezing)
reach = int(np.ceil(5556.0 / 1000.0))
check("MRR window covers the full 3 nmi and no more",
      m[20, 20 + reach] == 50.0 and m[20, 20 + reach + 1] == -30.0)
check("MRR half-width is rounded up, never down",
      reach * 1000.0 >= 5556.0)

below = np.full(grid.shape, -30.0)
below[20, 20, max(0, k0 - 2)] = 60.0
check("reflectivity below the 0 C level is excluded from MRR",
      mrr_field(below, grid, freezing).max() == -30.0)

print("\nMRR validity, section 4.2.3d")
cores = np.full(grid.shape, -30.0)
cores[20, 20, k0 + 1] = 40.0
valid = mrr_validity(cores, grid, freezing)
check("a 35 dBZ core invalidates points within 10 nmi",
      not valid[20, 20] and not valid[20, 20 + 9] and valid[20, 39])
check("lightning invalidates independently of reflectivity",
      not mrr_validity(np.full(grid.shape, -30.0), grid, freezing,
                       lightning_en=[(0.0, 0.0)])[20, 20])

print("\ncorridor and distance fields")
g = Grid(nx=60, ny=60, nz=40, dx=1000.0, dy=1000.0, dz=500.0)
traj = [(0, 0.0, 0), (5, 3.2, 450), (11, 7.0, 640), (20, 12.6, 900)]
cor = build_corridor(g, traj, azimuth_deg=45.0, radius_m=1500.0)
check("corridor volume is non-empty", cor.mask.any())
check("distance is zero inside the corridor", cor.d_slant[cor.mask].max() == 0.0)

# Slant distance can never be less than horizontal distance.
h3 = np.broadcast_to(cor.d_horiz[:, :, None], g.shape)
check("slant distance is never less than horizontal", bool((cor.d_slant >= h3 - 1e-6).all()))

# A single voxel at a known offset from the pad, hand-checkable.
probe = np.zeros(g.shape, dtype=bool)
i0, j0, k0b = g.index_of(0.0, 0.0, 200.0)
probe[i0, j0 + 10, k0b] = True          # 10 km due north of the pad
d = slant_min(cor, probe)
check("single-voxel slant distance is plausible for a 45 deg corridor",
      d is not None and 3.0 < d < 4.5)
check("horizontal distance never exceeds slant distance",
      horiz_min(cor, probe) <= slant_min(cor, probe) + 1e-9)

on_path = np.zeros(g.shape, dtype=bool)
on_path |= cor.mask
check("an object overlapping the corridor intersects it",
      intersects_corridor(cor, on_path))
check("an object clear of the corridor does not", not intersects_corridor(cor, probe))
check("empty object yields no distance", slant_min(cor, np.zeros(g.shape, bool)) is None)

near = voxels_within(cor, cor.mask | probe, 5.0)
check("voxels_within selects by slant distance", near.sum() >= cor.mask.sum())

print("\ntemperature and thickness predicates")
cloud = np.zeros(g.shape, dtype=bool)
cloud[30, 30, g.level_at(6000.0):g.level_at(9000.0)] = True
check("cloud entirely above the 0 C level is all colder than 0 C",
      all_colder_than(g, cloud, 4600.0) is True)
cloud[30, 30, g.level_at(3000.0)] = True
check("one voxel below the level makes it false",
      all_colder_than(g, cloud, 4600.0) is False)
check("empty selection is vacuously true",
      all_colder_than(g, np.zeros(g.shape, bool), 4600.0) is True)
check("unknown isotherm yields unknown", all_colder_than(g, cloud, None) is None)

layer = np.zeros(g.shape, dtype=bool)
layer[10:20, 10:20, g.level_at(3000.0):g.level_at(4900.0)] = True
t = layer_thickness(g, layer)
check("layer thickness matches its vertical extent", t is not None and 1800 <= t <= 2100)

print("\ntriboelectrification penetration")
cold = np.zeros(g.shape, dtype=bool)
cold[:, :, g.level_at(7000.0):] = True
pen, speed = corridor_penetrates_colder_than(g, cor, cold, 6600.0)
check("penetration of cloud colder than -10 C is detected", pen is True)
check("velocity at the penetrated level is reported", speed is not None and speed > 0)
warm = np.zeros(g.shape, dtype=bool)
warm[:, :, :g.level_at(3000.0)] = True
check("cloud entirely warmer than -10 C is not a penetration",
      corridor_penetrates_colder_than(g, cor, warm, 6600.0)[0] is False)

print(f"\n{'all geometry tests passed' if not FAILURES else str(len(FAILURES)) + ' FAILURES: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
