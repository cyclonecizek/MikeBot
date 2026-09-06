"""Geometry engine.

Produces the numbers the evaluator consumes. Three ideas carry the weight:

  The corridor does not move during a count, so distance is precomputed as a
  field over the grid and every query is a lookup plus a reduction over an
  object's voxels, not a per-object geometric solve.

  Two distinct fields, because the standard uses two distinct metrics and
  mixing them is an easy and dangerous bug. LLCCR 12 uses both in adjacent
  clauses: 12a is slant distance to the anvil, 12b is horizontal distance
  for the MRR test. `D_horiz` also serves lightning, since GLM has no
  altitude and LLCCR 34b collapses slant to horizontal between projections.

  MRR is a max filter, not a loop. Section 4.2.3a bounds the volume below by
  the 0 C level and above by 20 km MSL regardless of the evaluation point's
  altitude, so MRR is a 2D field: one slab maximum down the column, then a
  separable box maximum in x and y. Three linear passes give MRR everywhere
  at once.

Requires numpy and scipy. The evaluator itself does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

NM_TO_M = 1852.0
MRR_HALF_M = 5556.0        # 3 nmi, section 4.2.3a
MRR_TOP_M = 20_000.0       # 20 km MSL
MRR_FALLBACK_M = 7408.0    # 4 nmi, section 4.2.3c
CORE_DBZ = 35.0            # section 4.2.3d(1)
STANDOFF_M = 10.0 * NM_TO_M


@dataclass
class Grid:
    """Local ENU grid centred on the pad.

    Over the ranges these criteria use, flat-earth error is centimetres. The
    radar gridding must share this origin and datum exactly.
    """

    nx: int
    ny: int
    nz: int
    dx: float = 500.0
    dy: float = 500.0
    dz: float = 250.0
    z0: float = 0.0
    origin_lat: float = 28.6083      # LC-39A
    origin_lon: float = -80.6041

    @property
    def sampling(self) -> tuple[float, float, float]:
        return (self.dx, self.dy, self.dz)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    def east(self) -> np.ndarray:
        return (np.arange(self.nx) - self.nx / 2 + 0.5) * self.dx

    def north(self) -> np.ndarray:
        return (np.arange(self.ny) - self.ny / 2 + 0.5) * self.dy

    def up(self) -> np.ndarray:
        return self.z0 + (np.arange(self.nz) + 0.5) * self.dz

    def index_of(self, east: float, north: float, up: float) -> tuple[int, int, int]:
        i = int(round(east / self.dx + self.nx / 2 - 0.5))
        j = int(round(north / self.dy + self.ny / 2 - 0.5))
        k = int(round((up - self.z0) / self.dz - 0.5))
        return i, j, k

    def level_at(self, altitude_m: float) -> int:
        """Lowest level index at or above `altitude_m`, clipped to the grid."""
        k = int(np.ceil((altitude_m - self.z0) / self.dz - 0.5))
        return int(np.clip(k, 0, self.nz - 1))


@dataclass
class Corridor:
    """The flight path volume: normal plus malfunction envelope."""

    grid: Grid
    mask: np.ndarray             # bool (nx, ny, nz)
    d_slant: np.ndarray          # float (nx, ny, nz), metres
    d_horiz: np.ndarray          # float (nx, ny), metres
    ground_mask: np.ndarray      # bool (nx, ny)
    velocity_at: dict = field(default_factory=dict)


def build_corridor(grid: Grid, trajectory, azimuth_deg: float,
                   radius_m: float = 1500.0,
                   max_altitude_m: float = 37_000.0) -> Corridor:
    """Rasterise the corridor and precompute both distance fields.

    `trajectory` is a sequence of (downrange_nmi, altitude_km, velocity_ms).
    LLCCR 34c relieves the standoff above 37 km, which bounds the volume and
    keeps the transform cheap.
    """
    az = np.radians(azimuth_deg)
    centre = np.zeros(grid.shape, dtype=bool)

    pts = [(d * NM_TO_M, a * 1000.0, v) for d, a, v in trajectory]
    velocity_at: dict[int, float] = {}

    for (d0, a0, v0), (d1, a1, v1) in zip(pts, pts[1:]):
        span = max(abs(d1 - d0), abs(a1 - a0))
        steps = max(2, int(span / (min(grid.dx, grid.dz) * 0.5)))
        for t in np.linspace(0.0, 1.0, steps):
            down = d0 + t * (d1 - d0)
            alt = a0 + t * (a1 - a0)
            if alt > max_altitude_m:
                continue
            i, j, k = grid.index_of(down * np.sin(az), down * np.cos(az), alt)
            if 0 <= i < grid.nx and 0 <= j < grid.ny and 0 <= k < grid.nz:
                centre[i, j, k] = True
                velocity_at[k] = v0 + t * (v1 - v0)

    if not centre.any():
        raise ValueError("corridor centreline falls entirely outside the grid")

    # Dilate the centreline to the envelope radius, then take the distance
    # transform of its complement.
    centre_dist = ndimage.distance_transform_edt(~centre, sampling=grid.sampling)
    mask = centre_dist <= radius_m
    d_slant = ndimage.distance_transform_edt(~mask, sampling=grid.sampling)

    ground_mask = mask.any(axis=2)
    d_horiz = ndimage.distance_transform_edt(
        ~ground_mask, sampling=(grid.dx, grid.dy))

    return Corridor(grid=grid, mask=mask, d_slant=d_slant, d_horiz=d_horiz,
                    ground_mask=ground_mask, velocity_at=velocity_at)


# --------------------------------------------------------------------------
# MRR, section 4.2.3
# --------------------------------------------------------------------------

def mrr_field(refl: np.ndarray, grid: Grid, freezing_level_m: float,
              half_m: float = MRR_HALF_M) -> np.ndarray:
    """MRR at every horizontal grid point, as a 2D field.

    The specified volume runs from the 0 C level to 20 km MSL and extends
    3 nmi in each cardinal direction from the evaluation point. Its vertical
    bounds do not depend on the evaluation point, so this is 2D: a slab
    maximum down the column followed by a separable box maximum.
    """
    k0 = grid.level_at(freezing_level_m)
    k1 = grid.level_at(MRR_TOP_M)
    if k1 <= k0:
        return np.full((grid.nx, grid.ny), -np.inf)

    slab = np.nanmax(refl[:, :, k0:k1 + 1], axis=2)
    slab = np.where(np.isfinite(slab), slab, -np.inf)

    # Round the half-width up. Under-covering the specified volume would
    # shrink MRR, which makes an "MRR < +7.5 dBZ" exception easier to
    # satisfy -- a relaxation, and the wrong direction to err in.
    wx = 2 * int(np.ceil(half_m / grid.dx)) + 1
    wy = 2 * int(np.ceil(half_m / grid.dy)) + 1
    out = ndimage.maximum_filter1d(slab, size=wx, axis=0, mode="nearest")
    out = ndimage.maximum_filter1d(out, size=wy, axis=1, mode="nearest")
    return out


def mrr_validity(refl: np.ndarray, grid: Grid, freezing_level_m: float,
                 lightning_en: list[tuple[float, float]] | None = None,
                 standoff_m: float = STANDOFF_M) -> np.ndarray:
    """Section 4.2.3d as a 2D mask: True where an MRR evaluation point is usable.

    Invalid within 10 nmi of any 35 dBZ or greater reflectivity at or above
    the 0 C level, or within 10 nmi of any lightning in the previous five
    minutes. Applied as a mask rather than a check, so an exception whose
    evaluation points are invalid is simply unavailable.
    """
    k0 = grid.level_at(freezing_level_m)
    cores = np.nanmax(refl[:, :, k0:], axis=2) >= CORE_DBZ

    seed = cores.copy()
    for east, north in (lightning_en or []):
        i, j, _ = grid.index_of(east, north, grid.z0)
        if 0 <= i < grid.nx and 0 <= j < grid.ny:
            seed[i, j] = True

    if not seed.any():
        return np.ones((grid.nx, grid.ny), dtype=bool)

    dist = ndimage.distance_transform_edt(~seed, sampling=(grid.dx, grid.dy))
    return dist > standoff_m


def mrr_fallback(refl: np.ndarray, grid: Grid) -> np.ndarray:
    """Section 4.2.3c: when the volume maximum cannot be determined, the
    largest composite reflectivity within 4 nmi horizontally."""
    comp = np.nanmax(refl, axis=2)
    comp = np.where(np.isfinite(comp), comp, -np.inf)
    wx = 2 * int(np.ceil(MRR_FALLBACK_M / grid.dx)) + 1
    wy = 2 * int(np.ceil(MRR_FALLBACK_M / grid.dy)) + 1
    out = ndimage.maximum_filter1d(comp, size=wx, axis=0, mode="nearest")
    return ndimage.maximum_filter1d(out, size=wy, axis=1, mode="nearest")


# --------------------------------------------------------------------------
# Primitives. Every requirement leaf resolves to one of these.
# --------------------------------------------------------------------------

def slant_min(corridor: Corridor, voxels: np.ndarray) -> float | None:
    """Minimum slant distance from the corridor to an object, in nmi."""
    if not voxels.any():
        return None
    return float(corridor.d_slant[voxels].min() / NM_TO_M)


def horiz_min(corridor: Corridor, voxels: np.ndarray) -> float | None:
    """Minimum horizontal distance, between vertical projections, in nmi."""
    if not voxels.any():
        return None
    return float(corridor.d_horiz[voxels.any(axis=2)].min() / NM_TO_M)


def intersects_corridor(corridor: Corridor, voxels: np.ndarray) -> bool:
    """Through-cloud test. Grid discretisation makes a grazing contact
    ambiguous; through-cloud rules are uniformly stricter than the 0-3 nmi
    tier, so tie-breaking into intersection is the conservative side."""
    return bool((voxels & corridor.mask).any())


def voxels_within(corridor: Corridor, voxels: np.ndarray,
                  nmi_: float) -> np.ndarray:
    return voxels & (corridor.d_slant <= nmi_ * NM_TO_M)


def all_colder_than(grid: Grid, voxels: np.ndarray,
                    isotherm_m: float | None) -> bool | None:
    """'located entirely at altitudes where the temperature is colder than T'.

    Appears in LLCCR 12, 13, 14, 15, 16, 17, 19 and 20. Empty selection is
    vacuously true: no part of the cloud is in the region, so no part
    violates it.
    """
    if isotherm_m is None:
        return None
    if not voxels.any():
        return True
    k = grid.level_at(isotherm_m)
    return not bool(voxels[:, :, :k].any())


def max_reflectivity_within(corridor: Corridor, refl: np.ndarray,
                            voxels: np.ndarray, nmi_: float) -> float | None:
    sel = voxels_within(corridor, voxels, nmi_)
    if not sel.any():
        return None
    return float(np.nanmax(refl[sel]))


def mrr_max_within_horiz(corridor: Corridor, mrr: np.ndarray,
                         valid: np.ndarray, nmi_: float
                         ) -> tuple[float | None, bool]:
    """Largest MRR at any point within `nmi_` horizontally of the corridor,
    with a flag for whether every such point was a valid evaluation point."""
    sel = corridor.d_horiz <= nmi_ * NM_TO_M
    if not sel.any():
        return None, False
    return float(mrr[sel].max()), bool(valid[sel].all())


def mrr_max_in_corridor(corridor: Corridor, mrr: np.ndarray,
                        valid: np.ndarray) -> tuple[float | None, bool]:
    sel = corridor.ground_mask
    if not sel.any():
        return None, False
    return float(mrr[sel].max()), bool(valid[sel].all())


def layer_thickness(grid: Grid, voxels: np.ndarray) -> float | None:
    """Vertical extent from the base of the bottom layer to the top of the
    uppermost, per the thick cloud layer definition."""
    if not voxels.any():
        return None
    levels = np.flatnonzero(voxels.any(axis=(0, 1)))
    return float((levels[-1] - levels[0] + 1) * grid.dz)


def corridor_penetrates_colder_than(grid: Grid, corridor: Corridor,
                                    cloud: np.ndarray, isotherm_m: float | None
                                    ) -> tuple[bool | None, float | None]:
    """LLCCR 27: does the path pass through cloud colder than -10 C, and at
    what speed. Returns (penetrates, slowest velocity at a penetrated level)."""
    if isotherm_m is None:
        return None, None
    k = grid.level_at(isotherm_m)
    hit = cloud[:, :, k:] & corridor.mask[:, :, k:]
    if not hit.any():
        return False, None
    levels = np.flatnonzero(hit.any(axis=(0, 1))) + k
    speeds = [corridor.velocity_at[int(z)] for z in levels
              if int(z) in corridor.velocity_at]
    return True, (min(speeds) if speeds else None)
