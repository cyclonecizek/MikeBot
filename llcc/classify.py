"""Cloud classification.

The taxonomy in NASA-STD-4010B is not a taxonomy of appearance. An anvil is
an anvil because it came from a thunderstorm's outflow; a debris cloud is
debris because its parent decayed. So this reads lineage first and present
shape second. A cold fibrous sheet with no convective ancestry is cirriform;
the same sheet downwind of a cell that reached -30 C is an anvil under 4.1.4
or 4.1.5, with entirely different standoffs.

Every path that cannot reach a confident answer returns CUMULUS, which
carries the widest standoffs in section 4.1. The classifier exists to relax
the worst-case assumption where it can positively identify something, never
to introduce a new one.

Every threshold here is a tuning parameter, not a quantity from the standard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .world import CloudType


@dataclass
class Thresholds:
    """Tunable. None of these appear in NASA-STD-4010B."""

    convective_dbz_aloft: float = 30.0
    convective_aspect: float = 0.25
    stratiform_aspect: float = 0.12
    stratiform_area_km2: float = 250.0
    stratiform_max_dbz_aloft: float = 25.0
    anvil_max_dbz: float = 28.0
    anvil_min_base_temp_c: float = -8.0
    cirrus_max_dbz: float = 12.0
    cirrus_max_temp_c: float = -15.0
    debris_collapse_temp_c: float = -10.0
    anvil_parent_temp_c: float = -10.0
    debris_parent_temp_c: float = -20.0


@dataclass
class Features:
    top_m: float
    base_m: float
    depth_m: float
    area_km2: float
    equiv_diameter_m: float
    aspect: float
    max_dbz: float
    max_dbz_aloft: float
    top_temp_c: float | None
    base_temp_c: float | None
    coldest_top_c: float | None
    bright_band: bool


@dataclass
class Verdict:
    cloud_type: str
    reason: str
    confident: bool = True


def extract(refl, mask, grid, profile, bright_band: bool = False) -> Features:
    idx = np.argwhere(mask)
    levels = idx[:, 2]
    top_m = grid.z0 + (int(levels.max()) + 1) * grid.dz
    base_m = grid.z0 + int(levels.min()) * grid.dz

    footprint = int(mask.any(axis=2).sum())
    area_km2 = footprint * grid.dx * grid.dy / 1e6
    equiv_d = 2.0 * np.sqrt(max(area_km2, 1e-6) * 1e6 / np.pi)
    depth = max(top_m - base_m, grid.dz)

    z0c = profile.isotherm_altitude(0.0)
    aloft = mask.copy()
    if z0c is not None:
        aloft[:, :, :grid.level_at(z0c)] = False

    with np.errstate(invalid="ignore"):
        max_dbz = float(np.nanmax(refl[mask])) if mask.any() else -99.0
        max_aloft = float(np.nanmax(refl[aloft])) if aloft.any() else -99.0

    return Features(
        top_m=top_m, base_m=base_m, depth_m=depth, area_km2=area_km2,
        equiv_diameter_m=equiv_d, aspect=depth / max(equiv_d, 1.0),
        max_dbz=max_dbz, max_dbz_aloft=max_aloft,
        top_temp_c=profile.temp_at(top_m), base_temp_c=profile.temp_at(base_m),
        coldest_top_c=None, bright_band=bright_band,
    )


def classify(features, track=None, parents=(), connected=(),
             thresholds: Thresholds | None = None) -> Verdict:
    t = thresholds or Thresholds()

    parent_coldest = min(
        (p.coldest_top_c for p in parents if p.coldest_top_c is not None),
        default=None)
    detached_at = getattr(track, "detached_at", None) if track else None

    # Lineage first: 4.1.4 and 4.1.5 turn on where a cloud came from.
    if (parent_coldest is not None
            and parent_coldest <= t.anvil_parent_temp_c
            and features.max_dbz <= t.anvil_max_dbz
            and features.base_temp_c is not None
            and features.base_temp_c <= t.anvil_min_base_temp_c):
        attached = any(c in (CloudType.CUMULUS, CloudType.SMOKE_CUMULUS)
                       for c in connected)
        return Verdict(
            CloudType.ATTACHED_ANVIL if attached else CloudType.DETACHED_ANVIL,
            f"outflow from a parent that reached {parent_coldest:.0f} C, "
            f"base at {features.base_temp_c:.0f} C, peak "
            f"{features.max_dbz:.0f} dBZ")

    if (parent_coldest is not None
            and parent_coldest <= t.debris_parent_temp_c
            and detached_at is not None
            and features.top_temp_c is not None
            and features.top_temp_c > t.debris_collapse_temp_c):
        return Verdict(
            CloudType.DEBRIS,
            f"detached remnant of a parent that reached {parent_coldest:.0f} C, "
            f"top now {features.top_temp_c:.0f} C")

    # Shape second.
    if features.max_dbz_aloft >= t.convective_dbz_aloft:
        return Verdict(CloudType.CUMULUS,
                       f"{features.max_dbz_aloft:.0f} dBZ core above the 0 C level")
    if features.aspect >= t.convective_aspect and features.max_dbz >= 20.0:
        return Verdict(CloudType.CUMULUS,
                       f"aspect {features.aspect:.2f}, peak "
                       f"{features.max_dbz:.0f} dBZ")

    if (features.top_temp_c is not None
            and features.top_temp_c <= t.cirrus_max_temp_c
            and features.base_temp_c is not None
            and features.base_temp_c <= t.cirrus_max_temp_c
            and features.max_dbz <= t.cirrus_max_dbz
            and parent_coldest is None):
        return Verdict(CloudType.CIRRIFORM,
                       f"entirely colder than {t.cirrus_max_temp_c:.0f} C, "
                       f"peak {features.max_dbz:.0f} dBZ")

    broad = (features.area_km2 >= t.stratiform_area_km2
             and features.aspect <= t.stratiform_aspect
             and features.max_dbz_aloft <= t.stratiform_max_dbz_aloft)
    if broad or (features.bright_band
                 and features.max_dbz_aloft <= t.stratiform_max_dbz_aloft):
        why = "bright band present" if features.bright_band else (
            f"{features.area_km2:.0f} km2, aspect {features.aspect:.2f}, "
            f"{features.max_dbz_aloft:.0f} dBZ aloft")
        return Verdict(CloudType.THICK_LAYER, why)

    return Verdict(CloudType.CUMULUS,
                   "no confident match; defaulting to the most restrictive type",
                   confident=False)
