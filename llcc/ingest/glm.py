"""GLM lightning ingest.

The standard defines lightning as the entire discharge including all its
channels and branches, and LLCCR 34a requires that to be accounted for. So
this uses the union of group and event footprints, never `flash_lat` and
`flash_lon`: the flash centroid is energy-weighted and sits well inside the
true extent, which would systematically understate distance to the flight
path.

The NetCDF read is UNVERIFIED. `footprint_points` and `dilate_footprint`
are pure and tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from .s3 import download, glm_prefix, list_keys

BUCKET = "noaa-goes19"
# GLM navigates to an assumed lightning ellipsoid near 16 km, so emission
# from lower tops is displaced. Buffering the footprint is defensible and
# auditable; a parallax correction would need a cloud-top height we have
# deliberately chosen not to take from GOES.
PARALLAX_BUFFER_M = 4000.0


@dataclass
class Flash:
    time: datetime
    points_en: list[tuple[float, float]]     # group/event footprint, ENU metres
    energy_j: float = 0.0

    def min_distance_to(self, east: float, north: float) -> float:
        if not self.points_en:
            return float("inf")
        pts = np.asarray(self.points_en)
        return float(np.min(np.hypot(pts[:, 0] - east, pts[:, 1] - north)))


def latlon_to_en(lat, lon, origin_lat: float, origin_lon: float):
    """Local ENU offsets in metres. Flat-earth; centimetres of error at these
    ranges, and the radar gridding must share the same origin."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(origin_lat))
    return (lon - origin_lon) * m_per_deg_lon, (lat - origin_lat) * m_per_deg_lat


def footprint_points(group_lat, group_lon, origin_lat: float,
                     origin_lon: float) -> list[tuple[float, float]]:
    """Union of group locations as the flash footprint, in ENU metres."""
    east, north = latlon_to_en(group_lat, group_lon, origin_lat, origin_lon)
    return [(float(e), float(n)) for e, n in zip(np.atleast_1d(east),
                                                 np.atleast_1d(north))]


def dilate_footprint(points, buffer_m: float = PARALLAX_BUFFER_M, n: int = 8):
    """Ring the footprint with a fixed buffer for navigation and parallax error.

    Applied as dilation rather than correction: it is auditable, and it errs
    toward a larger apparent extent, which shortens no standoff.
    """
    out = list(points)
    for east, north in points:
        for k in range(n):
            a = 2 * np.pi * k / n
            out.append((east + buffer_m * np.cos(a),
                        north + buffer_m * np.sin(a)))
    return out


def read_flashes(path: str, origin_lat: float, origin_lon: float,
                 buffer_m: float = PARALLAX_BUFFER_M) -> list[Flash]:
    """UNVERIFIED: needs netCDF4. Read one GLM LCFA granule into Flash records."""
    from netCDF4 import Dataset

    out: list[Flash] = []
    with Dataset(path) as ds:
        base = datetime(2000, 1, 1, 12, 0, 0)
        g_time = np.asarray(ds.variables["group_time_offset"][:])
        g_lat = np.asarray(ds.variables["group_lat"][:])
        g_lon = np.asarray(ds.variables["group_lon"][:])
        g_parent = np.asarray(ds.variables["group_parent_flash_id"][:])
        g_energy = np.asarray(ds.variables["group_energy"][:])

        for fid in np.unique(g_parent):
            sel = g_parent == fid
            pts = footprint_points(g_lat[sel], g_lon[sel], origin_lat, origin_lon)
            out.append(Flash(
                time=base + timedelta(seconds=float(g_time[sel].min())),
                points_en=dilate_footprint(pts, buffer_m),
                energy_j=float(np.nansum(g_energy[sel])),
            ))
    return out


def load_window(when: datetime, minutes: float, origin_lat: float,
                origin_lon: float, cache_dir: str = "/tmp",
                workers: int = 12, max_granules: int = 400) -> list[Flash]:
    """UNVERIFIED: needs network and netCDF4.

    Ingest a wider domain than the 10 nmi the criteria use -- a cell that
    flashed 40 nmi offshore an hour ago still carries history that governs
    4.1.1.2 and every anvil clock. Objects arriving with no provenance have
    to be treated as indeterminate.
    """
    if minutes <= 0:
        return []

    start = when - timedelta(minutes=minutes)
    keys: list[str] = []
    probe = start.replace(minute=0, second=0, microsecond=0)
    while probe <= when:
        keys.extend(list_keys(BUCKET, glm_prefix(probe)))
        probe += timedelta(hours=1)

    # GLM granules cover 20 seconds each, so an hour is about 180 files and a
    # six-hour window is over a thousand. Downloading those one at a time is
    # the difference between a minute and an afternoon.
    if len(keys) > max_granules:
        raise RuntimeError(
            f"{len(keys)} GLM granules for a {minutes:.0f} min window exceeds "
            f"max_granules={max_granules}. Narrow --glm-minutes, or raise the "
            f"cap deliberately.")

    import concurrent.futures as cf

    def grab(key: str) -> str:
        path = f"{cache_dir}/{key.rsplit('/', 1)[-1]}"
        download(BUCKET, key, path)
        return path

    paths: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, path in enumerate(pool.map(grab, keys), start=1):
            paths.append(path)
            if n % 25 == 0 or n == len(keys):
                print(f"    GLM {n}/{len(keys)} granules", flush=True)

    flashes: list[Flash] = []
    for path in paths:
        try:
            flashes.extend(read_flashes(path, origin_lat, origin_lon))
        except Exception as exc:
            print(f"    skipped {path}: {exc}")
    return [f for f in flashes if start <= f.time <= when]
