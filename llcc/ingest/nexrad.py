"""NEXRAD Level II ingest.

The decode and gridding calls need Py-ART and are marked UNVERIFIED: they
are written against Py-ART's documented API but have not been run against a
real archive file in this environment. The array transforms they hand off to
are in `beam.py` and are tested.

Install with: pip install arm-pyart
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ..geometry import Grid
from .beam import meteorological_mask, observability_mask
from .s3 import download, list_keys, nexrad_prefix

BUCKET = "noaa-nexrad-level2"
# KMLB, Melbourne FL. Antenna at about 11 m MSL.
KMLB = {"lat": 28.1131, "lon": -80.6544, "alt_m": 11.0}


@dataclass
class Scan:
    time: datetime
    refl: np.ndarray            # dBZ on the grid, NaN where not sampled
    observable: np.ndarray      # bool
    site: str
    key: str = ""
    dual_pol: bool = False


def nearest_key(site: str, when: datetime, window_min: float = 15.0) -> str | None:
    """UNVERIFIED: needs network. Archive key whose volume start is closest.

    Level II filenames end in the volume start time, e.g.
    KMLB20260904_210412_V06. MDM sidecar files are skipped.
    """
    keys = [k for k in list_keys(BUCKET, nexrad_prefix(site, when))
            if not k.endswith("_MDM")]
    best, best_gap = None, timedelta(minutes=window_min)
    for key in keys:
        stamp = key.rsplit("/", 1)[-1]
        try:
            t = datetime.strptime(stamp[4:19], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        gap = abs(t - when)
        if gap <= best_gap:
            best, best_gap = key, gap
    return best


def grid_scan(radar, grid: Grid, radar_east: float = 0.0,
              radar_north: float = 0.0, radar_alt_m: float = 0.0,
              rho_min: float = 0.85) -> Scan:
    """UNVERIFIED: needs Py-ART. Grid one radar volume onto the ENU grid.

    Dual-pol QC runs on the polar gates before gridding, because filtering
    after interpolation smears non-meteorological returns into neighbouring
    cells rather than removing them.
    """
    import pyart  # noqa: F401  (imported here so the module loads without it)

    fields = radar.fields
    dual = "cross_correlation_ratio" in fields and "differential_reflectivity" in fields

    refl = fields["reflectivity"]["data"]
    rho = fields.get("cross_correlation_ratio", {}).get("data") if dual else None
    zdr = fields.get("differential_reflectivity", {}).get("data") if dual else None
    keep = meteorological_mask(np.ma.filled(refl, np.nan),
                               None if rho is None else np.ma.filled(rho, np.nan),
                               None if zdr is None else np.ma.filled(zdr, np.nan),
                               rho_min=rho_min)
    radar.add_field_like("reflectivity", "refl_qc",
                         np.ma.masked_where(~keep, refl), replace_existing=True)

    half_x = grid.nx * grid.dx / 2.0
    half_y = grid.ny * grid.dy / 2.0
    top = grid.z0 + grid.nz * grid.dz

    gridded = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(grid.nz, grid.ny, grid.nx),
        grid_limits=((grid.z0, top), (-half_y, half_y), (-half_x, half_x)),
        fields=["refl_qc"],
        weighting_function="Barnes2",
        # Radius of influence must grow with range or the gaps between tilts
        # slice a continuous cloud into disconnected slabs, which corrupts
        # the connectivity the whole segmentation rests on.
        roi_func="dist_beam",
        min_radius=max(grid.dx, grid.dz),
    )

    data = np.asarray(gridded.fields["refl_qc"]["data"])
    data = np.ma.filled(data, np.nan)
    refl_grid = np.transpose(data, (2, 1, 0))     # (z,y,x) -> (x,y,z)

    observable = observability_mask(grid, radar_east, radar_north, radar_alt_m)
    when = datetime.strptime(radar.time["units"].split("since")[-1].strip(),
                             "%Y-%m-%dT%H:%M:%SZ")
    return Scan(time=when, refl=refl_grid, observable=observable,
                site=radar.metadata.get("instrument_name", "?"), dual_pol=dual)


def load_scan(site: str, when: datetime, grid: Grid,
              radar_east: float = 0.0, radar_north: float = 0.0,
              radar_alt_m: float = 0.0, cache_dir: str = "/tmp") -> Scan:
    """UNVERIFIED: needs network and Py-ART. Fetch, decode and grid one volume."""
    import pyart

    key = nearest_key(site, when)
    if key is None:
        raise FileNotFoundError(f"no {site} volume near {when:%Y-%m-%d %H:%M}Z")
    path = f"{cache_dir}/{key.rsplit('/', 1)[-1]}"
    download(BUCKET, key, path)
    radar = pyart.io.read_nexrad_archive(path)
    scan = grid_scan(radar, grid, radar_east, radar_north, radar_alt_m)
    scan.key = key
    return scan
