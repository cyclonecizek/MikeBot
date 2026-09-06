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

# Renamed from noaa-nexrad-level2 in July 2025 at the request of the AWS
# Open Datasets team. Filename pattern and data format are unchanged;
# updates to the legacy bucket stopped on 1 September 2025.
BUCKET = "unidata-nexrad-level2"
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
              rho_min: float = 0.85,
              rho_min_surface: float = 0.93,
              texture_max_deg: float = 12.0,
              roi_nb: float = 0.7,
              roi_min_radius_m: float = 1000.0,
              freezing_level_m: float | None = None,
              verbose: bool = True) -> Scan:
    """UNVERIFIED: needs Py-ART. Grid one radar volume onto the ENU grid.

    Dual-pol QC runs on the polar gates before gridding, because filtering
    after interpolation smears non-meteorological returns into neighbouring
    cells rather than removing them.
    """
    import pyart  # noqa: F401  (imported here so the module loads without it)

    fields = radar.fields
    dual = "cross_correlation_ratio" in fields and "differential_reflectivity" in fields

    def pull(name):
        f = fields.get(name)
        return None if f is None else np.ma.filled(f["data"], np.nan)

    refl = fields["reflectivity"]["data"]
    heights = None
    if "gate_altitude" in dir(radar) and radar.gate_altitude is not None:
        heights = np.asarray(radar.gate_altitude["data"], dtype=float)

    report: dict = {}
    keep = meteorological_mask(
        np.ma.filled(refl, np.nan),
        rho_hv=pull("cross_correlation_ratio"),
        zdr=pull("differential_reflectivity"),
        phidp=pull("differential_phase"),
        velocity=pull("velocity"),
        heights_m=heights,
        radar_alt_m=radar_alt_m,
        freezing_level_m=freezing_level_m,
        rho_min=rho_min,
        rho_min_surface=rho_min_surface,
        texture_max_deg=texture_max_deg,
        report=report,
    )
    if verbose and report:
        total = report.get("total", 1)
        parts = [f"{k}={v}" for k, v in report.items()
                 if k not in ("total", "kept")]
        print(f"    QC kept {report.get('kept', 0)}/{total} gates "
              f"({report.get('kept', 0) / max(total, 1):.1%})  " + " ".join(parts))
    radar.add_field_like("reflectivity", "refl_qc",
                         np.ma.masked_where(~keep, refl), replace_existing=True)

    # A coverage field carrying 1.0 wherever a QC-passed gate exists. Gridded
    # alongside the reflectivity, it tells us which grid cells any gate
    # actually reached.
    #
    # This is not belt-and-braces. Py-ART can return ungridded cells as 0.0
    # rather than masked, and 0 dBZ is exactly the standard's cloud
    # threshold -- so every unsampled cell in the domain reads as cloud, and
    # the segmentation produces one enormous spurious object. Inferring
    # coverage from the reflectivity values themselves cannot work, because
    # 0 dBZ is both a legitimate measurement and the fill value.
    radar.add_field_like("reflectivity", "coverage",
                         np.ma.masked_where(~keep, np.ones_like(np.asarray(refl))),
                         replace_existing=True)

    half_x = grid.nx * grid.dx / 2.0
    half_y = grid.ny * grid.dy / 2.0
    top = grid.z0 + grid.nz * grid.dz

    gridded = pyart.map.grid_from_radars(
        (radar,),
        grid_shape=(grid.nz, grid.ny, grid.nx),
        grid_limits=((grid.z0, top), (-half_y, half_y), (-half_x, half_x)),
        fields=["refl_qc", "coverage"],
        # Without this Py-ART centres the grid on the RADAR, not on the pad,
        # so the reflectivity would sit 55 km from everything else in the
        # system -- the corridor, both distance fields, and the map tiles are
        # all in a pad-centred ENU frame. Every distance would be measured
        # from the wrong origin.
        grid_origin=(grid.origin_lat, grid.origin_lon),
        grid_origin_alt=grid.z0,
        weighting_function="Barnes2",
        # The radius of influence has to grow with range or the gaps between
        # tilts slice a continuous cloud into disconnected slabs, which
        # corrupts the connectivity the segmentation rests on. But it must
        # not grow far: Barnes weighting smears every surviving gate across
        # the whole radius and interpolates between them, so a scatter of
        # isolated returns becomes a continuous low-dBZ haze -- and since the
        # standard's cloud boundary is 0 dBZ, that haze becomes cloud. A
        # single spurious object tens of miles across is the result.
        #
        # nb below Py-ART's 1.5 default tightens the beam-width term.
        roi_func="dist_beam",
        nb=roi_nb,
        min_radius=roi_min_radius_m,
    )

    def unpack(name):
        raw = gridded.fields[name]["data"]
        arr = (np.ma.filled(raw.astype(float), np.nan)
               if np.ma.isMaskedArray(raw) else np.asarray(raw, dtype=float))
        return np.transpose(arr, (2, 1, 0))       # (z,y,x) -> (x,y,z)

    refl_grid = unpack("refl_qc")
    coverage = unpack("coverage")
    sampled = np.isfinite(coverage) & (coverage > 0.05)
    refl_grid = np.where(sampled, refl_grid, np.nan)

    if verbose:
        cells = refl_grid.size
        finite = np.isfinite(refl_grid)
        cloud = finite & (refl_grid >= 0.0)
        solid = finite & (refl_grid >= 15.0)
        print(f"    gridded {cells} cells: {finite.sum() / cells:.1%} sampled, "
              f"{cloud.sum() / cells:.2%} at or above 0 dBZ, "
              f"{solid.sum() / cells:.2%} at 15 dBZ or more")

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
