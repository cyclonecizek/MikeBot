"""Live run against real data. UNVERIFIED end to end.

Fetches a KMLB Level II volume and a GLM window, grids them, runs the
pipeline, and writes the display contract. Needs network plus Py-ART and
netCDF4:

    pip install arm-pyart netCDF4

    python3 run_live.py --time "2026-09-04T21:04:00" --sounding xmr.txt

The array transforms this leans on are tested; the fetch and decode calls
are not. Expect to shake them out on the first real file -- especially the
Py-ART field names, which vary by product, and the volume timestamp parse.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from llcc.evaluate import evaluate, format_verdict
from llcc.geometry import NM_TO_M, Grid, build_corridor
from llcc.ingest.glm import load_window
from llcc.ingest.nexrad import KMLB, load_scan
from llcc.ingest.profile import from_pairs, parse_uwyo_sounding
from llcc.pipeline import Pipeline, PipelineConfig
from llcc.radar import NO_ECHO, UNOBSERVABLE, RadarField, Raster
from llcc.serialize import SCHEMA_VERSION, snapshot_to_dict, verdict_to_dict, write_json
from llcc.world import VehicleConfig

PAD = {"lat": 28.6083, "lon": -80.6041}
AZIMUTH = 45.0
TRAJECTORY = [(0, 0.0, 0), (2.5, 1.6, 290), (5, 3.2, 450),
              (11, 7.0, 640), (20, 12.6, 900), (30, 18.0, 1050)]

FALLBACK_PROFILE = [(0, 29.0), (1000, 22.0), (2000, 15.5), (3000, 9.0),
                    (3600, 5.0), (4600, 0.0), (5600, -5.0), (6600, -10.0),
                    (7600, -15.0), (8700, -20.0), (10000, -33.0),
                    (12000, -52.0), (14000, -68.0), (16000, -80.0)]


def radar_offset_from_pad() -> tuple[float, float]:
    m_lat = 111_132.0
    m_lon = 111_320.0 * np.cos(np.radians(PAD["lat"]))
    return ((KMLB["lon"] - PAD["lon"]) * m_lon,
            (KMLB["lat"] - PAD["lat"]) * m_lat)


def to_plan(refl, obs, grid) -> Raster:
    seen = obs.any(axis=2)
    comp = np.max(np.where(obs & np.isfinite(refl), refl, -1e9), axis=2)
    vals = []
    for j in range(grid.ny - 1, -1, -1):
        for i in range(grid.nx):
            if not seen[i, j]:
                vals.append(UNOBSERVABLE)
            elif comp[i, j] < -29.0:
                vals.append(NO_ECHO)
            else:
                vals.append(int(np.clip(round(comp[i, j]), -30, 75)))
    return Raster(grid.nx, grid.ny, vals, "plan",
                  {"half_nmi": grid.nx * grid.dx / 2 / NM_TO_M})


def to_xsec(refl, obs, grid, nx=128, ny=72, max_nmi=25.0, max_km=18.0) -> Raster:
    a = np.radians(AZIMUTH)
    vals = []
    for jj in range(ny):
        alt = max_km * 1000.0 * (1 - (jj + 0.5) / ny)
        k = grid.level_at(alt)
        for ii in range(nx):
            down = max_nmi * NM_TO_M * (ii + 0.5) / nx
            i, j, _ = grid.index_of(down * np.sin(a), down * np.cos(a), alt)
            if not (0 <= i < grid.nx and 0 <= j < grid.ny) or not obs[i, j, k]:
                vals.append(UNOBSERVABLE)
                continue
            v = refl[i, j, k]
            vals.append(NO_ECHO if not np.isfinite(v) or v < -29.0
                        else int(np.clip(round(v), -30, 75)))
    return Raster(nx, ny, vals, "xsec", {"max_nmi": max_nmi, "max_km": max_km})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", required=True, help="ISO time, e.g. 2026-09-04T21:04:00")
    ap.add_argument("--site", default="KMLB")
    ap.add_argument("--sounding", help="University of Wyoming text sounding for XMR")
    ap.add_argument("--glm-minutes", type=float, default=240.0,
                    help="lightning history to ingest; wider than the criteria "
                         "need, so objects arrive with provenance")
    ap.add_argument("-o", "--outdir", default="docs/data")
    ap.add_argument("--assume-manifest-complete", action="store_true")
    args = ap.parse_args()

    when = datetime.fromisoformat(args.time)
    grid = Grid(nx=96, ny=96, nz=40, dx=1000.0, dy=1000.0, dz=500.0)
    corridor = build_corridor(grid, TRAJECTORY, AZIMUTH, radius_m=1500.0)

    if args.sounding:
        profile = parse_uwyo_sounding(Path(args.sounding).read_text())
        print(f"profile: {len(profile.levels)} levels from {args.sounding}")
    else:
        profile = from_pairs(FALLBACK_PROFILE)
        print("profile: built-in fallback. Supply --sounding for a real run.")
    print(f"  0 C at {profile.isotherm_altitude(0.0):.0f} m, "
          f"-20 C at {profile.isotherm_altitude(-20.0):.0f} m")

    radar_east, radar_north = radar_offset_from_pad()
    print(f"fetching {args.site} near {when:%Y-%m-%d %H:%M:%S}Z ...")
    scan = load_scan(args.site, when, grid, radar_east, radar_north,
                     KMLB["alt_m"])
    print(f"  {scan.key}  dual-pol={scan.dual_pol}  "
          f"observable={scan.observable.mean():.1%} of volume")

    print(f"fetching GLM back {args.glm_minutes:.0f} min ...")
    flashes = load_window(when, args.glm_minutes, PAD["lat"], PAD["lon"])
    print(f"  {len(flashes)} flashes")

    pipe = Pipeline(PipelineConfig(grid=grid, corridor=corridor, profile=profile,
                                   vehicle=VehicleConfig(
                                       name=args.site + " run",
                                       triboelectric_exemption="4.1.10.2b",
                                       triboelectric_basis="ESD analysis on file"),
                                   azimuth_deg=AZIMUTH))

    # Replay lightning history before the scan so clocks start correctly, then
    # ingest the scan itself.
    strikes = [p for f in flashes if f.time <= scan.time for p in f.points_en]
    snapshot = pipe.ingest(scan.refl, scan.time, observable=scan.observable,
                           strikes=strikes)
    verdict = evaluate(snapshot, scan.time,
                       assume_manifest_complete=args.assume_manifest_complete)
    print()
    print(format_verdict(verdict))

    radar = RadarField(valid_time=scan.time.isoformat(),
                       source=f"{args.site} Level II",
                       plan=to_plan(scan.refl, scan.observable, grid),
                       xsec=to_xsec(scan.refl, scan.observable, grid))
    doc = snapshot_to_dict(snapshot, pad="LC-39A", azimuth_deg=AZIMUTH, radar=radar)
    doc["trajectory"] = [[d, a, v] for d, a, v in TRAJECTORY]
    write_json(f"{args.outdir}/snapshot.json", doc)
    write_json(f"{args.outdir}/verdict.json", verdict_to_dict(verdict))
    write_json(f"{args.outdir}/replay.json", {
        "schema": SCHEMA_VERSION, "kind": "replay",
        "name": f"{args.site} {scan.time:%Y-%m-%d %H:%M}Z",
        "pad": "LC-39A", "azimuth_deg": AZIMUTH,
        "manifest_gate_suppressed": args.assume_manifest_complete,
        "frames": [{"snapshot": doc, "verdict": verdict_to_dict(verdict)}],
    })
    print(f"\nwrote {args.outdir}/snapshot.json, verdict.json, replay.json")


if __name__ == "__main__":
    main()
