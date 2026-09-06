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
from llcc.ingest.nexrad import KMLB, grid_scan
from llcc.ingest.sources import aws_index, fetch, resolve_volumes
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
                    (12000, -52.0), (14000, -68.0), (16000, -80.0), (18000, -76.0), (20000, -70.0)]


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
    ap.add_argument("--time", required=True,
                    help="ISO start time, e.g. 2026-08-26T16:00:00")
    ap.add_argument("--end", help="ISO end time. Omit for a single volume.")
    ap.add_argument("--source", default="aws", choices=("aws", "gcp", "thredds"),
                    help="aws: unidata-nexrad-level2, individual volumes. "
                         "gcp: hourly tars, needs unpacking. thredds: rolling "
                         "archive, recent dates only.")
    ap.add_argument("--list-only", action="store_true",
                    help="resolve volumes and stop, without decoding")
    ap.add_argument("--stride", type=int, default=1,
                    help="take every Nth volume; 2 or 3 keeps replay.json small")
    ap.add_argument("--site", default="KMLB")
    ap.add_argument("--sounding", help="University of Wyoming text sounding for XMR")
    ap.add_argument("--disturbed-weather", choices=("yes", "no", "unknown"),
                    default="no",
                    help="declare whether a front, trough, squall line or "
                         "tropical wave is driving the convection. Sea breeze "
                         "and outflow boundaries are NOT disturbed weather "
                         "(section 3.2). Defaults to no; 'unknown' leaves "
                         "LLCCR 21 indeterminate.")
    ap.add_argument("--rho-min", type=float, default=0.85,
                    help="correlation coefficient floor away from the surface")
    ap.add_argument("--rho-min-surface", type=float, default=0.93,
                    help="stricter floor below 1.2 km AGL, where clutter lives")
    ap.add_argument("--texture-max", type=float, default=12.0,
                    help="max PhiDP texture in degrees; raise if it eats "
                         "convective cores")
    ap.add_argument("--roi-nb", type=float, default=0.7,
                    help="beam-width factor for the gridding radius of "
                         "influence. Lower means less interpolation smear; "
                         "Py-ART's default is 1.5.")
    ap.add_argument("--min-neighbours", type=int, default=7,
                    help="cloud voxels required in a 3x3x3 neighbourhood. "
                         "Breaks percolating speckle; a one-level sheet has 9.")
    ap.add_argument("--threshold", action="append", default=[],
                    metavar="NAME=VALUE",
                    help="override a classifier threshold, repeatable. The "
                         "display's what-if panel prints these for you.")
    ap.add_argument("--no-classify", action="store_true",
                    help="treat every object as cumulus, the most restrictive "
                         "type. The honest baseline when the classifier is "
                         "not trusted.")
    ap.add_argument("--min-voxels", type=int, default=24,
                    help="smallest object that counts as a cloud, in voxels")
    ap.add_argument("--no-glm", action="store_true",
                    help="skip lightning entirely; much faster for a first run")
    ap.add_argument("--glm-minutes", type=float, default=60.0,
                    help="lightning history to ingest; wider than the criteria "
                         "need, so objects arrive with provenance")
    ap.add_argument("-o", "--outdir", default="docs/data")
    ap.add_argument("--assume-manifest-complete", action="store_true")
    args = ap.parse_args()

    when = datetime.fromisoformat(args.time)
    end = datetime.fromisoformat(args.end) if args.end else when
    grid = Grid(nx=96, ny=96, nz=40, dx=1000.0, dy=1000.0, dz=500.0)
    corridor = build_corridor(grid, TRAJECTORY, AZIMUTH, radius_m=1500.0)

    from llcc.classify import Thresholds
    thresholds = Thresholds()
    for spec in args.threshold:
        name, _, value = spec.partition("=")
        if not hasattr(thresholds, name):
            raise SystemExit(f"unknown threshold {name!r}; see llcc/classify.py")
        setattr(thresholds, name, float(value))
    if args.threshold:
        print("thresholds: " + ", ".join(args.threshold))

    if args.sounding:
        profile = parse_uwyo_sounding(Path(args.sounding).read_text())
        print(f"profile: {len(profile.levels)} levels from {args.sounding}")
    else:
        profile = from_pairs(FALLBACK_PROFILE)
        print("profile: built-in fallback. Supply --sounding for a real run.")
    print(f"  0 C at {profile.isotherm_altitude(0.0):.0f} m, "
          f"-20 C at {profile.isotherm_altitude(-20.0):.0f} m")

    radar_east, radar_north = radar_offset_from_pad()

    print(f"resolving {args.site} volumes from {args.source} ...")
    if args.source == "aws":
        volumes = aws_index(args.site, when, end)
    else:
        volumes = resolve_volumes(args.site, when, end, source=args.source)
    if not volumes:
        raise SystemExit(f"no {args.site} volumes between "
                         f"{when:%H:%M}Z and {end:%H:%M}Z")
    volumes = volumes[::max(1, args.stride)]
    print(f"{len(volumes)} volume(s) to process "
          f"({volumes[0].time:%H:%M:%S}Z to {volumes[-1].time:%H:%M:%S}Z)")
    if args.list_only:
        for v in volumes:
            print(f"  {v.time:%H:%M:%S}Z  {Path(v.path).name}")
        return

    if args.no_glm:
        print("skipping GLM (--no-glm): lightning criteria will read as clear")
        flashes = []
    else:
        span = args.glm_minutes + (end - when).total_seconds() / 60.0
        print(f"fetching GLM over {span:.0f} min "
              f"(~{span * 3:.0f} granules) ...")
        flashes = load_window(end, span, PAD["lat"], PAD["lon"])
        print(f"  {len(flashes)} flashes")

    pipe = Pipeline(PipelineConfig(
        grid=grid, corridor=corridor, profile=profile,
        vehicle=VehicleConfig(name=args.site + " run",
                              triboelectric_exemption="4.1.10.2b",
                              triboelectric_basis="ESD analysis on file"),
        azimuth_deg=AZIMUTH, min_voxels=args.min_voxels,
        classify=not args.no_classify, thresholds=thresholds,
        min_neighbours=args.min_neighbours,
        disturbed_weather=(None if args.disturbed_weather == "unknown"
                           else args.disturbed_weather == "yes")))

    import pyart

    frames = []
    last = datetime.min
    for n, vol in enumerate(volumes, start=1):
        fetch(vol)
        radar = pyart.io.read_nexrad_archive(vol.path)
        scan = grid_scan(radar, grid, radar_east, radar_north, KMLB["alt_m"],
                         rho_min=args.rho_min,
                         rho_min_surface=args.rho_min_surface,
                         texture_max_deg=args.texture_max,
                         roi_nb=args.roi_nb,
                         freezing_level_m=profile.isotherm_altitude(0.0))
        scan.key = vol.path
        # Trust the archive filename over the in-file time units string,
        # which varies between products.
        scan.time = vol.time

        # Feed only the flashes since the previous volume, so each ingest
        # advances the event log rather than replaying the whole window.
        strikes = [pt for f in flashes if last < f.time <= scan.time
                   for pt in f.points_en]
        last = scan.time

        snapshot = pipe.ingest(scan.refl, scan.time, observable=scan.observable,
                               strikes=strikes)
        verdict = evaluate(snapshot, scan.time,
                           assume_manifest_complete=args.assume_manifest_complete)

        radar_field = RadarField(
            valid_time=scan.time.isoformat(), source=f"{args.site} Level II",
            plan=to_plan(scan.refl, scan.observable, grid),
            xsec=to_xsec(scan.refl, scan.observable, grid))
        doc = snapshot_to_dict(snapshot, pad="LC-39A", azimuth_deg=AZIMUTH, origin=PAD, overlay=pipe.overlay(), classifier=pipe.classifier_state(),
                               radar=radar_field)
        doc["trajectory"] = [[d, a, v] for d, a, v in TRAJECTORY]
        frames.append({"snapshot": doc, "verdict": verdict_to_dict(verdict, snapshot=snapshot)})

        objs = len([k for k in snapshot.objects if k != "DOMAIN"])
        print(f"  [{n}/{len(volumes)}] {scan.time:%H:%M:%S}Z  "
              f"{verdict.state.name:<14} {objs} object(s), "
              f"{len(verdict.blocking)} blocking, "
              f"{scan.observable.mean():.0%} observable")

    write_json(f"{args.outdir}/replay.json", {
        "schema": SCHEMA_VERSION, "kind": "replay",
        "name": f"{args.site} {when:%Y-%m-%d %H:%M}-{end:%H:%M}Z",
        "pad": "LC-39A", "azimuth_deg": AZIMUTH,
        "manifest_gate_suppressed": args.assume_manifest_complete,
        "frames": frames,
    })
    write_json(f"{args.outdir}/snapshot.json", frames[-1]["snapshot"])
    write_json(f"{args.outdir}/verdict.json", frames[-1]["verdict"])
    print(f"\nwrote {len(frames)} frames to {args.outdir}/replay.json")


if __name__ == "__main__":
    main()
