"""Synthetic afternoon driven through the full pipeline.

Unlike `emit_verdict.py`, nothing here is hand-authored. Reflectivity fields
go in; segmentation, tracking, geometry and the evaluator produce everything
else, and the display raster is built from the very array the segmenter
consumed. That last point is the one that matters: the picture and the
verdict cannot disagree, because they come from the same numbers.

    python3 demo_pipeline.py -o docs/data
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np

from llcc.evaluate import evaluate
from llcc.geometry import NM_TO_M, Grid, build_corridor
from llcc.pipeline import Pipeline, PipelineConfig
from llcc.radar import NO_ECHO, UNOBSERVABLE, Raster, RadarField
from llcc.serialize import SCHEMA_VERSION, snapshot_to_dict, verdict_to_dict, write_json
from llcc.world import ThermalProfile, VehicleConfig

GRID = Grid(nx=96, ny=96, nz=40, dx=1000.0, dy=1000.0, dz=500.0)
T0 = datetime(2026, 9, 4, 20, 55, 0)
SCAN = timedelta(minutes=4, seconds=30)
AZIMUTH = 45.0

PROFILE = ThermalProfile(levels=[
    (0, 29.0), (1000, 22.0), (2000, 15.5), (3000, 9.0), (3600, 5.0),
    (4600, 0.0), (5600, -5.0), (6600, -10.0), (7600, -15.0), (8700, -20.0),
    (10000, -33.0), (12000, -52.0), (14000, -68.0), (16000, -80.0),
])
TRAJECTORY = [(0, 0.0, 0), (2.5, 1.6, 290), (5, 3.2, 450),
              (11, 7.0, 640), (20, 12.6, 900), (30, 18.0, 1050)]
VEHICLE = VehicleConfig(name="reference vehicle",
                        triboelectric_exemption="4.1.10.2b",
                        triboelectric_basis="vehicle ESD analysis on file")


def blob(f, ci, cj, peak, radius, base, top):
    i, j, k = np.ogrid[:GRID.nx, :GRID.ny, :GRID.nz]
    mid, span = (base + top) / 2, max(1.0, (top - base) / 2)
    d = np.sqrt(((i - ci) / radius) ** 2 + ((j - cj) / radius) ** 2
                + ((k - mid) / span) ** 2)
    np.maximum(f, peak * np.exp(-1.3 * d * d) - 2.0, out=f)
    return f


def observability_mask() -> np.ndarray:
    """KMLB sits about 30 nmi on a bearing of 200 degrees from the pad. Past
    roughly 100 km the lowest tilt overshoots low cloud, and there is a cone
    of silence overhead."""
    obs = np.ones(GRID.shape, dtype=bool)
    east = GRID.east()[:, None]
    north = GRID.north()[None, :]
    a = np.radians(200.0)
    rx, ry = 30 * NM_TO_M * np.sin(a), 30 * NM_TO_M * np.cos(a)
    r = np.hypot(east - rx, north - ry)
    obs[r > 100_000.0, :] = False
    obs[r < 3_000.0, :] = False
    return obs


def frames():
    """Afternoon convection: a cell builds, flashes, sheds an anvil that
    detaches, and a stratiform deck drifts across the corridor."""
    layer = lambda f: blob(f, 62, 62, 26, 20.0, 7, 11)
    out = []

    specs = [
        # cell,                          anvil (None until it forms)
        ((40, 52, 34, 6.0, 3, 14),       None,                        []),
        ((41, 53, 50, 7.0, 3, 24),       None,                        [(9000., 14000.)]),
        ((42, 54, 50, 7.0, 3, 26),       (48, 59, 13, 9.0, 20, 27),   []),
        ((43, 55, 46, 6.5, 3, 24),       (54, 64, 12, 10.0, 20, 27),  []),
        ((44, 56, 30, 6.0, 3, 15),       (60, 69, 11, 11.0, 20, 27),  []),
        ((45, 57, 22, 5.5, 3, 12),       (66, 74, 10, 11.0, 20, 27),  []),
    ]
    for n, (cell, anvil, strikes) in enumerate(specs):
        f = np.full(GRID.shape, -30.0)
        blob(f, *cell)
        if anvil:
            blob(f, *anvil)
        layer(f)
        out.append((T0 + n * SCAN, f, strikes))
    return out


def to_plan(refl, obs) -> Raster:
    """Composite reflectivity, quantised with the two sentinels."""
    seen = obs.any(axis=2)
    comp = np.max(np.where(obs, refl, -1e9), axis=2)
    vals = []
    for j in range(GRID.ny - 1, -1, -1):
        for i in range(GRID.nx):
            if not seen[i, j]:
                vals.append(UNOBSERVABLE)
            elif comp[i, j] < -29.0:
                vals.append(NO_ECHO)
            else:
                vals.append(int(np.clip(round(comp[i, j]), -30, 75)))
    half = GRID.nx * GRID.dx / 2 / NM_TO_M
    return Raster(GRID.nx, GRID.ny, vals, "plan", {"half_nmi": half})


def to_xsec(refl, obs, nx=128, ny=72, max_nmi=25.0, max_km=18.0) -> Raster:
    """Vertical slice along the flight azimuth."""
    a = np.radians(AZIMUTH)
    vals = []
    for jj in range(ny):
        alt = max_km * 1000.0 * (1 - (jj + 0.5) / ny)
        k = GRID.level_at(alt)
        for ii in range(nx):
            down = max_nmi * NM_TO_M * (ii + 0.5) / nx
            i, j, _ = GRID.index_of(down * np.sin(a), down * np.cos(a), alt)
            if not (0 <= i < GRID.nx and 0 <= j < GRID.ny):
                vals.append(UNOBSERVABLE); continue
            if not obs[i, j, k]:
                vals.append(UNOBSERVABLE); continue
            v = refl[i, j, k]
            vals.append(NO_ECHO if (not np.isfinite(v) or v < -29.0)
                        else int(np.clip(round(v), -30, 75)))
    return Raster(nx, ny, vals, "xsec", {"max_nmi": max_nmi, "max_km": max_km})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outdir", default="docs/data")
    ap.add_argument("--assume-manifest-complete", action="store_true", default=True)
    args = ap.parse_args()

    corridor = build_corridor(GRID, TRAJECTORY, AZIMUTH, radius_m=1500.0)
    obs = observability_mask()
    pipe = Pipeline(PipelineConfig(grid=GRID, corridor=corridor, profile=PROFILE,
                                   vehicle=VEHICLE, azimuth_deg=AZIMUTH))

    out = []
    for when, refl, strikes in frames():
        snap = pipe.ingest(refl, when, observable=obs, strikes=strikes)
        verdict = evaluate(snap, when,
                           assume_manifest_complete=args.assume_manifest_complete)
        radar = RadarField(valid_time=when.isoformat(),
                           source="synthetic field, segmented in place",
                           plan=to_plan(refl, obs), xsec=to_xsec(refl, obs))
        doc = snapshot_to_dict(snap, pad="LC-39A", azimuth_deg=AZIMUTH, radar=radar)
        doc["trajectory"] = [[d, a, v] for d, a, v in TRAJECTORY]
        out.append({"snapshot": doc, "verdict": verdict_to_dict(verdict)})
        n = len([k for k in snap.objects if k != "DOMAIN"])
        print(f"  {when:%H:%M:%S}Z  {verdict.state.name:<14} "
              f"{n} object(s), {len(verdict.blocking)} blocking")

    write_json(f"{args.outdir}/replay.json", {
        "schema": SCHEMA_VERSION, "kind": "replay",
        "name": "Synthetic afternoon, segmented and tracked from the field",
        "pad": "LC-39A", "azimuth_deg": AZIMUTH,
        "manifest_gate_suppressed": args.assume_manifest_complete,
        "frames": out,
    })
    write_json(f"{args.outdir}/snapshot.json", out[-1]["snapshot"])
    write_json(f"{args.outdir}/verdict.json", out[-1]["verdict"])
    print(f"wrote {len(out)} frames to {args.outdir}/replay.json")


if __name__ == "__main__":
    main()
