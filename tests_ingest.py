"""Ingest tests.

Only the pure transforms are covered. Fetch and decode need network access,
Py-ART and netCDF4, none of which are available here -- those functions are
marked UNVERIFIED in the source and must be shaken out against a real file.

Requires numpy.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from llcc.geometry import Grid
from llcc.ingest.beam import (
    beam_height, beam_width, detectable, meteorological_mask,
    minimum_detectable_dbz, observability_mask, phidp_texture,
)
from llcc.ingest.glm import (
    Flash, dilate_footprint, footprint_points, latlon_to_en,
)
from llcc.ingest.profile import from_pairs, isotherms, parse_uwyo_sounding
from llcc.ingest.s3 import glm_prefix, listing_url, nexrad_prefix, parse_listing

FAILURES: list[str] = []


def check(name, condition):
    if not condition:
        FAILURES.append(name)
    print(f"  {'pass' if condition else 'FAIL'}  {name}")


print("beam geometry")
check("beam height rises with elevation angle",
      beam_height(50_000, 0.5) < beam_height(50_000, 4.0))
h = beam_height(100_000, 0.0)
check("4/3-earth curvature lifts a level beam ~590 m at 100 km", 550 < h < 640)
check("curvature grows as the square of range",
      abs(beam_height(200_000, 0.0) / h - 4.0) < 0.1)
check("beam width is about 1.6 km at 100 km",
      1500 < beam_width(100_000) < 1750)
check("sensitivity degrades as 20 log10 range",
      abs((minimum_detectable_dbz(100_000) - minimum_detectable_dbz(10_000)) - 20.0) < 0.1)
check("0 dBZ is detectable near the radar", bool(detectable(20_000)))
check("0 dBZ is not detectable at long range", not bool(detectable(200_000)))

print("\nobservability mask")
g = Grid(nx=48, ny=48, nz=30, dx=2000.0, dy=2000.0, dz=500.0)
obs = observability_mask(g, radar_east=0.0, radar_north=0.0)
check("some volume is observable", obs.any())
check("not all volume is observable", not obs.all())
overhead = obs[g.nx // 2, g.ny // 2, -1]
check("a cone of silence exists directly overhead", not bool(overhead))
far = observability_mask(g, radar_east=0.0, radar_north=0.0, max_range_m=20_000.0)
check("range limit shrinks the observable volume", far.sum() < obs.sum())
blocked = observability_mask(g, 0.0, 0.0, blocked_sectors=[(0.0, 90.0)])
check("a blocked sector removes volume", blocked.sum() < obs.sum())

print("\ndual-pol quality control")
refl = np.array([40.0, 40.0, 40.0, 40.0])
rho = np.array([0.99, 0.60, 0.99, np.nan])
zdr = np.array([0.5, 4.0, 9.0, 0.5])
keep = meteorological_mask(refl, rho_hv=rho, zdr=zdr)
check("high rho and modest ZDR is kept", bool(keep[0]))
check("low rho is rejected as non-meteorological", not bool(keep[1]))
check("extreme ZDR is rejected", not bool(keep[2]))
check("missing rho is rejected rather than assumed good", not bool(keep[3]))
check("without dual-pol the filter degrades but does not crash",
      meteorological_mask(refl).all())

print("\nclutter discrimination")
rain_phi = np.cumsum(np.full((2, 40), 0.4), axis=1)
clut_phi = np.random.default_rng(1).uniform(0, 360, size=(2, 40))
check("smooth PhiDP gives low texture", float(phidp_texture(rain_phi).mean()) < 2.0)
check("erratic PhiDP gives high texture", float(phidp_texture(clut_phi).mean()) > 40.0)
check("phase wrapping does not inflate texture",
      float(phidp_texture(np.array([[358.0, 359.0, 0.0, 1.0, 2.0, 3.0]])).mean()) < 2.0)

r = np.full((2, 6), 35.0)
good = np.full((2, 6), 0.99)
kept = meteorological_mask(r, rho_hv=good, phidp=np.cumsum(np.full((2, 6), 0.3), axis=1))
check("smooth precipitation survives every test", bool(kept.all()))
kept = meteorological_mask(r, rho_hv=good,
                           phidp=np.random.default_rng(2).uniform(0, 360, (2, 6)))
check("erratic phase is rejected even with high rho", not bool(kept.any()))

# Height-dependent rho: the same 0.90 gate is clutter near the surface and
# legitimate melting-layer return at the bright band.
low = np.full((1, 3), 400.0)
mid = np.full((1, 3), 4200.0)
rho90 = np.full((1, 3), 0.90)
r3 = np.full((1, 3), 30.0)
check("rho 0.90 near the surface is rejected as clutter",
      not meteorological_mask(r3, rho_hv=rho90, heights_m=low).any())
check("the same rho in the melting layer is kept",
      meteorological_mask(r3, rho_hv=rho90, heights_m=mid,
                          freezing_level_m=4600.0).all())

check("stationary targets are rejected",
      not meteorological_mask(r3, rho_hv=np.full((1, 3), 0.90),
                              velocity=np.zeros((1, 3))).any())
check("stationary but very high rho is kept",
      meteorological_mask(r3, rho_hv=np.full((1, 3), 0.99),
                          velocity=np.zeros((1, 3))).all())

rep: dict = {}
meteorological_mask(r, rho_hv=np.full((2, 6), 0.5), report=rep)
check("the report names which test did the rejecting",
      rep.get("rho_hv", 0) == 12 and rep.get("kept") == 0)

print("\nS3 keys and listing")
xml = """<?xml version="1.0"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>true</IsTruncated>
  <NextContinuationToken>abc123</NextContinuationToken>
  <Contents><Key>2026/09/04/KMLB/KMLB20260904_210412_V06</Key></Contents>
  <Contents><Key>2026/09/04/KMLB/KMLB20260904_210851_V06</Key></Contents>
</ListBucketResult>"""
keys, token = parse_listing(xml)
check("listing yields its keys", len(keys) == 2 and keys[0].endswith("_V06"))
check("a truncated listing yields a continuation token", token == "abc123")
plain = xml.replace("<IsTruncated>true</IsTruncated>",
                    "<IsTruncated>false</IsTruncated>")
check("a complete listing yields no token", parse_listing(plain)[1] is None)
when = datetime(2026, 9, 4, 21, 4, 12)
check("NEXRAD prefix is date and site", nexrad_prefix("kmlb", when) == "2026/09/04/KMLB/")
check("GLM prefix uses day of year and hour",
      glm_prefix(when) == "GLM-L2-LCFA/2026/247/21/")
check("listing url carries the prefix",
      "prefix=2026%2F09%2F04%2FKMLB%2F" in listing_url("b", nexrad_prefix("KMLB", when)))

print("\nGLM footprints")
east, north = latlon_to_en(28.6083 + 0.01, -80.6041, 28.6083, -80.6041)
check("a degree offset north maps to positive north", float(north) > 0)
check("no offset maps to the origin",
      abs(float(latlon_to_en(28.6083, -80.6041, 28.6083, -80.6041)[0])) < 1e-6)

pts = footprint_points([28.61, 28.62, 28.63], [-80.60, -80.61, -80.62],
                       28.6083, -80.6041)
check("every group contributes a footprint point", len(pts) == 3)
ringed = dilate_footprint(pts, buffer_m=4000.0)
check("dilation adds a ring around each point", len(ringed) == 3 * 9)

f = Flash(time=when, points_en=ringed)
centroid_e = float(np.mean([p[0] for p in pts]))
centroid_n = float(np.mean([p[1] for p in pts]))
edge = f.min_distance_to(centroid_e + 30_000.0, centroid_n)
naive = float(np.hypot(30_000.0, 0.0))
check("footprint distance is shorter than centroid distance", edge < naive)

print("\nthermal profile")
prof = from_pairs([(0, 29.0), (4600, 0.0), (6600, -10.0), (8700, -20.0)])
check("0 C level is recovered", abs(prof.isotherm_altitude(0.0) - 4600) < 1)
check("-20 C level is recovered", abs(prof.isotherm_altitude(-20.0) - 8700) < 1)
check("all six criterion isotherms resolve",
      all(v is not None for v in isotherms(prof).values()))

# An inversion makes T(z) non-monotonic; the lowest crossing is conservative.
inv = from_pairs([(0, 2.0), (500, 8.0), (1500, 2.0), (4600, -20.0)])
z = inv.isotherm_altitude(0.0)
check("with an inversion the lowest 0 C crossing is taken", z is not None and z > 1500)

sounding = """-----------------------------------------------------------------------------
   PRES   HGHT   TEMP   DWPT   RELH   MIXR   DRCT   SKNT   THTA   THTE   THTV
    hPa     m      C      C      %    g/kg    deg   knot     K      K      K
-----------------------------------------------------------------------------
 1013.0     10   28.0   23.0     75  18.00    100     10  301.0  353.0  304.0
  850.0   1500   17.0   12.0     73  10.00    120     15  304.0  334.0  306.0
  500.0   5900   -8.0  -20.0     40   1.00    240     30  320.0  330.0  321.0
  300.0   9600  -35.0  -50.0     20   0.10    260     50  340.0  342.0  340.0
"""
p = parse_uwyo_sounding(sounding)
check("a Wyoming sounding parses", len(p.levels) == 4)
check("sounding 0 C level falls between its bracketing levels",
      1500 < p.isotherm_altitude(0.0) < 5900)

print(f"\n{'all ingest tests passed' if not FAILURES else str(len(FAILURES)) + ' FAILURES: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
