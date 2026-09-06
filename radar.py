"""Gridded reflectivity for the display.

The raster the operator sees must be the same field the segmentation ran on,
at the same valid time. If the picture and the verdict can disagree, the
picture is worse than no picture: it invites the officer to trust a boundary
the evaluator never used. So this ships the evaluator's own grid rather than
fetching tiles from a third-party service.

Encoding is base64 int8 dBZ, which keeps a 128x128 plan view at about 22 kB
and needs no image library on either end. Two sentinels:

    UNOBSERVABLE   the radar cannot see here -- beam overshoot, cone of
                   silence, or 0 dBZ below minimum detectable signal at
                   this range. Renders hatched, never as clear air.
    NO_ECHO        observed, and nothing there.

That distinction is the whole reason this is a ternary field and not a
binary one. A display that paints unobservable volume the same as clear air
is actively misleading, because the standard's cloud boundary is defined by
the 0 dBZ contour and an unobserved region has no contour.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field

UNOBSERVABLE = -128
NO_ECHO = -127
MIN_DBZ = -30.0
MAX_DBZ = 75.0


def encode(values: list[int]) -> str:
    return base64.b64encode(bytes((v + 256) % 256 for v in values)).decode("ascii")


def decode(payload: str) -> list[int]:
    return [b - 256 if b > 127 else b for b in base64.b64decode(payload)]


@dataclass
class Raster:
    nx: int
    ny: int
    values: list[int]                      # row-major, origin top-left
    kind: str = "plan"                     # "plan" | "xsec"
    extent: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "nx": self.nx,
            "ny": self.ny,
            "extent": self.extent,
            "encoding": "base64-int8-dbz",
            "unobservable": UNOBSERVABLE,
            "no_echo": NO_ECHO,
            "data": encode(self.values),
        }


@dataclass
class RadarField:
    valid_time: str
    source: str
    plan: Raster
    xsec: Raster
    plan_mode: str = "composite"           # "composite" | "cappi"
    plan_level_km: float | None = None

    def to_dict(self) -> dict:
        return {
            "valid_time": self.valid_time,
            "source": self.source,
            "plan_mode": self.plan_mode,
            "plan_level_km": self.plan_level_km,
            "plan": self.plan.to_dict(),
            "xsec": self.xsec.to_dict(),
        }


def from_grids(valid_time: str, source: str,
               plan_values, plan_extent: dict,
               xsec_values, xsec_extent: dict,
               plan_mode: str = "composite",
               plan_level_km: float | None = None) -> RadarField:
    """Build a RadarField from real gridded output.

    `plan_values` and `xsec_values` are 2D sequences (rows of dBZ), already
    quantised to integers, using UNOBSERVABLE and NO_ECHO for the two
    non-numeric states. Pass the same arrays the segmentation consumed.
    """
    def flatten(rows) -> tuple[int, int, list[int]]:
        ny = len(rows)
        nx = len(rows[0]) if ny else 0
        flat = [int(v) for row in rows for v in row]
        return nx, ny, flat

    pnx, pny, pflat = flatten(plan_values)
    xnx, xny, xflat = flatten(xsec_values)
    return RadarField(
        valid_time=valid_time,
        source=source,
        plan=Raster(pnx, pny, pflat, "plan", plan_extent),
        xsec=Raster(xnx, xny, xflat, "xsec", xsec_extent),
        plan_mode=plan_mode,
        plan_level_km=plan_level_km,
    )


# --------------------------------------------------------------------------
# Synthetic field for scenarios.
#
# Generated FROM the object geometry so the raster and the objects agree by
# construction. That is the correct relationship for a demo: in the real
# system the causality runs the other way, with objects segmented out of the
# measured field, but either way the two must never disagree.
# --------------------------------------------------------------------------

_PEAK_DBZ = {
    "cumulus": 46.0,
    "smoke_cumulus": 30.0,
    "attached_anvil": 16.0,
    "detached_anvil": 12.0,
    "debris": 22.0,
    "thick_layer": 24.0,
    "cirriform": 4.0,
    "unclassified": 20.0,
}

# KMLB relative to LC-39A: about 30 nmi on a bearing of 200 degrees.
RADAR_BEARING_DEG = 200.0
RADAR_RANGE_NMI = 30.0


def _radar_xy(max_nmi: float) -> tuple[float, float]:
    a = math.radians(RADAR_BEARING_DEG)
    return RADAR_RANGE_NMI * math.sin(a), RADAR_RANGE_NMI * math.cos(a)


def synth_plan(objects: dict, nx: int = 128, ny: int = 128,
               half_nmi: float = 26.0) -> Raster:
    """Composite reflectivity over a square centred on the pad."""
    rx, ry = _radar_xy(half_nmi)
    values = []
    for j in range(ny):
        north = half_nmi - (2 * half_nmi) * (j + 0.5) / ny
        for i in range(nx):
            east = -half_nmi + (2 * half_nmi) * (i + 0.5) / nx

            best = NO_ECHO
            for o in objects.values():
                if o.get("range_nmi") is None or o.get("radius_nmi") is None:
                    continue
                a = math.radians(o.get("bearing_deg") or 0.0)
                ox = o["range_nmi"] * math.sin(a)
                oy = o["range_nmi"] * math.cos(a)
                r = o["radius_nmi"]
                d = math.hypot(east - ox, north - oy) / r
                if d > 1.35:
                    continue
                peak = _PEAK_DBZ.get(o.get("cloud_type", "unclassified"), 20.0)
                val = peak * math.exp(-1.6 * d * d) - 6.0 * d
                if val > (best if best != NO_ECHO else MIN_DBZ):
                    best = val

            # Beam overshoot: past this range from KMLB the lowest tilt is
            # above low cloud, so absence of echo proves nothing.
            if math.hypot(east - rx, north - ry) > 52.0:
                values.append(UNOBSERVABLE)
                continue
            # Cone of silence directly over the radar.
            if math.hypot(east - rx, north - ry) < 2.2:
                values.append(UNOBSERVABLE)
                continue

            values.append(NO_ECHO if best == NO_ECHO
                          else max(-30, min(75, int(round(best)))))
    return Raster(nx, ny, values, "plan", {"half_nmi": half_nmi})


def synth_xsec(objects: dict, nx: int = 128, ny: int = 72,
               max_nmi: float = 25.0, max_km: float = 18.0) -> Raster:
    """Vertical slice along the flight azimuth."""
    values = []
    for j in range(ny):
        alt = max_km - max_km * (j + 0.5) / ny
        for i in range(nx):
            down = max_nmi * (i + 0.5) / nx

            best = NO_ECHO
            for o in objects.values():
                if o.get("downrange_nmi") is None or o.get("top_alt_km") is None:
                    continue
                base = o.get("base_alt_km") or 0.0
                top = o["top_alt_km"]
                r = o.get("radius_nmi") or 1.0
                dh = abs(down - o["downrange_nmi"]) / r
                if dh > 1.3 or not (base - 0.3 <= alt <= top + 0.3):
                    continue
                span = max(0.4, top - base)
                dv = abs(alt - (base + top) / 2) / (span / 2)
                peak = _PEAK_DBZ.get(o.get("cloud_type", "unclassified"), 20.0)
                val = peak * math.exp(-1.1 * (dh * dh + 0.55 * dv * dv)) - 4.0 * dh
                if val > (best if best != NO_ECHO else MIN_DBZ):
                    best = val

            # Above the highest usable tilt at range, nothing is sampled.
            if alt > 3.0 + down * 0.62:
                values.append(UNOBSERVABLE)
                continue

            values.append(NO_ECHO if best == NO_ECHO
                          else max(-30, min(75, int(round(best)))))
    return Raster(nx, ny, values, "xsec",
                  {"max_nmi": max_nmi, "max_km": max_km})


def synth_field(snapshot_objects: dict, valid_time: str,
                source: str = "KMLB Level II (synthetic)") -> RadarField:
    return RadarField(
        valid_time=valid_time,
        source=source,
        plan=synth_plan(snapshot_objects),
        xsec=synth_xsec(snapshot_objects),
    )
