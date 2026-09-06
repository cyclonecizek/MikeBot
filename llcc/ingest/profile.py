"""Thermal profile ingest.

Every threshold in the standard is a temperature; every radar measurement is
an altitude. This module produces the mapping.

Since cloud edges come from the 0 dBZ contour rather than from satellite
cloud-top temperature, this is load-bearing for *every* cloud-top criterion,
not just the radar-side ones.
"""

from __future__ import annotations

from ..world import ThermalProfile

# Cape Canaveral (XMR / WMO 74794): the sounding of record for the range.
XMR = {"station": "XMR", "wmo": 74794, "lat": 28.4667, "lon": -80.55}


def from_pairs(pairs, uncertainty_m: float = 300.0) -> ThermalProfile:
    """Build a profile from (altitude_m, temperature_c) pairs.

    `ThermalProfile.isotherm_altitude` returns the LOWEST crossing, which
    matters because T(z) is not monotonic -- morning inversions on the Cape
    are routine. The lowest crossing maximises the volume counted as colder
    than the threshold, which is the conservative direction.
    """
    levels = sorted((float(z), float(t)) for z, t in pairs)
    if len(levels) < 2:
        raise ValueError("a profile needs at least two levels")
    return ThermalProfile(levels=levels, uncertainty_m=uncertainty_m)


def parse_uwyo_sounding(text: str, uncertainty_m: float = 300.0) -> ThermalProfile:
    """Parse a University of Wyoming text sounding.

    Fixed-width columns: PRES HGHT TEMP DWPT ... Rows with missing height or
    temperature are skipped rather than interpolated, since a fabricated
    level would silently move an isotherm.
    """
    pairs = []
    for line in text.splitlines():
        if len(line) < 21:
            continue
        height, temp = line[7:14].strip(), line[14:21].strip()
        if not height or not temp:
            continue
        try:
            pairs.append((float(height), float(temp)))
        except ValueError:
            continue
    if len(pairs) < 2:
        raise ValueError("no usable levels found in sounding text")
    return from_pairs(pairs, uncertainty_m)


def parse_rap_column(pressures_pa, heights_m, temps_k,
                     uncertainty_m: float = 200.0) -> ThermalProfile:
    """Build a profile from a model column (RAP or HRRR native levels).

    HRRR at 3 km is the better primary source, but there is a real argument
    for taking isotherm heights from MRMS instead: not because they are more
    accurate, but because they are consistent with the grid the reflectivity
    is already on. A 300 m disagreement in the 0 C level moves the bottom of
    every MRR volume.
    """
    pairs = [(float(z), float(t) - 273.15) for z, t in zip(heights_m, temps_k)]
    return from_pairs(pairs, uncertainty_m)


def isotherms(profile: ThermalProfile) -> dict[float, float | None]:
    """The six levels the criteria actually reference."""
    return {t: profile.isotherm_altitude(t) for t in (5.0, 0.0, -5.0, -10.0,
                                                      -15.0, -20.0)}
