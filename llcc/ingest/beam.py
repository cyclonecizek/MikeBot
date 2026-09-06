"""Radar beam geometry, sensitivity, and dual-pol quality control.

Everything here is a pure function of arrays, so it is testable without a
radar file, a network, or Py-ART. The fetch-and-decode layer in `nexrad.py`
is deliberately thin for exactly that reason: the parts that can be wrong in
subtle, safety-relevant ways live here instead.

Two things this module exists to produce:

  The observability mask. Section 4.2.1d says a measurement inside the cone
  of silence or a blocked sector cannot be used unless the absence of cloud
  is established some other way. Beyond the range where 0 dBZ falls below
  minimum detectable signal, absence of echo proves nothing either. Both
  become UNOBSERVABLE rather than clear air.

  Meteorological-target QC. LLCCR 29b requires that a reflectivity
  measurement be due to a meteorological target. Florida boundary layers are
  full of insects and birds returning well above 0 dBZ, and the standard's
  cloud boundary is the 0 dBZ contour, so this filter is load-bearing rather
  than cosmetic.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
REFRACTION_K = 4.0 / 3.0          # standard atmosphere effective radius
BEAMWIDTH_DEG = 0.925             # WSR-88D half-power beamwidth
WSR88D_MDS_DBZ_AT_1KM = -32.0     # typical; site-specific, calibrate per radar


def beam_height(range_m, elevation_deg, radar_alt_m: float = 0.0):
    """Height above mean sea level of the beam centre.

    Four-thirds effective earth radius. The curvature term is what makes a
    distant low tilt overshoot low cloud, which is the dominant reason a
    scan cannot see something near the ground downrange.
    """
    r = np.asarray(range_m, dtype=float)
    theta = np.radians(elevation_deg)
    ke_a = REFRACTION_K * EARTH_RADIUS_M
    return (np.sqrt(r ** 2 + ke_a ** 2 + 2 * r * ke_a * np.sin(theta))
            - ke_a + radar_alt_m)


def beam_width(range_m, beamwidth_deg: float = BEAMWIDTH_DEG):
    """Vertical extent of the beam at range, in metres.

    At 100 km this is about 1.6 km, which quantises echo tops and therefore
    quantises cloud-top temperature. In a Florida summer profile 1.6 km is
    roughly 10 C -- the whole gap between the -10 C and -20 C criterion
    tiers, so this number should propagate into the display as a band.
    """
    return np.asarray(range_m, dtype=float) * np.radians(beamwidth_deg)


def minimum_detectable_dbz(range_m, mds_at_1km: float = WSR88D_MDS_DBZ_AT_1KM):
    """Sensitivity falls off as 20*log10(range)."""
    r_km = np.maximum(np.asarray(range_m, dtype=float) / 1000.0, 1e-3)
    return mds_at_1km + 20.0 * np.log10(r_km)


def detectable(range_m, threshold_dbz: float = 0.0,
               mds_at_1km: float = WSR88D_MDS_DBZ_AT_1KM):
    """True where `threshold_dbz` is above the noise floor.

    Past the range where this goes false, an empty gate does not mean clear
    air -- it means the radar could not have seen a 0 dBZ cloud even if one
    were there.
    """
    return minimum_detectable_dbz(range_m, mds_at_1km) <= threshold_dbz


def observability_mask(grid, radar_east: float, radar_north: float,
                       radar_alt_m: float = 0.0,
                       elevations_deg=(0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0, 5.1,
                                       6.4, 8.0, 10.0, 12.5, 15.6, 19.5),
                       max_range_m: float = 230_000.0,
                       threshold_dbz: float = 0.0,
                       mds_at_1km: float = WSR88D_MDS_DBZ_AT_1KM,
                       blocked_sectors=()) -> np.ndarray:
    """Which grid voxels this radar could actually interrogate.

    A voxel is observable when it lies inside some tilt's beam, is within
    unambiguous range, and sits where `threshold_dbz` clears the noise
    floor. The cone of silence falls out of the geometry: above the highest
    elevation angle there is no beam, so no tilt covers those voxels.
    """
    east = grid.east()[:, None, None]
    north = grid.north()[None, :, None]
    up = grid.up()[None, None, :]

    de = east - radar_east
    dn = north - radar_north
    ground = np.hypot(de, dn)
    slant = np.sqrt(ground ** 2 + (up - radar_alt_m) ** 2)

    ok = (slant <= max_range_m) & detectable(slant, threshold_dbz, mds_at_1km)

    if blocked_sectors:
        az = (np.degrees(np.arctan2(de, dn))) % 360.0
        for lo, hi in blocked_sectors:
            ok &= ~((az >= lo) & (az <= hi))

    half = beam_width(slant) / 2.0
    covered = np.zeros(grid.shape, dtype=bool)
    for elev in elevations_deg:
        centre = beam_height(slant, elev, radar_alt_m)
        covered |= np.abs(up - centre) <= half
    return ok & covered


def meteorological_mask(refl, rho_hv=None, zdr=None,
                        rho_min: float = 0.85,
                        zdr_max: float = 6.0,
                        refl_min: float = -30.0) -> np.ndarray:
    """LLCCR 29b: keep only gates plausibly due to a meteorological target.

    Biological scatterers -- insects, birds, bats -- return low correlation
    coefficient and large, erratic differential reflectivity. Ground clutter
    and anomalous propagation share the low-rho signature. Filtering on
    polarimetry before thresholding on reflectivity keeps the 0 dBZ contour
    from being drawn around a swarm of insects.

    With no dual-pol input the mask degrades to a finite-value check, which
    is honest but much weaker: say so in the output rather than implying the
    QC ran.
    """
    keep = np.isfinite(refl) & (refl >= refl_min)
    if rho_hv is not None:
        keep &= np.isfinite(rho_hv) & (rho_hv >= rho_min)
    if zdr is not None:
        keep &= np.isfinite(zdr) & (np.abs(zdr) <= zdr_max)
    return keep


def bright_band(rho_hv, zdr, refl, freezing_level_m, altitudes,
                depth_m: float = 1000.0,
                rho_max: float = 0.97, zdr_min: float = 1.0,
                refl_min: float = 20.0) -> np.ndarray:
    """Melting layer signature, which LLCCR 21b names explicitly.

    Frozen hydrometeors beginning to melt depress rho_hv and enhance ZDR in
    a shallow layer near the 0 C level. Restricting the search to that layer
    is what separates a bright band from ordinary low-rho clutter.
    """
    alt = np.asarray(altitudes)
    in_layer = (alt >= freezing_level_m - depth_m) & (alt <= freezing_level_m + 200.0)
    sig = (np.isfinite(rho_hv) & (rho_hv <= rho_max)
           & np.isfinite(zdr) & (zdr >= zdr_min)
           & np.isfinite(refl) & (refl >= refl_min))
    shape = [1] * sig.ndim
    shape[-1] = -1
    return sig & in_layer.reshape(shape)


def attenuation_budget(refl_along_path, gate_length_m: float,
                       wavelength_cm: float = 10.0) -> float:
    """Rough two-way path-integrated attenuation in dB.

    Section 4.2.1c requires that a measurement not be significantly
    attenuated by intervening precipitation. KMLB is S-band, where this is
    small, but the check matters if a shorter-wavelength radar is ever
    brought in -- 4.2.1a imposes extra conditions below 5 cm for exactly
    this reason.
    """
    z = 10.0 ** (np.asarray(refl_along_path, dtype=float) / 10.0)
    rain_mm_hr = (z / 200.0) ** (1.0 / 1.6)
    coeff = {10.0: 0.0003, 5.0: 0.0018, 3.0: 0.011}
    a = min(coeff, key=lambda w: abs(w - wavelength_cm))
    return float(2.0 * coeff[a] * np.nansum(rain_mm_hr) * gate_length_m / 1000.0)
