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
from scipy.ndimage import uniform_filter1d

EARTH_RADIUS_M = 6_371_000.0
REFRACTION_K = 4.0 / 3.0          # standard atmosphere effective radius
BEAMWIDTH_DEG = 0.925             # WSR-88D half-power beamwidth
# Back-calculated from the commonly quoted WSR-88D figure of about
# -7.5 dBZ at 50 km: -7.5 - 20*log10(50) = -41.5 at 1 km. An earlier value of
# -32 put 0 dBZ below the noise floor beyond ~25 km, which marked the entire
# grid unobservable when the radar sits 55 km from the pad. Site-specific;
# calibrate per radar if you have the figure.
WSR88D_MDS_DBZ_AT_1KM = -41.5


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

    # Envelope between the lowest and highest tilt rather than proximity to
    # any single beam centre. Above about 4 km the gaps between tilts exceed
    # the beam width, and testing each beam separately punches holes through
    # volume the gridding interpolates across perfectly well. The two real
    # failure modes are what this keeps: below the lowest tilt is overshoot,
    # above the highest is the cone of silence.
    half = beam_width(slant) / 2.0
    floor = beam_height(slant, min(elevations_deg), radar_alt_m) - half
    ceiling = beam_height(slant, max(elevations_deg), radar_alt_m) + half
    return ok & (up >= floor) & (up <= ceiling)


def phidp_texture(phidp, window: int = 9):
    """Gate-to-gate variability of differential phase along each ray.

    The single strongest non-meteorological discriminator available in Level
    II. Precipitation produces a smoothly varying PhiDP; ground clutter,
    anomalous propagation, biota and RF interference produce an erratic one.
    Computed as a windowed mean absolute gate-to-gate difference, which is
    far less sensitive to phase wrapping than a windowed standard deviation.
    """
    phi = np.asarray(phidp, dtype=float)
    diff = np.diff(phi, axis=-1)
    diff = (diff + 180.0) % 360.0 - 180.0        # unwrap to [-180, 180)
    diff = np.abs(np.nan_to_num(diff, nan=0.0))
    pad = np.concatenate([diff[..., :1], diff], axis=-1)
    # uniform_filter1d preserves length; np.convolve with mode="same" returns
    # the longer of its two inputs, which silently reshapes short rays.
    window = max(1, min(window, pad.shape[-1]))
    return uniform_filter1d(pad, size=window, axis=-1, mode="nearest")


def meteorological_mask(refl, rho_hv=None, zdr=None, phidp=None, velocity=None,
                        heights_m=None, radar_alt_m: float = 0.0,
                        freezing_level_m: float | None = None,
                        rho_min: float = 0.85,
                        rho_min_surface: float = 0.93,
                        rho_min_melting: float = 0.80,
                        surface_agl_m: float = 1200.0,
                        zdr_max: float = 5.0,
                        texture_max_deg: float = 12.0,
                        velocity_min_ms: float = 0.5,
                        refl_min: float = -30.0,
                        report: dict | None = None) -> np.ndarray:
    """LLCCR 29b: keep only gates plausibly due to a meteorological target.

    The standard's cloud boundary is the 0 dBZ contour, so anything that
    survives this filter becomes cloud -- which makes the filter load-bearing
    rather than cosmetic. Five tests, each degrading gracefully when its
    input is absent:

      Correlation coefficient, with a **height-dependent threshold**. A
      single global rho cut cannot work: ground clutter and AP sit near the
      surface and need a strict cut, while the melting layer legitimately
      depresses rho to 0.85-0.95 and needs a relaxed one. A flat 0.85 is
      simultaneously too loose near the ground and too tight in the bright
      band.

      Differential phase texture. Erratic PhiDP is the clearest signature of
      clutter, AP, biota and interference.

      Near-zero radial velocity combined with unremarkable rho: ground
      targets do not move. Only applied where rho is not already convincing,
      so genuine zero-Doppler precipitation is not discarded.

      Differential reflectivity, for large erratic returns from biota.

      A finite-value and floor check.

    Pass `report` to receive per-test rejection counts, which is how you find
    out whether a filter is doing its job or eating your weather.
    """
    refl = np.asarray(refl, dtype=float)
    keep = np.isfinite(refl) & (refl >= refl_min)
    counts = {"total": int(refl.size), "not_finite_or_floor": int((~keep).sum())}

    if rho_hv is not None:
        rho = np.asarray(rho_hv, dtype=float)
        threshold = np.full(rho.shape, rho_min, dtype=float)
        if heights_m is not None:
            agl = np.asarray(heights_m, dtype=float) - radar_alt_m
            threshold = np.where(agl <= surface_agl_m, rho_min_surface, threshold)
            if freezing_level_m is not None:
                melting = ((np.asarray(heights_m) >= freezing_level_m - 1500.0)
                           & (np.asarray(heights_m) <= freezing_level_m + 400.0))
                threshold = np.where(melting, rho_min_melting, threshold)
        bad = ~(np.isfinite(rho) & (rho >= threshold))
        counts["rho_hv"] = int((keep & bad).sum())
        keep &= ~bad

    if phidp is not None:
        texture = phidp_texture(phidp)
        bad = ~np.isfinite(texture) | (texture > texture_max_deg)
        counts["phidp_texture"] = int((keep & bad).sum())
        keep &= ~bad

    if velocity is not None:
        vel = np.abs(np.asarray(velocity, dtype=float))
        stationary = np.isfinite(vel) & (vel < velocity_min_ms)
        if rho_hv is not None:
            rho = np.asarray(rho_hv, dtype=float)
            stationary &= ~(np.isfinite(rho) & (rho >= 0.97))
        counts["stationary"] = int((keep & stationary).sum())
        keep &= ~stationary

    if zdr is not None:
        z = np.asarray(zdr, dtype=float)
        bad = ~(np.isfinite(z) & (np.abs(z) <= zdr_max))
        counts["zdr"] = int((keep & bad).sum())
        keep &= ~bad

    counts["kept"] = int(keep.sum())
    if report is not None:
        report.update(counts)
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
