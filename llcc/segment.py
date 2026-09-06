"""Segmentation of the reflectivity field into cloud objects.

The important decision here is that **connected components are not objects.**

Section 4.3 requires that when clouds become physically connected they are
still assessed individually, until the individual clouds are no longer
distinguishable, and 4.3a through 4.3d mandate independent assessment for
the common pairings. A raw connected-component label dissolves the
individuals into one blob at exactly the moment the standard says not to.

So the two concepts are separated:

    objects      watershed basins seeded on reflectivity cores. Identity.
    connections  a graph edge between objects sharing a 0 dBZ component.

A single component may then hold three pairwise-connected objects, which is
precisely the situation 4.3e describes and lets it be implemented literally.

The field is ternary throughout: cloud, clear, and unobservable. Absence of
echo where the radar cannot see is not clear air, and must never be
segmented as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

CLOUD_DBZ = 0.0        # the standard's non-transparency threshold
SEED_DBZ = 15.0        # hysteresis seed; grow down to CLOUD_DBZ
PRECIP_DBZ = 18.0      # section 4.2.2a(3)
MODERATE_DBZ = 30.0    # section 4.2.2b(3)


@dataclass
class Segment:
    """One watershed basin: a cloud object at a single valid time."""

    label: int
    mask: np.ndarray
    component: int
    voxels: int
    top_level: int
    base_level: int
    max_refl: float
    centroid: tuple[float, float, float]
    touches_edge: bool
    observability: float = 1.0

    def top_altitude(self, grid) -> float:
        return grid.z0 + (self.top_level + 1) * grid.dz

    def base_altitude(self, grid) -> float:
        return grid.z0 + self.base_level * grid.dz


@dataclass
class Segmentation:
    labels: np.ndarray
    segments: dict[int, Segment]
    components: np.ndarray
    connections: list[tuple[int, int]] = field(default_factory=list)
    dropped: int = 0

    def cluster_of(self, label: int) -> int:
        return self.segments[label].component


def cloud_field(refl: np.ndarray, observable: np.ndarray | None = None,
                cloud_dbz: float = CLOUD_DBZ) -> np.ndarray:
    """The non-transparent volume: observed reflectivity at or above 0 dBZ.

    Unobservable voxels are excluded rather than treated as clear, which is
    the whole reason the input is ternary.
    """
    field_ = np.isfinite(refl) & (refl >= cloud_dbz)
    if observable is not None:
        field_ &= observable
    return field_


def segment_field(refl: np.ndarray, grid, observable: np.ndarray | None = None,
                  cloud_dbz: float = CLOUD_DBZ, seed_dbz: float = SEED_DBZ,
                  min_separation_m: float = 6000.0,
                  min_voxels: int = 24,
                  min_footprint_cells: int = 6,
                  min_levels: int = 2,
                  min_neighbours: int = 7) -> Segmentation:
    """Hysteresis threshold, then watershed into objects, then build the
    connection graph.

    Seeding on cores rather than on every local bump is the hysteresis: a
    hard 0 dBZ cut makes objects blink in and out at range, and every blink
    would look like a split or a merge and spuriously start a three-hour
    clock. A component with no core still becomes one object, because thin
    anvil and cirrus never reach the seed threshold but are still clouds.
    """
    cloud = cloud_field(refl, observable, cloud_dbz)

    # Spatial coherence first, and this one matters more than the size
    # filter. Isolated gates just above 0 dBZ survive polarimetric QC, and
    # scattered through a volume they link up into one percolating network
    # that spans the domain -- a single "cloud" tens of miles across made of
    # noise. Requiring a voxel to have company in its 3x3x3 neighbourhood
    # breaks those filaments without eroding real cloud: a voxel inside a
    # one-level-thick sheet still has nine, so thin anvil and cirrus survive
    # a threshold of seven, while an isolated gate has one and a filament
    # has three.
    #
    # LLCCR 33c requires allowance for the radar's spatial resolution when
    # computing a cloud boundary, which is the licence for doing this at all.
    if min_neighbours > 1:
        counts = ndimage.uniform_filter(cloud.astype(np.float32), size=3,
                                        mode="constant") * 27.0
        cloud &= counts >= min_neighbours

    # Then drop specks before labelling. On real data the 0 dBZ field is peppered
    # with residual clutter, sea return and isolated gates that survive
    # polarimetric QC. Each one would otherwise become a "cloud" carrying its
    # own standoff buffer and lineage, which is both meaningless under the
    # standard and expensive: every object costs full-grid geometry.
    if min_voxels > 1:
        pre, n_pre = ndimage.label(cloud)
        if n_pre:
            sizes = np.bincount(pre.ravel())
            sizes[0] = 0
            cloud &= np.isin(pre, np.flatnonzero(sizes >= min_voxels))

    components, n_components = ndimage.label(cloud)

    if n_components == 0:
        return Segmentation(np.zeros_like(components), {}, components, [])

    filled = np.where(cloud, refl, -np.inf)
    sep = max(1, int(round(min_separation_m / min(grid.dx, grid.dy))))

    coords = peak_local_max(np.where(cloud, refl, -1e9), min_distance=sep,
                            threshold_abs=seed_dbz, labels=cloud)
    markers = np.zeros(cloud.shape, dtype=np.int32)
    for n, (i, j, k) in enumerate(coords, start=1):
        markers[i, j, k] = n

    # Any component with no core gets one marker at its own maximum, so
    # non-convective cloud is not silently dropped.
    next_marker = int(markers.max()) + 1
    seeded = set(np.unique(components[markers > 0])) - {0}
    for cid in range(1, n_components + 1):
        if cid in seeded:
            continue
        sel = components == cid
        flat = np.argmax(np.where(sel, filled, -np.inf))
        markers[np.unravel_index(flat, cloud.shape)] = next_marker
        next_marker += 1

    labels = watershed(-np.where(cloud, refl, -1e9), markers, mask=cloud)

    segments: dict[int, Segment] = {}
    dropped = 0
    for lab in np.unique(labels):
        if lab == 0:
            continue
        mask = labels == lab
        idx = np.argwhere(mask)
        levels = idx[:, 2]

        # A basin must be big enough to be a cloud rather than a fragment.
        # Footprint and depth are separate tests: a wide shallow deck and a
        # narrow deep tower are both real, a two-gate speck is neither.
        footprint = int(mask.any(axis=2).sum())
        depth = int(levels.max() - levels.min() + 1)
        if (mask.sum() < min_voxels or footprint < min_footprint_cells
                or depth < min_levels):
            labels[mask] = 0
            dropped += 1
            continue

        comp = int(components[tuple(idx[0])])
        touches = bool(
            idx[:, 0].min() == 0 or idx[:, 0].max() == grid.nx - 1
            or idx[:, 1].min() == 0 or idx[:, 1].max() == grid.ny - 1
        )
        segments[int(lab)] = Segment(
            label=int(lab), mask=mask, component=comp, voxels=int(mask.sum()),
            top_level=int(levels.max()), base_level=int(levels.min()),
            max_refl=float(np.nanmax(refl[mask])),
            centroid=tuple(float(v) for v in idx.mean(axis=0)),
            touches_edge=touches,
        )

    connections = []
    by_component: dict[int, list[int]] = {}
    for lab, seg in segments.items():
        by_component.setdefault(seg.component, []).append(lab)
    for members in by_component.values():
        members.sort()
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                connections.append((members[a], members[b]))

    seg = Segmentation(labels, segments, components, connections)
    seg.dropped = dropped
    return seg


def footprint_raster(labels: np.ndarray, order: list[int]) -> np.ndarray:
    """2D map of which object occupies each column, for the display.

    Objects are drawn from their actual footprint rather than as a circle of
    equivalent area. A sprawling object and a compact one of the same area
    demand very different standoffs in practice, and a circle hides that.
    """
    out = np.zeros(labels.shape[:2], dtype=np.uint8)
    for n, lab in enumerate(order[:250], start=1):
        out[(labels == lab).any(axis=2)] = n
    return out


def observability(mask: np.ndarray, observable: np.ndarray | None) -> float:
    """Fraction of an object's bounding volume the radar could actually see.

    A low value is what drives an indeterminate result rather than a verdict:
    the segmentation is only as trustworthy as its coverage.
    """
    if observable is None:
        return 1.0
    idx = np.argwhere(mask)
    if not len(idx):
        return 0.0
    lo, hi = idx.min(axis=0), idx.max(axis=0) + 1
    box = observable[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return float(box.mean()) if box.size else 0.0


def precipitation_flags(refl: np.ndarray, mask: np.ndarray) -> tuple[bool, bool]:
    """Section 4.2.2a(3) and 4.2.2b(3), the radar detection methods."""
    if not mask.any():
        return False, False
    peak = float(np.nanmax(refl[mask]))
    return peak >= PRECIP_DBZ, peak >= MODERATE_DBZ


def bright_band_field(refl: np.ndarray, grid, freezing_level_m: float | None,
                      enhancement_db: float = 5.0,
                      band_m: float = 1200.0) -> np.ndarray:
    """Melting layer signature in reflectivity alone, as a 2D mask.

    LLCCR 21b names a radar bright band explicitly. Frozen hydrometeors
    beginning to melt scatter like large wet particles, producing a
    horizontally extensive reflectivity maximum in a shallow layer just below
    the 0 C level. Comparing that band against the layer above it isolates
    the enhancement.

    Polarimetric detection on the polar gates is stronger -- depressed rho_hv
    with enhanced ZDR -- but this works on the gridded field the segmentation
    already has, and it degrades to "no bright band" rather than to a wrong
    answer when the freezing level is unknown.
    """
    if freezing_level_m is None:
        return np.zeros((grid.nx, grid.ny), dtype=bool)

    k_bb0 = grid.level_at(freezing_level_m - band_m)
    k_bb1 = grid.level_at(freezing_level_m)
    k_up1 = grid.level_at(freezing_level_m + band_m)
    if k_bb1 <= k_bb0 or k_up1 <= k_bb1:
        return np.zeros((grid.nx, grid.ny), dtype=bool)

    with np.errstate(invalid="ignore"):
        band = np.nanmax(refl[:, :, k_bb0:k_bb1], axis=2)
        above = np.nanmax(refl[:, :, k_bb1:k_up1], axis=2)
    band = np.where(np.isfinite(band), band, -np.inf)
    above = np.where(np.isfinite(above), above, -np.inf)
    return np.isfinite(band) & (band - above >= enhancement_db) & (band >= 15.0)
