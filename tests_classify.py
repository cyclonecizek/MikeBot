"""Classifier tests.

The taxonomy turns on lineage as much as on shape, so these build both:
synthetic reflectivity fields for the shape tests, and stub tracks with
parents for the lineage tests.

The most important assertion here is the last one. Anything the classifier
cannot place confidently must come back cumulus, because cumulus carries the
widest standoffs in section 4.1. The classifier exists to relax the
worst-case assumption where it can positively identify something, never to
introduce a new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from llcc.classify import Thresholds, classify, extract
from llcc.geometry import Grid
from llcc.world import CloudType, ThermalProfile

FAILURES: list[str] = []
GRID = Grid(nx=64, ny=64, nz=40, dx=1000.0, dy=1000.0, dz=500.0)
PROFILE = ThermalProfile(levels=[
    (0, 29.0), (1000, 22.0), (2000, 15.5), (3000, 9.0), (3600, 5.0),
    (4600, 0.0), (5600, -5.0), (6600, -10.0), (7600, -15.0), (8700, -20.0),
    (10000, -33.0), (12000, -52.0), (14000, -68.0), (16000, -80.0),
    (20000, -70.0),
])


def check(name, condition):
    if not condition:
        FAILURES.append(name)
    print(f"  {'pass' if condition else 'FAIL'}  {name}")


@dataclass
class StubTrack:
    coldest_top_c: float | None = None
    detached_at: datetime | None = None
    parent_ids: list = None


def box(ci, cj, half, k0, k1, dbz):
    """A slab of uniform reflectivity, plus its mask."""
    refl = np.full(GRID.shape, np.nan)
    mask = np.zeros(GRID.shape, dtype=bool)
    mask[ci - half:ci + half, cj - half:cj + half, k0:k1] = True
    refl[mask] = dbz
    return refl, mask


def feats(refl, mask, bright=False, coldest=None):
    f = extract(refl, mask, GRID, PROFILE, bright_band=bright)
    f.coldest_top_c = coldest
    return f


print("shape: convective versus stratiform")
# A tower: narrow, deep, strong core carried well above the freezing level.
refl, mask = box(32, 32, 4, GRID.level_at(1000), GRID.level_at(11000), 45.0)
v = classify(feats(refl, mask))
check("a strong core above the 0 C level is cumulus",
      v.cloud_type == CloudType.CUMULUS and "0 C level" in v.reason)

# A deck: broad, shallow, weak, sitting across the freezing level.
refl, mask = box(32, 32, 22, GRID.level_at(3000), GRID.level_at(4800), 22.0)
v = classify(feats(refl, mask))
check("a broad shallow weak layer is a thick cloud layer",
      v.cloud_type == CloudType.THICK_LAYER)

# A bright band settles it even for a smaller deck.
refl, mask = box(32, 32, 10, GRID.level_at(3200), GRID.level_at(4800), 24.0)
v = classify(feats(refl, mask, bright=True))
check("a bright band identifies a layer regardless of size",
      v.cloud_type == CloudType.THICK_LAYER and "bright band" in v.reason)

# Tall and narrow but weak still reads convective on aspect.
refl, mask = box(32, 32, 3, GRID.level_at(1500), GRID.level_at(9000), 24.0)
v = classify(feats(refl, mask))
check("a tall narrow tower is cumulus on aspect alone",
      v.cloud_type == CloudType.CUMULUS and "aspect" in v.reason)

print("\nshape: cirriform")
refl, mask = box(32, 32, 14, GRID.level_at(9000), GRID.level_at(11500), 6.0)
v = classify(feats(refl, mask))
check("cold, thin and weak with no ancestry is cirriform",
      v.cloud_type == CloudType.CIRRIFORM)
check("the same sheet with convective ancestry is NOT cirriform",
      classify(feats(refl, mask), parents=[StubTrack(coldest_top_c=-35.0)]
               ).cloud_type != CloudType.CIRRIFORM)

print("\nlineage outranks shape")
# The same high, weak sheet: anvil when it came from a cell, cirrus when not.
refl, mask = box(40, 40, 16, GRID.level_at(8000), GRID.level_at(11500), 18.0)
parent = StubTrack(coldest_top_c=-35.0)
v = classify(feats(refl, mask), track=StubTrack(), parents=[parent])
check("outflow from a cold parent is a detached anvil",
      v.cloud_type == CloudType.DETACHED_ANVIL)
v = classify(feats(refl, mask), track=StubTrack(), parents=[parent],
             connected=[CloudType.CUMULUS])
check("the same anvil connected to a cumulus is attached",
      v.cloud_type == CloudType.ATTACHED_ANVIL)
check("a warm parent does not make an anvil",
      classify(feats(refl, mask), parents=[StubTrack(coldest_top_c=-4.0)]
               ).cloud_type != CloudType.DETACHED_ANVIL)

# Debris: cold ancestry, detached, and the top has since collapsed.
refl, mask = box(20, 20, 6, GRID.level_at(1000), GRID.level_at(5600), 26.0)
v = classify(feats(refl, mask),
             track=StubTrack(detached_at=datetime(2026, 8, 26, 17, 0)),
             parents=[StubTrack(coldest_top_c=-32.0)])
check("a collapsed remnant of a deep cell is debris",
      v.cloud_type == CloudType.DEBRIS)
check("still-cold-topped outflow is anvil, not debris",
      classify(feats(*box(40, 40, 16, GRID.level_at(8000),
                          GRID.level_at(11500), 18.0)),
               track=StubTrack(detached_at=datetime(2026, 8, 26, 17, 0)),
               parents=[StubTrack(coldest_top_c=-32.0)]
               ).cloud_type == CloudType.DETACHED_ANVIL)

print("\nthe default is the restrictive one")
# Middling: not broad enough for a layer, not strong or tall enough for a
# core, not cold enough for cirrus.
refl, mask = box(32, 32, 7, GRID.level_at(2000), GRID.level_at(4000), 14.0)
v = classify(feats(refl, mask))
check("an unclassifiable object falls back to cumulus",
      v.cloud_type == CloudType.CUMULUS)
check("the fallback is flagged as not confident", not v.confident)
check("a confident call is flagged as such",
      classify(feats(*box(32, 32, 4, GRID.level_at(1000),
                          GRID.level_at(11000), 45.0))).confident)

print("\nthresholds are tunable, not from the standard")
tight = Thresholds(convective_dbz_aloft=50.0)
refl, mask = box(32, 32, 4, GRID.level_at(1000), GRID.level_at(11000), 45.0)
check("raising the convective threshold changes the call",
      classify(feats(refl, mask), thresholds=tight).reason
      != classify(feats(refl, mask)).reason)

print("\nfeature extraction")
refl, mask = box(32, 32, 10, GRID.level_at(2000), GRID.level_at(6000), 30.0)
f = extract(refl, mask, GRID, PROFILE)
check("depth matches the slab", 3800 <= f.depth_m <= 4200)
check("area matches the footprint", 380 <= f.area_km2 <= 420)
check("top temperature comes from the profile",
      f.top_temp_c is not None and -12 < f.top_temp_c < -4)
check("reflectivity aloft excludes what is below the 0 C level",
      f.max_dbz_aloft == 30.0)
refl2, mask2 = box(32, 32, 10, GRID.level_at(1000), GRID.level_at(4000), 30.0)
check("an entirely warm object has no reflectivity aloft",
      extract(refl2, mask2, GRID, PROFILE).max_dbz_aloft < 0)

print("\nstandoff is dynamic, not per-type")
from datetime import timedelta

from llcc.serialize import required_standoff_nmi
from llcc.world import CloudObject, Event, EventKind, WorldSnapshot

NOW = datetime(2026, 8, 26, 17, 0, 0)


def snap_with(obj, events=()):
    return WorldSnapshot(time=NOW, objects={obj.id: obj}, events=list(events),
                         profile=PROFILE)


def cu(top):
    return CloudObject(id="o", cloud_type=CloudType.CUMULUS, top_temp_c=top)


check("a warm cumulus has no standoff, only the through-cloud rule",
      required_standoff_nmi(cu(-4.0), snap_with(cu(-4.0)), NOW) == 0.0)
check("a cumulus colder than -10 C demands 5 nmi",
      required_standoff_nmi(cu(-12.0), snap_with(cu(-12.0)), NOW) == 5.0)
check("a cumulus colder than -20 C demands 10 nmi",
      required_standoff_nmi(cu(-25.0), snap_with(cu(-25.0)), NOW) == 10.0)
check("an unknown cloud top demands the widest standoff",
      required_standoff_nmi(cu(None), snap_with(cu(None)), NOW) == 10.0)

layer = CloudObject(id="o", cloud_type=CloudType.THICK_LAYER, top_temp_c=-30.0)
check("a thick cloud layer has no standoff at all",
      required_standoff_nmi(layer, snap_with(layer), NOW) == 0.0)

anvil = CloudObject(id="o", cloud_type=CloudType.DETACHED_ANVIL)
check("a quiet detached anvil demands 3 nmi",
      required_standoff_nmi(anvil, snap_with(anvil), NOW) == 3.0)
check("a recent discharge widens the anvil standoff to 10 nmi",
      required_standoff_nmi(anvil, snap_with(anvil, [
          Event(EventKind.LIGHTNING, NOW - timedelta(minutes=10), "o")]),
          NOW) == 10.0)

att = CloudObject(id="o", cloud_type=CloudType.ATTACHED_ANVIL,
                  parent_coldest_top_temp_c=-30.0)
check("an attached anvil demands 3 nmi when quiet",
      required_standoff_nmi(att, snap_with(att), NOW) == 3.0)
check("a discharge an hour ago widens it to 5 nmi",
      required_standoff_nmi(att, snap_with(att, [
          Event(EventKind.LIGHTNING, NOW - timedelta(minutes=60), "o")]),
          NOW) == 5.0)
warm_parent = CloudObject(id="o", cloud_type=CloudType.ATTACHED_ANVIL,
                          parent_coldest_top_temp_c=-4.0)
check("an anvil from a warm parent is out of scope for 4.1.4",
      required_standoff_nmi(warm_parent, snap_with(warm_parent), NOW) == 0.0)

deb = CloudObject(id="o", cloud_type=CloudType.DEBRIS)
check("debris inside its 3-hour period demands 3 nmi",
      required_standoff_nmi(deb, snap_with(deb, [
          Event(EventKind.DETACHMENT, NOW - timedelta(hours=1), "o")]),
          NOW) == 3.0)
check("debris past its 3-hour period demands none",
      required_standoff_nmi(deb, snap_with(deb, [
          Event(EventKind.DETACHMENT, NOW - timedelta(hours=4), "o")]),
          NOW) == 0.0)

print(f"\n{'all classifier tests passed' if not FAILURES else str(len(FAILURES)) + ' FAILURES: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
