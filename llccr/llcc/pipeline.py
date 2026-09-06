"""The assembler.

Closes the loop: a gridded reflectivity scan goes in, a WorldSnapshot comes
out, and the evaluator consumes it exactly as it consumes a scenario file.
That equivalence is the whole point of fixing the contract early -- live
data and archived replay are interchangeable inputs.

    refl, observable  -->  segment  -->  track  -->  geometry  -->  snapshot

Classification is a hook, not a component. The default labels every object
the most restrictive type it could be, which yields a working and absurdly
conservative system on day one. A real classifier then earns its keep
incrementally by *relaxing* that assumption for objects it can positively
identify, so the system degrades toward safe rather than toward unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from .geometry import (
    NM_TO_M, Corridor, Grid, all_colder_than, corridor_penetrates_colder_than,
    horiz_min, intersects_corridor, layer_thickness, mrr_field,
    mrr_max_in_corridor, mrr_max_within_horiz, mrr_validity, slant_min,
    voxels_within,
)
from .segment import PRECIP_DBZ, precipitation_flags, segment_field
from .track import Tracker
from .world import (
    CloudObject, CloudType, Event, EventKind, ThermalProfile, VehicleConfig,
    WorldSnapshot,
)

THREE_HOURS = timedelta(hours=3)


def worst_case_classifier(track, segment, grid, profile) -> str:
    """Day-one placeholder: assume the most restrictive type.

    Cumulus carries the widest standoffs of any type in section 4.1, so
    labelling everything cumulus is the conservative assumption. It is not a
    claim about what the cloud is; it is a claim about what we are willing
    to rule out, which is nothing.
    """
    return CloudType.CUMULUS


@dataclass
class PipelineConfig:
    grid: Grid
    corridor: Corridor
    profile: ThermalProfile
    vehicle: VehicleConfig | None = None
    field_mills_available: bool = False
    classifier: callable = worst_case_classifier
    pad: str = "LC-39A"
    azimuth_deg: float = 45.0


@dataclass
class Pipeline:
    config: PipelineConfig
    tracker: Tracker = field(default_factory=Tracker)
    events: list[Event] = field(default_factory=list)
    _flashed: set[str] = field(default_factory=set)

    def _temp_at(self, altitude_m: float):
        return self.config.profile.temp_at(altitude_m)

    def _isotherm(self, temp_c: float):
        return self.config.profile.isotherm_altitude(temp_c)

    def _attribute_lightning(self, strikes, when: datetime,
                             mapping: dict[int, str], segmentation) -> None:
        """Attach each flash to the nearest object and record its distance to
        the flight path.

        GLM has no altitude, so per LLCCR 34b the slant distance collapses to
        the horizontal distance between vertical projections -- which is what
        `d_horiz` already holds.
        """
        grid, corridor = self.config.grid, self.config.corridor
        for east, north in strikes:
            i, j, _ = grid.index_of(east, north, grid.z0)
            i = int(np.clip(i, 0, grid.nx - 1))
            j = int(np.clip(j, 0, grid.ny - 1))
            distance_nmi = float(corridor.d_horiz[i, j] / NM_TO_M)

            best, best_d = None, np.inf
            for lab, tid in mapping.items():
                ground = segmentation.segments[lab].mask.any(axis=2)
                idx = np.argwhere(ground)
                if not len(idx):
                    continue
                d = np.min(np.hypot((idx[:, 0] - i) * grid.dx,
                                    (idx[:, 1] - j) * grid.dy))
                if d < best_d:
                    best, best_d = tid, d

            if best is not None:
                self._flashed.add(best)
            self.events.append(Event(
                EventKind.LIGHTNING, when, best or "unattributed", "GLM",
                {"distance_nmi": distance_nmi,
                 "attribution_m": None if best is None else float(best_d)}))

    def ingest(self, refl: np.ndarray, when: datetime,
               observable: np.ndarray | None = None,
               strikes: list[tuple[float, float]] | None = None
               ) -> WorldSnapshot:
        cfg = self.config
        grid, corridor = cfg.grid, cfg.corridor

        segmentation = segment_field(refl, grid, observable)
        mapping, track_events = self.tracker.update(
            segmentation, refl, grid, when, self._temp_at, observable)

        # Tracker events become world events, which is how a split timestamp
        # reaches the debris and detached-anvil clocks.
        for ev in track_events:
            if ev.kind == "split":
                self.events.append(Event(EventKind.DETACHMENT, ev.time,
                                         ev.track_id, "tracker"))
            elif ev.kind == "collapse":
                self.events.append(Event(EventKind.PARENT_TOP_COLLAPSE, ev.time,
                                         ev.track_id, "tracker", ev.detail))

        self._attribute_lightning(strikes or [], when, mapping, segmentation)

        z0c = self._isotherm(0.0)
        zm10 = self._isotherm(-10.0)
        zm20 = self._isotherm(-20.0)
        zm15 = self._isotherm(-15.0)

        mrr = mrr_field(refl, grid, z0c if z0c is not None else 0.0)
        recent = [
            (e.detail.get("east", 0.0), e.detail.get("north", 0.0))
            for e in self.events
            if e.kind == EventKind.LIGHTNING and when - e.time <= timedelta(minutes=5)
            and "east" in e.detail
        ]
        valid = mrr_validity(refl, grid, z0c if z0c is not None else 0.0, recent)

        objects: dict[str, CloudObject] = {}
        cloud_all = np.zeros(grid.shape, dtype=bool)

        for lab, tid in mapping.items():
            seg = segmentation.segments[lab]
            track = self.tracker.tracks[tid]
            cloud_all |= seg.mask

            top_m = seg.top_altitude(grid)
            parents = [self.tracker.tracks[p] for p in track.parent_ids
                       if p in self.tracker.tracks]
            parent_cold = [p.coldest_top_c for p in parents
                           if p.coldest_top_c is not None]
            precip, moderate = precipitation_flags(refl, seg.mask)
            near5 = voxels_within(corridor, seg.mask, 5.0)

            mrr1, ok1 = mrr_max_within_horiz(corridor, mrr, valid, 1.0)
            mrrc, okc = mrr_max_in_corridor(corridor, mrr, valid)

            centroid_i, centroid_j, _ = seg.centroid
            east = (centroid_i - grid.nx / 2 + 0.5) * grid.dx
            north = (centroid_j - grid.ny / 2 + 0.5) * grid.dy
            az = np.radians(cfg.azimuth_deg)

            objects[tid] = CloudObject(
                id=tid,
                cloud_type=cfg.classifier(track, seg, grid, cfg.profile),
                parent_ids=list(track.parent_ids),
                first_seen=track.first_seen,
                unknown_provenance=track.unknown_provenance,
                observability=track.observability,

                slant_min_nmi=slant_min(corridor, seg.mask),
                horiz_min_nmi=horiz_min(corridor, seg.mask),
                intersects_corridor=intersects_corridor(corridor, seg.mask),

                top_temp_c=self._temp_at(top_m),
                coldest_top_temp_c=track.coldest_top_c,
                parent_coldest_top_temp_c=min(parent_cold) if parent_cold else None,
                top_colder_than_0c_last_3h=track.colder_than_within(
                    0.0, THREE_HOURS, when),
                top_colder_than_5c_last_3h=track.colder_than_within(
                    -5.0, THREE_HOURS, when),
                refl_above_10dbz_last_3h=track.refl_above_within(
                    10.0, THREE_HOURS, when),

                portion_within_5nmi_all_below_0c=all_colder_than(grid, near5, z0c),
                portion_within_10nmi_all_below_0c=all_colder_than(
                    grid, voxels_within(corridor, seg.mask, 10.0), z0c),
                entirely_colder_than_15c=all_colder_than(grid, seg.mask, zm15),

                mrr_max_within_1nmi_dbz=mrr1,
                mrr_max_in_corridor_dbz=mrrc,
                mrr_valid=bool(ok1 and okc),
                layer_thickness_m=layer_thickness(grid, seg.mask),
                layer_spans_0_to_minus20=bool(
                    z0c is not None and zm20 is not None
                    and seg.mask[:, :, grid.level_at(z0c):grid.level_at(zm20)].any()),
                refl_0dbz_within_5nmi=bool(near5.any()),

                producing_precip=precip,
                moderate_precip_within_5nmi=bool(
                    near5.any() and np.nanmax(refl[near5]) >= PRECIP_DBZ),
                is_thunderstorm=tid in self._flashed,
                parent_had_part_colder_than_minus20=bool(
                    any(c <= -20.0 for c in parent_cold)) if parent_cold else None,
                formed_by_thunderstorm=any(
                    p.id in self._flashed for p in parents) or None,

                bearing_deg=float((np.degrees(np.arctan2(east, north))) % 360.0),
                range_nmi=float(np.hypot(east, north) / NM_TO_M),
                radius_nmi=float(np.sqrt(seg.mask.any(axis=2).sum()
                                         * grid.dx * grid.dy / np.pi) / NM_TO_M),
                base_alt_km=seg.base_altitude(grid) / 1000.0,
                top_alt_km=top_m / 1000.0,
                downrange_nmi=float((east * np.sin(az) + north * np.cos(az))
                                    / NM_TO_M),
            )

        penetrates, speed = corridor_penetrates_colder_than(
            grid, corridor, cloud_all, zm10)
        objects["DOMAIN"] = CloudObject(
            id="DOMAIN",
            corridor_penetrates_below_minus10c=penetrates,
            velocity_at_penetration_ms=speed,
            producing_cloud_beyond_10nmi=False,
        )

        connections = self.tracker.connections(segmentation, mapping)

        return WorldSnapshot(
            time=when,
            objects=objects,
            events=list(self.events),
            profile=cfg.profile,
            connections=connections,
            field_mills_available=cfg.field_mills_available,
            vehicle=cfg.vehicle,
        )
