"""World model consumed by the evaluator.

The evaluator is a pure function of (WorldSnapshot, t). All memory lives here:
the object store with its lineage edges, and the event log. Nothing in the
evaluator retains state between calls, which is what makes replay possible.

Geometry values on CloudObject are supplied by the geometry engine (corridor
rasterisation plus the two distance fields). In this harness they are read
straight from the scenario file, so that a failing test points at the
evaluator rather than at the distance transform.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

NM_TO_M = 1852.0


class CloudType:
    CUMULUS = "cumulus"
    ATTACHED_ANVIL = "attached_anvil"
    DETACHED_ANVIL = "detached_anvil"
    DEBRIS = "debris"
    THICK_LAYER = "thick_layer"
    SMOKE_CUMULUS = "smoke_cumulus"
    CIRRIFORM = "cirriform"
    UNCLASSIFIED = "unclassified"


class EventKind:
    LIGHTNING = "lightning"
    DETACHMENT = "detachment"
    PARENT_TOP_COLLAPSE = "parent_top_collapse"
    SMOKE_PLUME_DETACHMENT = "smoke_plume_detachment"


@dataclass(frozen=True)
class Event:
    kind: str
    time: datetime
    object_id: str
    source: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class ThermalProfile:
    """Temperature as a function of altitude, from HRRR/RAP or a sounding.

    T(z) is not monotonic in general. `isotherm_altitude` returns the LOWEST
    crossing, which maximises the volume counted as colder than the threshold
    and is therefore the conservative choice.
    """

    levels: list[tuple[float, float]]  # (altitude_m_msl, temp_c), ascending
    uncertainty_m: float = 300.0

    def temp_at(self, altitude_m: float) -> float | None:
        levels = self.levels
        if not levels or altitude_m < levels[0][0] or altitude_m > levels[-1][0]:
            return None
        for i in range(1, len(levels)):
            z0, t0 = levels[i - 1]
            z1, t1 = levels[i]
            if z0 <= altitude_m <= z1:
                if z1 == z0:
                    return t0
                f = (altitude_m - z0) / (z1 - z0)
                return t0 + f * (t1 - t0)
        return None

    def isotherm_altitude(self, temp_c: float) -> float | None:
        levels = self.levels
        for i in range(1, len(levels)):
            z0, t0 = levels[i - 1]
            z1, t1 = levels[i]
            lo, hi = min(t0, t1), max(t0, t1)
            if lo <= temp_c <= hi:
                if t1 == t0:
                    return z0
                f = (temp_c - t0) / (t1 - t0)
                return z0 + f * (z1 - z0)
        return None


@dataclass
class Series:
    """A sampled scalar with explicit coverage.

    `sustained` is the workhorse for every 'have been ... for at least
    15 minutes' clause. A data gap inside the window makes the answer
    unknown, not true: you cannot assert a predicate held across time you
    did not observe.
    """

    samples: list[tuple[datetime, float]]
    max_gap: timedelta = timedelta(minutes=2)

    def _window(self, now: datetime, window: timedelta) -> list[tuple[datetime, float]]:
        start = now - window
        times = [s[0] for s in self.samples]
        i = max(0, bisect_left(times, start) - 1)
        return [s for s in self.samples[i:] if s[0] <= now]

    def covers(self, now: datetime, window: timedelta) -> bool:
        pts = self._window(now, window)
        if len(pts) < 2:
            return False
        if pts[0][0] > now - window:
            return False
        if now - pts[-1][0] > self.max_gap:
            return False
        for a, b in zip(pts, pts[1:]):
            if b[0] - a[0] > self.max_gap:
                return False
        return True

    def sustained(self, predicate, now: datetime, window: timedelta) -> bool | None:
        if not self.covers(now, window):
            return None
        pts = [p for p in self._window(now, window) if p[0] >= now - window]
        if not pts:
            return None
        return all(predicate(v) for _, v in pts)

    def latest(self, now: datetime) -> float | None:
        pts = [s for s in self.samples if s[0] <= now]
        if not pts:
            return None
        if now - pts[-1][0] > self.max_gap:
            return None
        return pts[-1][1]


@dataclass
class CloudObject:
    """A tracked cloud object with lineage and geometry summary.

    `parent_ids` are the lineage edges from the tracker. Lightning and
    detachment events reach an object through its ancestry, which is what
    LLCCR 13/14/16/17 mean by 'the parent cloud or anvil cloud ... before
    detachment'.
    """

    id: str
    cloud_type: str = CloudType.UNCLASSIFIED
    parent_ids: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    unknown_provenance: bool = False
    observability: float = 1.0

    slant_min_nmi: float | None = None
    horiz_min_nmi: float | None = None
    intersects_corridor: bool | None = None

    top_temp_c: float | None = None
    coldest_top_temp_c: float | None = None
    parent_coldest_top_temp_c: float | None = None
    top_colder_than_0c_last_3h: bool | None = None
    top_colder_than_5c_last_3h: bool | None = None
    refl_above_10dbz_last_3h: bool | None = None
    will_grow_colder_than_0c: bool | None = None
    will_grow_colder_than_5c: bool | None = None

    portion_within_5nmi_all_below_0c: bool | None = None
    portion_within_10nmi_all_below_0c: bool | None = None
    entirely_colder_than_15c: bool | None = None
    contains_liquid_water: bool | None = None
    ever_connected_to_convection: bool | None = None

    mrr_max_within_1nmi_dbz: float | None = None
    mrr_max_in_corridor_dbz: float | None = None
    mrr_valid: bool | None = None
    layer_thickness_m: float | None = None
    layer_spans_0_to_minus20: bool | None = None
    refl_0dbz_within_5nmi: bool | None = None
    mrr_ge_7p5_within_2nmi_last_hour: bool | None = None

    producing_precip: bool | None = None
    moderate_precip_within_5nmi: bool | None = None
    bright_band_within_5nmi: bool | None = None
    associated_with_disturbed_weather: bool | None = None
    tops_colder_than_0c_in_system: bool | None = None

    corridor_penetrates_below_minus10c: bool | None = None
    velocity_at_penetration_ms: float | None = None

    attached_to_smoke_plume: bool | None = None
    is_thunderstorm: bool | None = None
    # Set by the override pass. The tracker's own label is preserved in
    # override_original_type so the display never presents a human judgment
    # as though the system produced it.
    overridden_by: str | None = None
    override_original_type: str | None = None
    # Display geometry, supplied by the geometry engine. Polar about the pad.
    bearing_deg: float | None = None
    range_nmi: float | None = None
    radius_nmi: float | None = None
    base_alt_km: float | None = None
    top_alt_km: float | None = None
    downrange_nmi: float | None = None
    producing_cloud_beyond_10nmi: bool | None = None

    series: dict[str, Series] = field(default_factory=dict)


@dataclass
class VehicleConfig:
    """Per-vehicle configuration. No defaults: an unconfigured vehicle must
    fail loudly as indeterminate rather than being silently exempted or
    silently held.

    `triboelectric_exemption` records which subsection of 4.1.10.2 is being
    invoked, and `triboelectric_basis` the document that substantiates it.
    LLCCR 28a is the surface-treatment route (resistivity < 1e9 ohms/square,
    bonding < 1e5 ohms); 28b is demonstration by test or analysis. Neither is
    a waiver under section 1.3 -- both are exemptions the standard grants,
    and both require evidence on file.
    """

    name: str = ""
    triboelectric_exemption: str | None = None   # "4.1.10.2a" | "4.1.10.2b" | "none"
    triboelectric_basis: str = ""


@dataclass
class WorldSnapshot:
    time: datetime
    objects: dict[str, CloudObject]
    events: list[Event]
    profile: ThermalProfile
    connections: list[tuple[str, str]] = field(default_factory=list)
    field_mills_available: bool = False
    vehicle: VehicleConfig | None = None
    feed_status: dict[str, datetime] = field(default_factory=dict)

    def ancestry(self, object_id: str, _seen: set[str] | None = None) -> set[str]:
        seen = _seen if _seen is not None else set()
        if object_id in seen:
            return seen
        seen.add(object_id)
        obj = self.objects.get(object_id)
        if obj:
            for pid in obj.parent_ids:
                self.ancestry(pid, seen)
        return seen

    def events_for(self, object_id: str, kind: str | None = None) -> list[Event]:
        line = self.ancestry(object_id)
        out = [e for e in self.events if e.object_id in line]
        if kind is not None:
            out = [e for e in out if e.kind == kind]
        return sorted(out, key=lambda e: e.time)

    def last_event_time(self, object_id: str, kind: str) -> datetime | None:
        evs = self.events_for(object_id, kind)
        return evs[-1].time if evs else None

    def connected_to(self, object_id: str) -> set[str]:
        out = set()
        for a, b in self.connections:
            if a == object_id:
                out.add(b)
            elif b == object_id:
                out.add(a)
        return out

    def connected_clusters(self) -> list[set[str]]:
        """Connected components of the physical-connection graph."""
        remaining = set(self.objects)
        clusters: list[set[str]] = []
        while remaining:
            seed = remaining.pop()
            cluster = {seed}
            frontier = [seed]
            while frontier:
                node = frontier.pop()
                for nb in self.connected_to(node):
                    if nb in remaining:
                        remaining.remove(nb)
                        cluster.add(nb)
                        frontier.append(nb)
            clusters.append(cluster)
        return clusters


def nmi(value: float) -> float:
    return value


def minutes(value: float) -> timedelta:
    return timedelta(minutes=value)


def hours(value: float) -> timedelta:
    return timedelta(hours=value)


def all_objects(snapshot: WorldSnapshot) -> Iterable[CloudObject]:
    return snapshot.objects.values()
