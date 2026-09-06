"""Object tracking and lineage.

The evaluator's clocks are all derived from events, and the events come from
here. A split timestamp *is* the detachment time that starts the debris
clock in 4.1.6.1a and gates 4.1.5.1a's three-hour test, so association is
not a cosmetic concern -- getting it wrong moves a three-hour hold.

TITAN-style: advect each track forward by storm motion, match on volume
overlap, and treat merges and splits as explicit typed events rather than as
tracking failures.

Two asymmetries are deliberate, and both are safety properties rather than
tuning choices:

  Connection is declared on first evidence; disconnection only after it has
  held for several scans. Declaring a detachment late pushes the clock start
  later, which lengthens the hold. Declaring a reconnection late would let a
  still-attached anvil be assessed as detached, which is the opposite.

  A track that appears at the domain edge, or after a data gap, carries
  unknown provenance. It cannot satisfy any "for the previous 3 hours" test
  and must be evaluated on the conservative branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from skimage.registration import phase_cross_correlation

DISCONNECT_PERSISTENCE = 2      # scans an edge must be absent before it drops
MIN_OVERLAP = 0.15              # fraction of the smaller volume
COLLAPSE_TEMP_C = -10.0         # section 4.1.6.1b


@dataclass
class Track:
    """Persistent identity across scans, with the history the clocks need."""

    id: str
    first_seen: datetime
    last_seen: datetime
    parent_ids: list[str] = field(default_factory=list)
    unknown_provenance: bool = False

    mask: np.ndarray | None = None
    top_level: int = 0
    max_refl: float = -np.inf

    coldest_top_c: float | None = None
    top_history: list[tuple[datetime, float]] = field(default_factory=list)
    max_refl_history: list[tuple[datetime, float]] = field(default_factory=list)
    detached_at: datetime | None = None
    observability: float = 1.0

    def record(self, when: datetime, top_c: float | None, max_refl: float) -> None:
        self.last_seen = when
        self.max_refl = max_refl
        self.max_refl_history.append((when, max_refl))
        if top_c is not None:
            self.top_history.append((when, top_c))
            if self.coldest_top_c is None or top_c < self.coldest_top_c:
                self.coldest_top_c = top_c

    def colder_than_within(self, threshold_c: float, window: timedelta,
                           now: datetime) -> bool | None:
        """'For the previous 3 hours, the cloud top has not been colder than X'.

        Returns None when history does not span the window, so an object
        that has not been watched long enough cannot satisfy the test.
        """
        if not self.top_history:
            return None
        if self.top_history[0][0] > now - window:
            return None
        return any(t <= threshold_c for when, t in self.top_history
                   if when >= now - window)

    def refl_above_within(self, threshold_dbz: float, window: timedelta,
                          now: datetime) -> bool | None:
        if not self.max_refl_history:
            return None
        if self.max_refl_history[0][0] > now - window:
            return None
        return any(v > threshold_dbz for when, v in self.max_refl_history
                   if when >= now - window)


@dataclass
class TrackEvent:
    kind: str          # continuation | birth | split | merge | death
    time: datetime
    track_id: str
    parents: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)


def estimate_motion(prev_refl: np.ndarray, refl: np.ndarray) -> tuple[float, float]:
    """Storm motion in grid cells per scan, from the composite fields."""
    a = np.nan_to_num(np.nanmax(prev_refl, axis=2), nan=-30.0, neginf=-30.0)
    b = np.nan_to_num(np.nanmax(refl, axis=2), nan=-30.0, neginf=-30.0)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0, 0.0
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=4)
    return float(shift[0]), float(shift[1])


def _advect(mask: np.ndarray, di: int, dj: int) -> np.ndarray:
    out = np.zeros_like(mask)
    nx, ny = mask.shape[0], mask.shape[1]
    si0, si1 = max(0, di), min(nx, nx + di)
    di0, di1 = max(0, -di), min(nx, nx - di)
    sj0, sj1 = max(0, dj), min(ny, ny + dj)
    dj0, dj1 = max(0, -dj), min(ny, ny - dj)
    if si1 > si0 and sj1 > sj0:
        out[si0:si1, sj0:sj1] = mask[di0:di1, dj0:dj1]
    return out


class Tracker:
    """Frame-to-frame association with an explicit lineage graph."""

    def __init__(self, max_gap: timedelta = timedelta(minutes=12)):
        self.tracks: dict[str, Track] = {}
        self.max_gap = max_gap
        self._next = 1
        self._prev_refl: np.ndarray | None = None
        self._prev_time: datetime | None = None
        self._edge_absent: dict[tuple[str, str], int] = {}
        self._edges: set[tuple[str, str]] = set()

    def _new_id(self) -> str:
        tid = f"t{self._next:04d}"
        self._next += 1
        return tid

    def update(self, segmentation, refl: np.ndarray, grid, when: datetime,
               temp_at, observable: np.ndarray | None = None
               ) -> tuple[dict[int, str], list[TrackEvent]]:
        """Associate this frame's segments with existing tracks.

        `temp_at` maps an altitude in metres to a temperature in Celsius.
        Returns the label-to-track mapping and the events this frame produced.
        """
        from .segment import observability

        events: list[TrackEvent] = []
        gap = (self._prev_time is not None
               and when - self._prev_time > self.max_gap)

        di = dj = 0
        if self._prev_refl is not None and not gap:
            si, sj = estimate_motion(self._prev_refl, refl)
            di, dj = int(round(si)), int(round(sj))

        live = {tid: t for tid, t in self.tracks.items()
                if t.mask is not None and when - t.last_seen <= self.max_gap}
        advected = {tid: _advect(t.mask, di, dj) for tid, t in live.items()}

        # Overlap matrix between advected tracks and this frame's segments.
        overlap: dict[int, dict[str, float]] = {}
        for lab, seg in segmentation.segments.items():
            scores = {}
            for tid, prev in advected.items():
                inter = int((prev & seg.mask).sum())
                if not inter:
                    continue
                denom = max(1, min(int(prev.sum()), seg.voxels))
                if inter / denom >= MIN_OVERLAP:
                    scores[tid] = inter / denom
            overlap[lab] = scores

        claims: dict[str, list[int]] = {}
        for lab, scores in overlap.items():
            for tid in scores:
                claims.setdefault(tid, []).append(lab)

        mapping: dict[int, str] = {}
        for lab, seg in segmentation.segments.items():
            scores = overlap[lab]
            if not scores:
                tid = self._new_id()
                self.tracks[tid] = Track(
                    id=tid, first_seen=when, last_seen=when,
                    unknown_provenance=bool(seg.touches_edge or gap))
                mapping[lab] = tid
                events.append(TrackEvent("birth", when, tid, detail={
                    "reason": "domain edge" if seg.touches_edge
                              else ("after data gap" if gap else "new echo")}))
                continue

            best = max(scores, key=scores.get)
            siblings = claims.get(best, [])

            if len(siblings) > 1:
                # One track now covers several segments: a split. The largest
                # child keeps the identity; the others are new tracks whose
                # lineage points back at the parent, so lightning and history
                # propagate across the edge.
                largest = max(siblings, key=lambda x: segmentation.segments[x].voxels)
                if lab == largest:
                    mapping[lab] = best
                else:
                    tid = self._new_id()
                    parent = self.tracks[best]
                    self.tracks[tid] = Track(
                        id=tid, first_seen=when, last_seen=when,
                        parent_ids=[best],
                        coldest_top_c=parent.coldest_top_c,
                        top_history=list(parent.top_history),
                        max_refl_history=list(parent.max_refl_history),
                        detached_at=when)
                    mapping[lab] = tid
                    events.append(TrackEvent("split", when, tid, parents=[best]))
            elif len(scores) > 1:
                # Several tracks now cover one segment: a merge. Provenance
                # becomes the union, so no history is lost.
                keep = max(scores, key=scores.get)
                mapping[lab] = keep
                others = [t for t in scores if t != keep]
                self.tracks[keep].parent_ids = sorted(
                    set(self.tracks[keep].parent_ids) | set(others))
                events.append(TrackEvent("merge", when, keep, parents=others))
            else:
                mapping[lab] = best
                events.append(TrackEvent("continuation", when, best))

        # Record features, and watch for a parent top collapsing warmer than
        # -10 C, which is one of the three bases in section 4.1.6.1.
        for lab, tid in mapping.items():
            seg = segmentation.segments[lab]
            track = self.tracks[tid]
            top_c = temp_at(seg.top_altitude(grid))
            previous = track.top_history[-1][1] if track.top_history else None
            track.mask = seg.mask
            track.top_level = seg.top_level
            track.observability = observability(seg.mask, observable)
            track.record(when, top_c, seg.max_refl)
            if (previous is not None and top_c is not None
                    and previous <= COLLAPSE_TEMP_C < top_c):
                events.append(TrackEvent("collapse", when, tid, detail={
                    "from_c": previous, "to_c": top_c}))

        for tid, track in list(self.tracks.items()):
            if tid not in mapping.values() and track.mask is not None \
                    and when - track.last_seen > self.max_gap:
                events.append(TrackEvent("death", when, tid))
                track.mask = None

        self._prev_refl = refl
        self._prev_time = when
        return mapping, events

    def connections(self, segmentation, mapping: dict[int, str]) -> list[tuple[str, str]]:
        """Physical connection edges, with asymmetric persistence.

        Present edges are reported immediately. Absent edges survive
        DISCONNECT_PERSISTENCE scans before being dropped, so a flickering
        0 dBZ boundary cannot manufacture a detachment.
        """
        seen = set()
        for a, b in segmentation.connections:
            ta, tb = mapping.get(a), mapping.get(b)
            if ta and tb and ta != tb:
                seen.add(tuple(sorted((ta, tb))))

        for edge in seen:
            self._edges.add(edge)
            self._edge_absent.pop(edge, None)

        for edge in list(self._edges):
            if edge in seen:
                continue
            n = self._edge_absent.get(edge, 0) + 1
            self._edge_absent[edge] = n
            if n >= DISCONNECT_PERSISTENCE:
                self._edges.discard(edge)
                self._edge_absent.pop(edge, None)

        return sorted(self._edges)
