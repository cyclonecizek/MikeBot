"""Expression AST for encoding the LLCC.

Requirements are encoded as data, not as imperative code, so that the
encoding can be diffed line by line against the document text and so that
every evaluation produces an evidence trace for free.

Every leaf resolves to one of the geometry primitives, an object attribute,
a sustained-series check, or an event-log timer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .tvl import Tri, and_, compare, not_, or_, tri
from .world import CloudObject, EventKind, WorldSnapshot


@dataclass
class Context:
    snapshot: WorldSnapshot
    obj: CloudObject
    now: datetime


@dataclass
class Trace:
    label: str
    value: Tri
    detail: str = ""
    children: list["Trace"] = field(default_factory=list)

    def render(self, indent: int = 0) -> str:
        pad = "  " * indent
        line = f"{pad}[{self.value}] {self.label}"
        if self.detail:
            line += f"  ({self.detail})"
        out = [line]
        for child in self.children:
            out.append(child.render(indent + 1))
        return "\n".join(out)


class Expr:
    def eval(self, ctx: Context) -> Trace:  # pragma: no cover - interface
        raise NotImplementedError

    def release_times(self, ctx: Context) -> list[datetime]:
        return []


@dataclass
class Const(Expr):
    value: Tri
    label: str = "const"

    def eval(self, ctx: Context) -> Trace:
        return Trace(self.label, self.value)


@dataclass
class Unimplemented(Expr):
    label: str = "not implemented"

    def eval(self, ctx: Context) -> Trace:
        return Trace(self.label, Tri.UNKNOWN, "encoding absent")


@dataclass(init=False)
class And(Expr):
    terms: tuple[Expr, ...]

    def __init__(self, *terms: Expr):
        self.terms = terms

    def eval(self, ctx: Context) -> Trace:
        kids = [t.eval(ctx) for t in self.terms]
        return Trace("all of", and_(*[k.value for k in kids]), "", kids)

    def release_times(self, ctx: Context) -> list[datetime]:
        out: list[datetime] = []
        for t in self.terms:
            out.extend(t.release_times(ctx))
        return out


@dataclass(init=False)
class Or(Expr):
    terms: tuple[Expr, ...]

    def __init__(self, *terms: Expr):
        self.terms = terms

    def eval(self, ctx: Context) -> Trace:
        kids = [t.eval(ctx) for t in self.terms]
        return Trace("any of", or_(*[k.value for k in kids]), "", kids)

    def release_times(self, ctx: Context) -> list[datetime]:
        out: list[datetime] = []
        for t in self.terms:
            out.extend(t.release_times(ctx))
        return out


@dataclass
class Not(Expr):
    term: Expr

    def eval(self, ctx: Context) -> Trace:
        kid = self.term.eval(ctx)
        return Trace("not", not_(kid.value), "", [kid])

    def release_times(self, ctx: Context) -> list[datetime]:
        return self.term.release_times(ctx)


@dataclass
class Attr(Expr):
    name: str
    label: str = ""

    def eval(self, ctx: Context) -> Trace:
        value = getattr(ctx.obj, self.name, None)
        return Trace(self.label or self.name, tri(value), f"{self.name}={value}")


@dataclass
class Cmp(Expr):
    name: str
    op: str
    rhs: float
    label: str = ""

    def eval(self, ctx: Context) -> Trace:
        lhs = getattr(ctx.obj, self.name, None)
        value = compare(lhs, self.op, self.rhs)
        text = f"{self.name}={lhs} {self.op} {self.rhs}"
        return Trace(self.label or f"{self.name} {self.op} {self.rhs}", value, text)


@dataclass(init=False)
class IsType(Expr):
    types: tuple[str, ...]

    def __init__(self, *types: str):
        self.types = types

    def eval(self, ctx: Context) -> Trace:
        actual = ctx.obj.cloud_type
        label = "type in " + "/".join(self.types)
        if actual == "unclassified":
            return Trace(label, Tri.UNKNOWN, "object unclassified")
        return Trace(label, tri(actual in self.types), f"type={actual}")


@dataclass
class Sustained(Expr):
    """'... have been <predicate> for at least <window>'.

    Returns UNKNOWN when series coverage has a gap inside the window.
    """

    series: str
    op: str
    rhs: float
    window: timedelta
    label: str = ""
    absolute: bool = False

    def eval(self, ctx: Context) -> Trace:
        series = ctx.obj.series.get(self.series)
        label = self.label or f"{self.series} {self.op} {self.rhs} sustained"
        if series is None:
            return Trace(label, Tri.UNKNOWN, "series unavailable")

        def pred(v: float) -> bool:
            value = abs(v) if self.absolute else v
            return compare(value, self.op, self.rhs) is Tri.TRUE

        result = series.sustained(pred, ctx.now, self.window)
        detail = "coverage gap" if result is None else f"window={self.window}"
        return Trace(label, tri(result), detail)


@dataclass
class Elapsed(Expr):
    """TRUE once `duration` has passed since the last matching event.

    The release time is always derived from the event log, never stored, so
    that a flash inherited across a lineage split retroactively extends the
    clock on the child object.
    """

    kind: str
    duration: timedelta
    label: str = ""

    def _last(self, ctx: Context) -> datetime | None:
        return ctx.snapshot.last_event_time(ctx.obj.id, self.kind)

    def eval(self, ctx: Context) -> Trace:
        last = self._last(ctx)
        label = self.label or f"{self.duration} since {self.kind}"
        if last is None:
            return Trace(label, Tri.TRUE, "no such event on record")
        release = last + self.duration
        value = tri(ctx.now >= release)
        return Trace(label, value, f"last={last:%H:%M:%S}Z release={release:%H:%M:%S}Z")

    def release_times(self, ctx: Context) -> list[datetime]:
        last = self._last(ctx)
        if last is None:
            return []
        release = last + self.duration
        return [release] if ctx.now < release else []


@dataclass
class RecentLightning(Expr):
    """TRUE if no lightning within `max_nmi` of the flight path in `window`.

    Distances come from the geometry layer. GLM has no altitude, so per
    LLCCR 34b slant distance collapses to horizontal distance between
    vertical projections; the geometry engine writes that into the event.
    """

    max_nmi: float
    window: timedelta
    label: str = ""

    def _qualifying(self, ctx: Context) -> list[datetime]:
        out = []
        for ev in ctx.snapshot.events:
            if ev.kind != EventKind.LIGHTNING:
                continue
            dist = ev.detail.get("distance_nmi")
            if dist is None or dist <= self.max_nmi:
                out.append(ev.time)
        return sorted(out)

    def eval(self, ctx: Context) -> Trace:
        label = self.label or f"no lightning <= {self.max_nmi} nmi in {self.window}"
        times = self._qualifying(ctx)
        if not times:
            return Trace(label, Tri.TRUE, "no qualifying discharges")
        last = times[-1]
        release = last + self.window
        return Trace(
            label,
            tri(ctx.now >= release),
            f"last={last:%H:%M:%S}Z release={release:%H:%M:%S}Z",
        )

    def release_times(self, ctx: Context) -> list[datetime]:
        times = self._qualifying(ctx)
        if not times:
            return []
        release = times[-1] + self.window
        return [release] if ctx.now < release else []


def _debris_period_start(ctx: "Context") -> tuple[datetime | None, str]:
    """Section 4.1.6.1: the '3-hour period' starts at the LATEST of
    detachment from the parent, formation by collapse of the parent top to
    warmer than -10 C, or a discharge within or from the debris cloud.

    The first two are observations about this object and travel with its
    lineage. The third is scoped to the debris cloud itself (4.1.6.1c), so
    it uses direct attribution and does not inherit the parent's later
    flashes.
    """
    snap = ctx.snapshot
    oid = ctx.obj.id
    basis = {
        "detachment": snap.last_event_time(oid, EventKind.DETACHMENT),
        "parent top collapse": snap.last_direct_event_time(
            oid, EventKind.PARENT_TOP_COLLAPSE),
        "discharge in debris cloud": snap.last_direct_event_time(
            oid, EventKind.LIGHTNING),
    }
    present = {k: v for k, v in basis.items() if v is not None}
    if not present:
        return None, "no detachment, collapse or discharge on record"
    label, when = max(present.items(), key=lambda kv: kv[1])
    return when, f"latest basis: {label} at {when:%H:%M:%S}Z"


@dataclass
class DebrisPeriodDeterminable(Expr):
    """LLCCR 18 is a calculation requirement, not a standoff. It is met when
    at least one of the three basis observations exists; without any of them
    the period cannot be calculated at all."""

    label: str = "3-hour period can be calculated"

    def eval(self, ctx: Context) -> Trace:
        start, detail = _debris_period_start(ctx)
        return Trace(self.label, tri(start is not None), detail)


@dataclass
class DebrisPeriodElapsed(Expr):
    """TRUE once the 4.1.6.1 '3-hour period' has run out.

    UNKNOWN when the period start cannot be determined, so an
    uncalculable period blocks rather than reading as expired.
    """

    duration: timedelta = timedelta(hours=3)
    label: str = "3-hour period elapsed"

    def eval(self, ctx: Context) -> Trace:
        start, detail = _debris_period_start(ctx)
        if start is None:
            return Trace(self.label, Tri.UNKNOWN, detail)
        release = start + self.duration
        return Trace(self.label, tri(ctx.now >= release),
                     f"{detail}; release {release:%H:%M:%S}Z")

    def release_times(self, ctx: Context) -> list[datetime]:
        start, _ = _debris_period_start(ctx)
        if start is None:
            return []
        release = start + self.duration
        return [release] if ctx.now < release else []


@dataclass
class FieldMills(Expr):
    """Field mill availability gate.

    With mills out of scope this evaluates FALSE, which makes every
    mill-dependent exception unavailable and forces the conservative branch
    without any special-casing in the requirement encodings.
    """

    label: str = "working field mills available"

    def eval(self, ctx: Context) -> Trace:
        avail = ctx.snapshot.field_mills_available
        return Trace(self.label, tri(avail), "out of scope this iteration" if not avail else "")


@dataclass
class MrrValid(Expr):
    """LLCCR 31d gate: MRR evaluation points must be clear of 35 dBZ cores
    and of any lightning in the previous five minutes. An MRR-based
    exception is unavailable when its evaluation points are invalid."""

    label: str = "MRR evaluation points valid"

    def eval(self, ctx: Context) -> Trace:
        return Trace(self.label, tri(ctx.obj.mrr_valid), f"mrr_valid={ctx.obj.mrr_valid}")


@dataclass
class TriboExemption(Expr):
    """Section 4.1.10.2 exemption from the triboelectrification criterion.

    Unconfigured yields UNKNOWN so that a missing vehicle configuration
    surfaces as indeterminate instead of quietly exempting or quietly
    holding the vehicle.
    """

    label: str = "4.1.10.2 exemption on file"

    def eval(self, ctx: Context) -> Trace:
        vehicle = ctx.snapshot.vehicle
        if vehicle is None or vehicle.triboelectric_exemption is None:
            return Trace(self.label, Tri.UNKNOWN, "vehicle configuration not supplied")
        basis = vehicle.triboelectric_exemption
        if basis == "none":
            return Trace(self.label, Tri.FALSE, "no exemption claimed")
        if basis not in ("4.1.10.2a", "4.1.10.2b"):
            return Trace(self.label, Tri.UNKNOWN, f"unrecognised basis {basis!r}")
        detail = f"{basis}"
        if vehicle.triboelectric_basis:
            detail += f" per {vehicle.triboelectric_basis}"
        return Trace(self.label, Tri.TRUE, detail)
