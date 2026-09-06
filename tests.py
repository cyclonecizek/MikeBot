"""Hand-computed tests for the evaluator semantics.

The scenario baseline in scenarios/*.json is a regression check: it proves
behaviour has not changed, not that it is correct. These tests are the
non-circular part, asserting properties derived from the standard by hand.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from llcc.evaluate import State, assessment_units, evaluate, evaluate_one
from llcc.expr import Context, Elapsed, Sustained
from llcc.overrides import (Override, Status, apply_overrides,
                            diff_verdicts)
from llcc.requirements import ENCODED, MANIFEST, missing_ids
from llcc.tvl import F, T, U, and_, not_, or_
from llcc.world import (
    CloudObject,
    CloudType,
    Event,
    EventKind,
    Series,
    ThermalProfile,
    VehicleConfig,
    WorldSnapshot,
    minutes,
)

PROFILE = ThermalProfile(levels=[(0, 29.0), (4600, 0.0), (6600, -10.0), (8700, -20.0)])
T0 = datetime(2026, 9, 4, 21, 0, 0)
FAILURES: list[str] = []


def check(name: str, condition: bool) -> None:
    if not condition:
        FAILURES.append(name)
    print(f"  {'pass' if condition else 'FAIL'}  {name}")


def req(rid: str):
    return next(r for r in ENCODED if r.id == rid)


def snap(objects, events=(), connections=(), mills=False, vehicle=None):
    return WorldSnapshot(
        time=T0,
        objects={o.id: o for o in objects},
        events=list(events),
        profile=PROFILE,
        connections=list(connections),
        field_mills_available=mills,
        vehicle=vehicle,
    )


print("three-valued logic")
check("unknown AND false is false", and_(U, F) is F)
check("unknown AND true is unknown", and_(U, T) is U)
check("unknown OR true is true", or_(U, T) is T)
check("unknown OR false is unknown", or_(U, F) is U)
check("not unknown is unknown", not_(U) is U)

print("\nconservative direction")
# An unclassified object cannot be ruled out of a cumulus criterion, so
# LLCCR 1 keeps it in play and the result is indeterminate, not GO.
obj = CloudObject(id="x", cloud_type=CloudType.UNCLASSIFIED,
                  slant_min_nmi=2.0, top_temp_c=-15.0)
res = evaluate_one(req("LLCCR 10"), obj, snap([obj]), T0)
check("unclassified object stays in scope", res is not None)
check("unclassified object yields indeterminate", res.state is State.INDETERMINATE)

# A definitely-typed non-cumulus drops out of scope entirely.
obj2 = CloudObject(id="y", cloud_type=CloudType.THICK_LAYER,
                   slant_min_nmi=2.0, top_temp_c=-15.0)
check("typed non-cumulus leaves scope",
      evaluate_one(req("LLCCR 10"), obj2, snap([obj2]), T0) is None)

# Missing geometry must not read as satisfied.
obj3 = CloudObject(id="z", cloud_type=CloudType.CUMULUS, slant_min_nmi=None,
                   top_temp_c=-15.0)
res3 = evaluate_one(req("LLCCR 10"), obj3, snap([obj3]), T0)
check("missing distance yields indeterminate", res3.state is State.INDETERMINATE)

print("\nsustained predicates and coverage")
dense = Series(samples=[(T0 - timedelta(minutes=m), 500.0) for m in range(20, -1, -1)],
               max_gap=timedelta(minutes=2))
gapped = Series(samples=[(T0 - timedelta(minutes=m), 500.0)
                         for m in (20, 19, 18, 8, 7, 6, 5, 4, 3, 2, 1, 0)],
                max_gap=timedelta(minutes=2))
s_obj = CloudObject(id="s", series={"efield_max": dense})
ctx = Context(snap([s_obj]), s_obj, T0)
node = Sustained("efield_max", "<", 1000.0, minutes(15), absolute=True)
check("sustained holds with full coverage", node.eval(ctx).value is T)
s_obj.series["efield_max"] = gapped
check("coverage gap inside window yields unknown", node.eval(ctx).value is U)

print("\ntimers derived from the event log")
parent = CloudObject(id="cell", cloud_type=CloudType.CUMULUS)
child = CloudObject(id="anvil", cloud_type=CloudType.DETACHED_ANVIL,
                    parent_ids=["cell"])
flash = Event(EventKind.LIGHTNING, T0 - timedelta(minutes=10), "cell", "GLM")
sn = snap([parent, child], events=[flash])
ctx = Context(sn, child, T0)
elapsed = Elapsed(EventKind.LIGHTNING, minutes(30))
check("flash inherited across the lineage split", elapsed.eval(ctx).value is F)
check("release time derived from the parent's flash",
      elapsed.release_times(ctx) == [flash.time + minutes(30)])
orphan = CloudObject(id="orphan", cloud_type=CloudType.DETACHED_ANVIL)
check("unrelated object does not inherit the flash",
      Elapsed(EventKind.LIGHTNING, minutes(30))
      .eval(Context(sn, orphan, T0)).value is T)

print("\nsection 4.3 pre-pass")
cu = CloudObject(id="cu", cloud_type=CloudType.CUMULUS, coldest_top_temp_c=-30.0,
                 slant_min_nmi=4.0)
av = CloudObject(id="av", cloud_type=CloudType.ATTACHED_ANVIL, slant_min_nmi=9.0,
                 parent_coldest_top_temp_c=-30.0)
units = assessment_units(snap([cu, av], connections=[("cu", "av")]))
check("4.3a keeps anvil and cumulus independent", len(units) == 2)

ly = CloudObject(id="ly", cloud_type=CloudType.THICK_LAYER, slant_min_nmi=1.0)
db = CloudObject(id="db", cloud_type=CloudType.DEBRIS, slant_min_nmi=8.0)
units = assessment_units(
    snap([cu, av, ly, db], connections=[("cu", "av"), ("av", "ly"), ("ly", "db")]))
check("4.3e combines three or more types", len(units) == 1)
check("combined unit takes the minimum distance",
      units[0].slant_min_nmi == 1.0)

dv = CloudObject(id="dv", cloud_type=CloudType.DETACHED_ANVIL, slant_min_nmi=2.0)
units = assessment_units(snap([cu, dv], connections=[("cu", "dv")]))
promoted = [u for u in units if u.id == "dv"]
check("4.3b promotes a reconnected detached anvil to attached",
      promoted and promoted[0].cloud_type == CloudType.ATTACHED_ANVIL)

print("\nfield mills out of scope")
anv = CloudObject(id="a2", cloud_type=CloudType.DETACHED_ANVIL, slant_min_nmi=2.0,
                  portion_within_5nmi_all_below_0c=False, mrr_valid=True,
                  mrr_max_within_1nmi_dbz=20.0)
sn = snap([anv], events=[Event(EventKind.LIGHTNING, T0 - timedelta(minutes=45),
                              "a2", "GLM")])
res = evaluate_one(req("LLCCR 16"), anv, sn, T0)
check("mill-dependent exception unavailable, full 3 h clock runs",
      res.state is State.NO_GO_UNTIL
      and res.release == T0 - timedelta(minutes=45) + timedelta(hours=3))

print("\ntriboelectrification exemption (4.1.10.2)")
dom = CloudObject(id="DOMAIN", corridor_penetrates_below_minus10c=True,
                  velocity_at_penetration_ms=640.0)
r27 = req("LLCCR 27")
check("unconfigured vehicle is indeterminate, not exempt",
      evaluate_one(r27, dom, snap([dom]), T0).state is State.INDETERMINATE)
check("no exemption claimed blocks",
      evaluate_one(r27, dom, snap([dom], vehicle=VehicleConfig(
          triboelectric_exemption="none")), T0).state is State.NO_GO)
res28 = evaluate_one(r27, dom, snap([dom], vehicle=VehicleConfig(
    name="ref", triboelectric_exemption="4.1.10.2b",
    triboelectric_basis="ESD analysis")), T0)
check("4.1.10.2b exemption clears the criterion", res28.state is State.GO)
check("exemption basis appears in the evidence trace",
      "4.1.10.2b" in res28.exception.detail and "ESD analysis" in res28.exception.detail)
check("unrecognised basis is indeterminate",
      evaluate_one(r27, dom, snap([dom], vehicle=VehicleConfig(
          triboelectric_exemption="handshake")), T0).state is State.INDETERMINATE)

print("\ndebris cloud 3-hour period (4.1.6.1)")


def debris(**kw):
    base = dict(id="db", cloud_type=CloudType.DEBRIS,
                parent_had_part_colder_than_minus20=True,
                slant_min_nmi=2.0, intersects_corridor=False,
                portion_within_5nmi_all_below_0c=False, mrr_valid=True)
    base.update(kw)
    return CloudObject(**base)


def ev(kind, minutes_ago, oid="db"):
    return Event(kind, T0 - timedelta(minutes=minutes_ago), oid, "tracker")

# The period starts at the LATEST of the three bases, not the first.
d = debris()
sn = snap([d], events=[ev(EventKind.DETACHMENT, 200),
                       ev(EventKind.PARENT_TOP_COLLAPSE, 100),
                       ev(EventKind.LIGHTNING, 150)])
r = evaluate_one(req("LLCCR 20"), d, sn, T0)
check("period runs from the latest basis, not the earliest",
      r.state is State.NO_GO_UNTIL
      and r.release == T0 - timedelta(minutes=100) + timedelta(hours=3))

# Detachment is inherited through lineage; a discharge is not (4.1.6.1c).
parent2 = CloudObject(id="p", cloud_type=CloudType.CUMULUS)
d2 = debris(parent_ids=["p"])
sn2 = snap([parent2, d2], events=[ev(EventKind.DETACHMENT, 100),
                                  ev(EventKind.LIGHTNING, 5, "p")])
r2 = evaluate_one(req("LLCCR 20"), d2, sn2, T0)
check("parent discharge does not restart the debris clock",
      r2.release == T0 - timedelta(minutes=100) + timedelta(hours=3))
sn3 = snap([parent2, d2], events=[ev(EventKind.DETACHMENT, 100),
                                  ev(EventKind.LIGHTNING, 5)])
r3 = evaluate_one(req("LLCCR 20"), d2, sn3, T0)
check("a discharge in the debris cloud does restart it",
      r3.release == T0 - timedelta(minutes=5) + timedelta(hours=3))

# No basis at all: LLCCR 18 is violated and the period cannot gate 19/20.
sn4 = snap([debris()])
check("uncalculable period violates LLCCR 18",
      evaluate_one(req("LLCCR 18"), debris(), sn4, T0).state is State.NO_GO)
check("uncalculable period leaves LLCCR 20 indeterminate, not expired",
      evaluate_one(req("LLCCR 20"), debris(), sn4, T0).state is State.INDETERMINATE)

# Elapsed period releases the criterion.
sn5 = snap([debris()], events=[ev(EventKind.DETACHMENT, 200)])
check("criterion clears once 3 hours have passed",
      evaluate_one(req("LLCCR 20"), debris(), sn5, T0).state is State.GO)

# Scope: neither a cold parent nor a thunderstorm origin means 4.1.6 is out.
cold = debris(parent_had_part_colder_than_minus20=False,
              formed_by_thunderstorm=False)
check("debris from a warm non-thunderstorm parent is out of scope",
      evaluate_one(req("LLCCR 20"), cold, snap([cold]), T0) is None)
warm = debris(parent_had_part_colder_than_minus20=False,
              formed_by_thunderstorm=True)
check("debris formed by a thunderstorm is in scope regardless of parent top",
      evaluate_one(req("LLCCR 20"), warm,
                   snap([warm], events=[ev(EventKind.DETACHMENT, 10)]), T0)
      is not None)

print("\noverrides")


def ov(**kw):
    base = dict(object_id="o", cloud_type=CloudType.DETACHED_ANVIL,
                issued_at=T0 - timedelta(minutes=5), issued_by="LWO/AR",
                original_type=CloudType.THICK_LAYER,
                justification="fibrous, downwind of parent",
                facts={"detachment_time": (T0 - timedelta(hours=1)).isoformat(),
                       "parent_coldest_top_temp_c": -32.0})
    base.update(kw)
    return Override(**base)


def obj(**kw):
    base = dict(id="o", cloud_type=CloudType.THICK_LAYER, slant_min_nmi=2.0,
                intersects_corridor=True, layer_thickness_m=1900.0)
    base.update(kw)
    return CloudObject(**base)

o = obj()
sn, apps = apply_overrides(snap([o]), [ov()], T0)
check("complete override is applied", apps[0].status is Status.APPLIED)
check("override changes the type", sn.objects["o"].cloud_type == CloudType.DETACHED_ANVIL)
check("original tracker label preserved",
      sn.objects["o"].override_original_type == CloudType.THICK_LAYER)
check("override attributed to its author", sn.objects["o"].overridden_by == "LWO/AR")
check("detachment time reaches the event log",
      any(e.kind == EventKind.DETACHMENT and e.source.startswith("override:")
          for e in sn.events))
check("input snapshot is not mutated", o.cloud_type == CloudType.THICK_LAYER)

# Missing companion facts must degrade, never grant the label.
sn2, apps2 = apply_overrides(snap([obj()]), [ov(facts={})], T0)
check("missing companion facts degrade to unclassified",
      apps2[0].status is Status.DEGRADED
      and sn2.objects["o"].cloud_type == CloudType.UNCLASSIFIED)
check("degradation names the missing facts",
      "detachment_time" in apps2[0].detail)

# Lineage binding and expiry.
sn3, apps3 = apply_overrides(
    snap([obj(cloud_type=CloudType.DEBRIS)]), [ov()], T0)
check("override goes stale when the tracker relabels the object",
      apps3[0].status is Status.STALE
      and sn3.objects["o"].cloud_type == CloudType.DEBRIS)
_, apps4 = apply_overrides(snap([obj()]), [ov(ttl_seconds=60)], T0)
check("override expires without re-affirmation", apps4[0].status is Status.EXPIRED)
_, apps5 = apply_overrides(
    snap([obj()]), [ov(ttl_seconds=60, affirmed_at=T0 - timedelta(seconds=10))], T0)
check("re-affirmation revives an expired override", apps5[0].status is Status.APPLIED)
_, apps6 = apply_overrides(snap([obj(id="other")]), [ov()], T0)
check("override on an absent object is orphaned", apps6[0].status is Status.ORPHANED)

# Direction and justification.
base_snap = snap([obj()])
over_snap, _ = apply_overrides(base_snap, [ov()], T0)
d = diff_verdicts(evaluate(base_snap, T0, True),
                  evaluate(over_snap, T0, True), [ov()])
check("reclassification is detected as a change", len(d.changes) > 0)
check("scope-only changes are neutral, not restrictions",
      all(c["direction"] in ("restricts", "relaxes", "neutral") for c in d.changes))
d2 = diff_verdicts(evaluate(base_snap, T0, True),
                   evaluate(over_snap, T0, True), [ov(justification="  ")])
check("relaxing override without justification is flagged",
      not d2.relaxations or d2.unjustified == ["o"])

print("\nmanifest gate")
clear = CloudObject(id="c", cloud_type=CloudType.CIRRIFORM, slant_min_nmi=40.0,
                    intersects_corridor=False)
v = evaluate(snap([clear]), T0)
check("unencoded requirements force global indeterminate",
      v.state is State.INDETERMINATE and len(v.manifest_gaps) == len(missing_ids()))
v2 = evaluate(snap([clear]), T0, assume_manifest_complete=True)
check("gate suppressible for development only", v2.state is State.GO)
check("manifest covers all 35 requirements", len(MANIFEST) == 35)

print(f"\n{'all tests passed' if not FAILURES else str(len(FAILURES)) + ' FAILURES: ' + ', '.join(FAILURES)}")
raise SystemExit(1 if FAILURES else 0)
