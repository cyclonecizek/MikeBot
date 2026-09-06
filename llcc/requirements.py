"""LLCC requirements encoded as data, mirroring NASA-STD-4010B section 4.

Each requirement is `shall not launch if TRIGGER, unless EXCEPTION`, so the
evaluator computes `permitted = NOT trigger OR exception` uniformly. The
capitalised AND/OR structure in the document maps directly onto the And/Or
nodes below, which is the point: the encoding is reviewable against the text.

MANIFEST lists all 35 LLCCRs from Appendix A. Anything not in ENCODED forces
a global indeterminate result rather than being silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .expr import (
    And,
    Attr,
    Cmp,
    Const,
    DebrisPeriodDeterminable,
    DebrisPeriodElapsed,
    Elapsed,
    Expr,
    FieldMills,
    IsType,
    MrrValid,
    Not,
    Or,
    RecentLightning,
    Sustained,
    TriboExemption,
)
from .tvl import Tri
from .world import CloudType, EventKind, hours, minutes

NEVER = Const(Tri.FALSE, "no exception in this requirement")

# The standard's own section headings. Shown in place of the LLCCR number,
# which is right for auditing and wrong for reading at a console.
RULE_NAMES: dict[str, str] = {
    "4.1.1": "Lightning",
    "4.1.2": "Surface Electric Fields",
    "4.1.3": "Cumulus Clouds",
    "4.1.4": "Attached Anvil Clouds",
    "4.1.5": "Detached Anvil Clouds",
    "4.1.6": "Debris Clouds",
    "4.1.7": "Disturbed Weather",
    "4.1.8": "Thick Cloud Layers",
    "4.1.9": "Smoke Plumes",
    "4.1.10": "Triboelectrification",
    "4.2.1": "Radar Reflectivity Measurement",
    "4.2.2": "Quantification of Precipitation",
    "4.2.3": "Computation of MRR",
    "4.2.4": "Surface Electric Field Measurement",
    "4.2.5": "Non-Transparent Boundaries",
    "4.2.6": "Slant Distance from Lightning",
    "4.3": "Physically Connected Clouds",
}


def rule_name(section: str) -> str:
    """Longest matching section prefix wins, so 4.1.10.1 resolves to
    Triboelectrification rather than to Lightning."""
    best = ""
    for prefix in RULE_NAMES:
        if section.startswith(prefix) and len(prefix) > len(best):
            best = prefix
    return RULE_NAMES.get(best, "Launch Commit Criteria")


@dataclass
class Requirement:
    id: str
    section: str
    title: str
    trigger: Expr
    exception: Expr = field(default_factory=lambda: NEVER)
    scope: Expr | None = None
    per_object: bool = True


def _mills_optional_window(series: str, lo: float, hi: float) -> Expr:
    """Rev B cumulus clauses read 'if surface electric field mills are
    available ... their measurements have remained between -100 and +500'.
    The clause is conditional on availability, so with no mills it is
    vacuously satisfied."""
    return Or(
        Not(FieldMills()),
        And(
            Sustained(series, ">=", lo, minutes(15), "E >= -100 V/m for 15 min"),
            Sustained(series, "<=", hi, minutes(15), "E <= +500 V/m for 15 min"),
        ),
    )


_CU_TOP_MILD = And(
    Cmp("top_temp_c", "<=", 5.0, "top <= +5 C"),
    Cmp("top_temp_c", ">", -5.0, "top > -5 C"),
)

_CU_EXC_1 = And(
    Not(Attr("top_colder_than_0c_last_3h", "top reached 0 C in last 3 h")),
    Attr("will_grow_colder_than_0c", "forecast growth colder than 0 C"),
    _mills_optional_window("efield_local", -100.0, 500.0),
)

_CU_EXC_2 = And(
    Not(Attr("refl_above_10dbz_last_3h", "reflectivity > +10 dBZ in last 3 h")),
    Not(Attr("top_colder_than_5c_last_3h", "top reached -5 C in last 3 h")),
    Attr("will_grow_colder_than_5c", "forecast growth colder than -5 C"),
    _mills_optional_window("efield_local", -100.0, 500.0),
)

_CU_EXC_3 = And(
    Not(Attr("producing_precip", "cloud producing precipitation")),
    FieldMills(),
    Sustained("efield_local", ">=", -100.0, minutes(15), "E >= -100 V/m for 15 min"),
    Sustained("efield_local", "<=", 500.0, minutes(15), "E <= +500 V/m for 15 min"),
)


_DEBRIS_SCOPE = And(
    IsType(CloudType.DEBRIS),
    Or(
        Attr("parent_had_part_colder_than_minus20", "parent had part colder than -20 C"),
        Attr("formed_by_thunderstorm", "formed by a thunderstorm"),
    ),
)


ENCODED: list[Requirement] = [
    Requirement(
        id="LLCCR 5",
        section="4.1.1.1",
        title="Lightning within 10 nmi of the flight path",
        per_object=False,
        trigger=Not(RecentLightning(10.0, minutes(30))),
        exception=And(
            Attr("producing_cloud_beyond_10nmi", "producing cloud > 10 nmi"),
            FieldMills(),
            Sustained("efield_max", "<", 1000.0, minutes(15),
                      "|E| < 1000 V/m for 15 min", absolute=True),
        ),
    ),
    Requirement(
        id="LLCCR 6",
        section="4.1.1.2",
        title="Lightning within or from a thunderstorm, path <= 10 nmi",
        scope=Attr("is_thunderstorm", "object is a thunderstorm"),
        trigger=And(
            Cmp("slant_min_nmi", "<=", 10.0, "slant <= 10 nmi"),
            Not(Elapsed(EventKind.LIGHTNING, minutes(30), "30 min since discharge")),
        ),
    ),
    Requirement(
        id="LLCCR 9",
        section="4.1.3.1",
        title="Cumulus cloud, flight path through the cloud",
        scope=IsType(CloudType.CUMULUS, CloudType.SMOKE_CUMULUS),
        trigger=And(
            Attr("intersects_corridor", "path through cloud"),
            Or(
                And(_CU_TOP_MILD, Not(Or(_CU_EXC_1, _CU_EXC_2, _CU_EXC_3))),
                Cmp("top_temp_c", "<=", -5.0, "top <= -5 C"),
            ),
        ),
    ),
    Requirement(
        id="LLCCR 10",
        section="4.1.3.2",
        title="Cumulus cloud, path 0-5 nmi, top <= -10 C",
        scope=IsType(CloudType.CUMULUS, CloudType.SMOKE_CUMULUS),
        trigger=And(
            Cmp("slant_min_nmi", ">", 0.0, "slant > 0 nmi"),
            Cmp("slant_min_nmi", "<=", 5.0, "slant <= 5 nmi"),
            Cmp("top_temp_c", "<=", -10.0, "top <= -10 C"),
        ),
    ),
    Requirement(
        id="LLCCR 11",
        section="4.1.3.3",
        title="Cumulus cloud, path 5-10 nmi, top <= -20 C",
        scope=IsType(CloudType.CUMULUS, CloudType.SMOKE_CUMULUS),
        trigger=And(
            Cmp("slant_min_nmi", ">", 5.0, "slant > 5 nmi"),
            Cmp("slant_min_nmi", "<=", 10.0, "slant <= 10 nmi"),
            Cmp("top_temp_c", "<=", -20.0, "top <= -20 C"),
        ),
    ),
    Requirement(
        id="LLCCR 12",
        section="4.1.4.1",
        title="Attached anvil, path through or <= 3 nmi",
        scope=And(
            IsType(CloudType.ATTACHED_ANVIL),
            Cmp("parent_coldest_top_temp_c", "<=", -10.0, "parent ever <= -10 C"),
        ),
        trigger=Cmp("slant_min_nmi", "<=", 3.0, "slant <= 3 nmi"),
        exception=And(
            Attr("portion_within_5nmi_all_below_0c", "portion <= 5 nmi all colder than 0 C"),
            MrrValid(),
            Cmp("mrr_max_within_1nmi_dbz", "<", 7.5, "MRR < +7.5 dBZ within 1 nmi"),
        ),
    ),
    Requirement(
        id="LLCCR 13",
        section="4.1.4.2",
        title="Attached anvil, path 3-5 nmi, 3 h after discharge",
        scope=And(
            IsType(CloudType.ATTACHED_ANVIL),
            Cmp("parent_coldest_top_temp_c", "<=", -10.0, "parent ever <= -10 C"),
        ),
        trigger=And(
            Cmp("slant_min_nmi", ">", 3.0, "slant > 3 nmi"),
            Cmp("slant_min_nmi", "<=", 5.0, "slant <= 5 nmi"),
            Not(Elapsed(EventKind.LIGHTNING, hours(3), "3 h since discharge")),
        ),
        exception=Attr("portion_within_5nmi_all_below_0c",
                       "portion <= 5 nmi all colder than 0 C"),
    ),
    Requirement(
        id="LLCCR 14",
        section="4.1.4.3",
        title="Attached anvil, path 5-10 nmi, 30 min after discharge",
        scope=And(
            IsType(CloudType.ATTACHED_ANVIL),
            Cmp("parent_coldest_top_temp_c", "<=", -10.0, "parent ever <= -10 C"),
        ),
        trigger=And(
            Cmp("slant_min_nmi", ">", 5.0, "slant > 5 nmi"),
            Cmp("slant_min_nmi", "<=", 10.0, "slant <= 10 nmi"),
            Not(Elapsed(EventKind.LIGHTNING, minutes(30), "30 min since discharge")),
        ),
        exception=Attr("portion_within_10nmi_all_below_0c",
                       "portion <= 10 nmi all colder than 0 C"),
    ),
    Requirement(
        id="LLCCR 15",
        section="4.1.5.1",
        title="Detached anvil, flight path through the cloud",
        scope=IsType(CloudType.DETACHED_ANVIL),
        trigger=Attr("intersects_corridor", "path through cloud"),
        exception=Or(
            And(
                Elapsed(EventKind.LIGHTNING, hours(4), "4 h since discharge"),
                Elapsed(EventKind.DETACHMENT, hours(3), "3 h since detachment"),
            ),
            And(
                Attr("portion_within_5nmi_all_below_0c",
                     "portion <= 5 nmi all colder than 0 C"),
                MrrValid(),
                Cmp("mrr_max_in_corridor_dbz", "<", 7.5, "MRR < +7.5 dBZ in path"),
            ),
        ),
    ),
    Requirement(
        id="LLCCR 16",
        section="4.1.5.2",
        title="Detached anvil, path 0-3 nmi",
        scope=IsType(CloudType.DETACHED_ANVIL),
        trigger=And(
            Cmp("slant_min_nmi", ">", 0.0, "slant > 0 nmi"),
            Cmp("slant_min_nmi", "<=", 3.0, "slant <= 3 nmi"),
        ),
        exception=Or(
            And(
                Attr("portion_within_5nmi_all_below_0c",
                     "portion <= 5 nmi all colder than 0 C"),
                MrrValid(),
                Cmp("mrr_max_within_1nmi_dbz", "<", 7.5, "MRR < +7.5 dBZ within 1 nmi"),
            ),
            And(
                Elapsed(EventKind.LIGHTNING, minutes(30), "30 min since discharge"),
                Or(
                    Elapsed(EventKind.LIGHTNING, hours(3),
                            "additional 2.5 h since discharge"),
                    And(
                        FieldMills(),
                        Sustained("efield_max", "<", 1000.0, minutes(15),
                                  "|E| < 1000 V/m for 15 min", absolute=True),
                        Sustained("refl_max_5nmi", "<", 10.0, minutes(15),
                                  "cloud reflectivity < +10 dBZ for 15 min"),
                    ),
                ),
            ),
        ),
    ),
    Requirement(
        id="LLCCR 17",
        section="4.1.5.3",
        title="Detached anvil, path 3-10 nmi, 30 min after discharge",
        scope=IsType(CloudType.DETACHED_ANVIL),
        trigger=And(
            Cmp("slant_min_nmi", ">", 3.0, "slant > 3 nmi"),
            Cmp("slant_min_nmi", "<=", 10.0, "slant <= 10 nmi"),
            Not(Elapsed(EventKind.LIGHTNING, minutes(30), "30 min since discharge")),
        ),
        exception=Attr("portion_within_10nmi_all_below_0c",
                       "portion <= 10 nmi all colder than 0 C"),
    ),
    Requirement(
        id="LLCCR 18",
        section="4.1.6.1",
        title="Debris cloud 3-hour period must be calculable",
        scope=_DEBRIS_SCOPE,
        trigger=Not(DebrisPeriodDeterminable()),
    ),
    Requirement(
        id="LLCCR 19",
        section="4.1.6.2",
        title="Debris cloud, flight path through the cloud",
        scope=_DEBRIS_SCOPE,
        trigger=And(
            Attr("intersects_corridor", "path through cloud"),
            Not(DebrisPeriodElapsed()),
        ),
        exception=And(
            Attr("portion_within_5nmi_all_below_0c",
                 "portion <= 5 nmi all colder than 0 C"),
            MrrValid(),
            Cmp("mrr_max_in_corridor_dbz", "<", 7.5, "MRR < +7.5 dBZ in path"),
        ),
    ),
    Requirement(
        id="LLCCR 20",
        section="4.1.6.3",
        title="Debris cloud, path 0-3 nmi",
        scope=_DEBRIS_SCOPE,
        trigger=And(
            Cmp("slant_min_nmi", ">", 0.0, "slant > 0 nmi"),
            Cmp("slant_min_nmi", "<=", 3.0, "slant <= 3 nmi"),
            Not(DebrisPeriodElapsed()),
        ),
        exception=Or(
            And(
                FieldMills(),
                Sustained("efield_max", "<", 1000.0, minutes(15),
                          "|E| < 1000 V/m for 15 min", absolute=True),
                Sustained("refl_max_5nmi", "<", 10.0, minutes(15),
                          "debris reflectivity < +10 dBZ for 15 min"),
            ),
            And(
                Attr("portion_within_5nmi_all_below_0c",
                     "portion <= 5 nmi all colder than 0 C"),
                MrrValid(),
                Cmp("mrr_max_within_1nmi_dbz", "<", 7.5,
                    "MRR < +7.5 dBZ within 1 nmi"),
            ),
        ),
    ),
    Requirement(
        id="LLCCR 21",
        section="4.1.7",
        title="Disturbed weather",
        trigger=And(
            Attr("intersects_corridor", "path through cloud"),
            Attr("associated_with_disturbed_weather", "associated with disturbed weather"),
            Attr("tops_colder_than_0c_in_system", "system tops colder than 0 C"),
            Or(
                Attr("moderate_precip_within_5nmi", "moderate precipitation <= 5 nmi"),
                Attr("bright_band_within_5nmi", "melting layer signature <= 5 nmi"),
            ),
        ),
    ),
    Requirement(
        id="LLCCR 22",
        section="4.1.8.1",
        title="Thick cloud layer",
        scope=IsType(CloudType.THICK_LAYER, CloudType.CIRRIFORM),
        trigger=And(
            Attr("intersects_corridor", "path through layer"),
            Cmp("layer_thickness_m", ">=", 1400.0, "thickness >= 1.4 km"),
            Attr("layer_spans_0_to_minus20", "part between 0 C and -20 C"),
        ),
        exception=Or(
            And(
                IsType(CloudType.CIRRIFORM),
                Not(Attr("ever_connected_to_convection", "ever connected to convection")),
                Attr("entirely_colder_than_15c", "entirely colder than -15 C"),
                Not(Attr("contains_liquid_water", "evidence of liquid water")),
            ),
            Not(Attr("refl_0dbz_within_5nmi", "0 dBZ present <= 5 nmi")),
            Not(Attr("mrr_ge_7p5_within_2nmi_last_hour",
                     "MRR >= +7.5 dBZ within 2 nmi in last hour")),
        ),
    ),
    Requirement(
        id="LLCCR 25",
        section="4.1.9.1",
        title="Smoke cumulus attached to or recently detached from plume",
        scope=IsType(CloudType.SMOKE_CUMULUS),
        trigger=And(
            Attr("intersects_corridor", "path through cloud"),
            Or(
                Attr("attached_to_smoke_plume", "attached to plume"),
                Not(Elapsed(EventKind.SMOKE_PLUME_DETACHMENT, minutes(60),
                            "60 min since plume detachment")),
            ),
        ),
    ),
    Requirement(
        id="LLCCR 27",
        section="4.1.10.1",
        title="Triboelectrification",
        per_object=False,
        trigger=And(
            Attr("corridor_penetrates_below_minus10c", "path through cloud colder than -10 C"),
            Cmp("velocity_at_penetration_ms", "<=", 910.0, "velocity <= 910 m/s"),
        ),
        exception=TriboExemption(),
    ),
]


MANIFEST: dict[str, str] = {
    "LLCCR 1": "4a ambiguity resolution",
    "LLCCR 2": "4b equipment and procedures",
    "LLCCR 3": "4c evaluate all available measurements",
    "LLCCR 4": "4d alternative criteria",
    "LLCCR 5": "4.1.1.1 lightning within 10 nmi",
    "LLCCR 6": "4.1.1.2 thunderstorm lightning",
    "LLCCR 7": "4.1.2.1 surface electric field 1000-1500 V/m",
    "LLCCR 8": "4.1.2.2 surface electric field >= 1500 V/m",
    "LLCCR 9": "4.1.3.1 cumulus through cloud",
    "LLCCR 10": "4.1.3.2 cumulus 0-5 nmi",
    "LLCCR 11": "4.1.3.3 cumulus 5-10 nmi",
    "LLCCR 12": "4.1.4.1 attached anvil <= 3 nmi",
    "LLCCR 13": "4.1.4.2 attached anvil 3-5 nmi",
    "LLCCR 14": "4.1.4.3 attached anvil 5-10 nmi",
    "LLCCR 15": "4.1.5.1 detached anvil through cloud",
    "LLCCR 16": "4.1.5.2 detached anvil 0-3 nmi",
    "LLCCR 17": "4.1.5.3 detached anvil 3-10 nmi",
    "LLCCR 18": "4.1.6.1 debris cloud 3-hour period",
    "LLCCR 19": "4.1.6.2 debris cloud through cloud",
    "LLCCR 20": "4.1.6.3 debris cloud 0-3 nmi",
    "LLCCR 21": "4.1.7 disturbed weather",
    "LLCCR 22": "4.1.8.1 thick cloud layer",
    "LLCCR 23": "4.1.8.2 cirriform exemption",
    "LLCCR 24": "4.1.8.3 reflectivity exemption",
    "LLCCR 25": "4.1.9.1 smoke plumes",
    "LLCCR 26": "4.1.9.2 smoke cumulus also assessed as cumulus",
    "LLCCR 27": "4.1.10.1 triboelectrification",
    "LLCCR 28": "4.1.10.2 triboelectrification exemption",
    "LLCCR 29": "4.2.1 radar reflectivity measurement",
    "LLCCR 30": "4.2.2 quantification of precipitation",
    "LLCCR 31": "4.2.3 computation of MRR",
    "LLCCR 32": "4.2.4 surface electric field measurement",
    "LLCCR 33": "4.2.5 non-transparent boundaries",
    "LLCCR 34": "4.2.6 slant distance from lightning",
    "LLCCR 35": "4.3 physically connected cloud scenarios",
}

# LLCCR 23 and 24 are encoded inside LLCCR 22's exception; 26 and 28 likewise
# ride along with 25 and 27. They count as covered.
COVERED_INLINE = {"LLCCR 23", "LLCCR 24", "LLCCR 26", "LLCCR 28"}


def encoded_ids() -> set[str]:
    return {r.id for r in ENCODED} | COVERED_INLINE


def missing_ids() -> list[str]:
    return sorted(
        set(MANIFEST) - encoded_ids(),
        key=lambda s: int(s.split()[1]),
    )
