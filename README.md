# LLCC criterion evaluator and replay harness

Reference implementation of the evaluator layer for an automated launch
weather assessment against NASA-STD-4010B, targeting LC-39A.

Requires Python 3.12+. No third-party dependencies.

    python3 tests.py
    python3 run_replay.py scenarios/lc39a_20260904.json --check
    python3 run_replay.py scenarios/lc39a_20260904.json -v --assume-manifest-complete

## Layout

| File | Role |
|---|---|
| `llcc/tvl.py` | Kleene three-valued logic |
| `llcc/world.py` | Thermal profile, cloud objects with lineage, event log, sampled series |
| `llcc/expr.py` | Expression AST and the leaf primitives |
| `llcc/requirements.py` | Encoded LLCCRs and the Appendix A manifest |
| `llcc/evaluate.py` | Section 4.3 pre-pass, requirement walker, mission reduction |
| `llcc/replay.py` | Scenario loader and regression check |
| `llcc/radar.py` | Gridded reflectivity: encoding, and a synthetic field for scenarios |
| `llcc/overrides.py` | LWO reclassification: apply pass, validation, verdict diff |
| `llcc/serialize.py` | snapshot.json / verdict.json contract |
| `emit_verdict.py` | Writes the display data into `docs/data/` |
| `docs/index.html` | Static display, no build step |

## Design commitments

`permitted = NOT trigger OR exception` in three-valued logic. An unknown
trigger and an unknown exception both propagate to unknown, and anything
not definitely true blocks. Conservative direction falls out of the
semantics rather than out of per-requirement reasoning.

Requirements are data, not code, so the encoding can be diffed against the
document text and every evaluation yields an evidence trace.

Timers are derived from the event log on every call, never stored. A flash
inherited across a lineage split retroactively extends the child's clocks,
which a decrementing counter would miss.

Sustained predicates check coverage as well as values. A data gap inside a
15-minute window yields unknown, not true.

Requirements not yet encoded force a global indeterminate via the manifest.
`--assume-manifest-complete` suppresses that for development and prints a
warning; it should never be set in an operational path.

Field mills are out of scope, so `FieldMills` evaluates false and every
mill-dependent exception is unavailable without special-casing. The Rev B
cumulus clauses that read "if mills are available" remain usable, because
that clause is conditional on availability.

## Radar field

The display renders the reflectivity grid underneath the object overlay, in
both the plan view and the vertical slice along the flight azimuth. It is
shipped in `snapshot.json` as base64 int8 dBZ, about 21 kB for a 128x128
plan view, and needs no image library on either end.

The raster must be the same field the segmentation consumed, at the same
valid time. Third-party radar tiles are tempting and free, but then the
picture and the verdict can disagree, which is worse than showing nothing.
`llcc.radar.from_grids` takes the arrays your gridding step already
produced.

Two sentinels, and the distinction matters:

    NO_ECHO (-127)       observed, nothing there. Renders transparent.
    UNOBSERVABLE (-128)  beam overshoot, cone of silence, or 0 dBZ below
                         minimum detectable signal at this range. Renders
                         hatched grey, never as clear air.

Because cloud boundaries are the 0 dBZ contour, values below zero are hidden
by default, so the visible edge of the coloured region *is* the boundary the
segmenter used. The "show below 0 dBZ" toggle reveals the sub-zero returns
when you want to see how close something is to crossing.

Scenario rasters are synthesised from the object geometry, so raster and
objects agree by construction. In the real system the causality runs the
other way -- objects are segmented out of the measured field -- but either
way the two must never disagree.

## Officer reclassification

A certified LWO can relabel a misidentified object. Overrides are a separate
input document, applied by a pass that produces a new snapshot, so the
evaluator stays pure and both verdicts remain reproducible.

```json
"overrides": [{
  "object_id": "ly1",
  "cloud_type": "detached_anvil",
  "original_type": "thick_layer",
  "issued_at": "2026-09-04T21:09:00",
  "issued_by": "LWO/AR",
  "justification": "Fibrous texture and downwind position from cell1.",
  "facts": {
    "detachment_time": "2026-09-04T20:40:00",
    "parent_coldest_top_temp_c": -32.0
  },
  "ttl_seconds": 3600
}]
```

Embed them in a scenario for replay, or load a standalone file with
`llcc.overrides.load_overrides`.

**Companion facts.** A type label asserts more than a name. Detached anvil
requires a detachment time and a parent that reached -10 C, because
LLCCR 15a, 16b and 17 depend on them. Supply them or the object degrades to
unclassified, which forces the conservative branch. A detachment time is
written into the event log so the clocks actually see it.

**Statuses.** `applied`, `degraded` (facts missing), `stale` (the tracker
relabelled the object, so the basis changed), `expired` (TTL lapsed without
re-affirmation), `orphaned` (object gone).

**Direction.** Reclassification swaps which requirements apply, so it can
relax as easily as restrict. Both verdicts are computed and diffed. A
requirement that starts or stops applying counts as neutral; only movement
up or down the blocking scale counts. Relaxing overrides without a
justification are listed in `unjustified_relaxations` and flagged in red on
the display.

**Section 4.3 cascade.** Relabelling changes which carve-out governs the
whole connected cluster, so the result may not be what the officer expects.
Read the diff before committing.

## Publishing

GitHub Pages hosts the display but cannot run anything. The split:

    backend (always on)          ingest, gridding, tracking, geometry, evaluate
        |  snapshot.json + verdict.json
    object store (R2/S3, CORS)   short cache, live artifacts
        |  fetch
    GitHub Pages                 docs/index.html, pure renderer

`docs/index.html` loads `data/replay.json` if present and gives a frame
stepper; otherwise it polls `data/snapshot.json` and `data/verdict.json`
every 30 s. Point those fetches at the object store for a live feed.

    python3 emit_verdict.py scenarios/lc39a_20260904.json --assume-manifest-complete
    python3 -m http.server 8000 --directory docs

Set Pages to build from Actions and `.github/workflows/pages.yml` will run
the tests, run the regression check, emit the data, and deploy `docs/`.

Do not drive a live display from scheduled Actions. The 5-minute cron
minimum, queueing, and dropped runs cannot hold a 4.5-minute volume scan
cadence, and committing JSON every few minutes grows the repo without bound.

## Vehicle configuration

Section 4.1.10.2 exempts a vehicle from the triboelectrification criterion
on one of two bases: 28a, surface treatment meeting the resistivity and
bonding limits; or 28b, demonstration by test or analysis. Neither is a
waiver under section 1.3 -- both are exemptions the standard grants, and
both require evidence on file.

```json
"vehicle": {
  "name": "reference vehicle",
  "triboelectric_exemption": "4.1.10.2b",
  "triboelectric_basis": "vehicle electrostatic discharge analysis on file"
}
```

`triboelectric_exemption` has no default. Omitting it yields indeterminate,
not exempt and not held, so a missing vehicle configuration fails loudly.
`"none"` claims no exemption and lets LLCCR 27 block normally. The basis
string is carried into the evidence trace.

The exemption covers 4.1.10.1 only. Penetrating cloud colder than -10 C
still matters to the cumulus, anvil, debris, and thick-layer criteria; it
is only the ice-impact charging hazard that the exemption addresses.

## Status

22 of 35 requirements encoded (18 explicit, 4 inline as exceptions).
The debris cloud family (LLCCR 18-20) is in, including the three-way
'latest of' period in 4.1.6.1. Remaining gaps are the surface electric field
criteria (7, 8), the section 4 preamble requirements (1-4), and the
measurement requirements (29-35), which belong with the geometry and radar
layers.
Geometry values on `CloudObject` are supplied by the scenario file rather
than by a distance-field engine, so a failing test points at the evaluator.
Classification is not implemented; unclassified objects stay in scope under
LLCCR 1 and report indeterminate.
