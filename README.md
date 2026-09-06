# LLCC criterion evaluator and replay harness

Reference implementation of the evaluator layer for an automated launch
weather assessment against NASA-STD-4010B, targeting LC-39A.

Requires Python 3.12+. No third-party dependencies.

    python3 tests.py                 # evaluator, stdlib only
    python3 tests_geometry.py        # geometry, needs numpy + scipy
    python3 tests_segment.py         # segmentation, also needs scikit-image
    python3 tests_pipeline.py        # end to end, scan to verdict
    python3 tests_ingest.py          # beam geometry, QC, parsing
    python3 demo_pipeline.py         # regenerate docs/data from the pipeline
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
| `llcc/ingest/beam.py` | Beam geometry, MDS, observability, dual-pol QC (tested) |
| `llcc/ingest/nexrad.py` | Level II fetch, decode, gridding (UNVERIFIED) |
| `llcc/ingest/glm.py` | GLM fetch and flash footprints (partly UNVERIFIED) |
| `llcc/ingest/profile.py` | Soundings and model columns to a thermal profile |
| `llcc/pipeline.py` | Assembler: scan -> segment -> track -> geometry -> snapshot |
| `llcc/segment.py` | Hysteresis threshold, watershed, connection graph |
| `llcc/track.py` | Motion, association, split/merge events, lineage |
| `llcc/geometry.py` | Corridor, distance fields, MRR max filter, primitives (numpy+scipy) |
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

## Ingest

Split by what can be verified.

**Tested:** beam geometry, minimum detectable signal, the observability
mask, dual-pol QC, S3 listing parsing, GLM footprint construction, and
sounding parsing. All pure functions of arrays or text.

**UNVERIFIED:** every function that touches the network or needs Py-ART or
netCDF4. They are written against the documented APIs but have never been
run against a real file. Marked in the source. Shake them out on first use;
do not assume they work because the tests pass. The Py-ART field names and
the volume timestamp parse are the likeliest to bite.

    pip install arm-pyart netCDF4
    python3 run_live.py --time "2026-09-04T21:04:00" --sounding xmr.txt

Points worth knowing:

- The observability mask is real geometry: 4/3-earth beam height, beam width
  growing to about 1.6 km at 100 km, and sensitivity falling as
  20*log10(range). Past the range where 0 dBZ drops below the noise floor,
  an empty gate means nothing, so it is UNOBSERVABLE rather than clear. The
  cone of silence falls out of the tilt list rather than being special-cased.
- That 1.6 km of beam width is roughly 10 C in a Florida summer profile --
  the entire gap between the -10 C and -20 C criterion tiers. It should
  propagate into the display as a band, not a number.
- Dual-pol QC runs on polar gates *before* gridding. Filtering afterwards
  smears non-meteorological returns into neighbouring cells instead of
  removing them. Without dual-pol the filter degrades to a finite-value
  check, which is much weaker -- the output says so rather than implying the
  QC ran.
- GLM uses the union of group footprints, never the flash centroid. The
  centroid is energy-weighted and sits well inside the true extent, which
  would systematically understate distance to the flight path. Navigation
  and parallax error are handled by dilating the footprint rather than
  correcting it: auditable, and it shortens no standoff.
- Ingest a much wider lightning window than the criteria need. A cell that
  flashed 40 nmi offshore an hour ago still governs 4.1.1.2 and every anvil
  clock; objects arriving without that history are stuck as indeterminate.

## Pipeline

`llcc/pipeline.py` closes the loop. A gridded scan goes in, a `WorldSnapshot`
comes out, and the evaluator consumes it exactly as it consumes a scenario
file -- live data and archived replay are interchangeable inputs, which is
what fixing the contract early bought.

    refl, observable  ->  segment  ->  track  ->  geometry  ->  snapshot

Tracker events become world events: a split emits a detachment, a top
collapsing warmer than -10 C emits the 4.1.6.1b basis. That is how a
tracking decision reaches a three-hour clock.

**Classification is a hook, not a component.** The default labels every
object cumulus, which carries the widest standoffs in section 4.1. That is
not a claim about what the cloud is; it is a claim about what we are willing
to rule out, which is nothing. A real classifier then earns its keep by
*relaxing* that assumption for objects it can positively identify, so the
system degrades toward safe rather than toward unknown.

`demo_pipeline.py` runs a synthetic afternoon through the whole chain and
writes `docs/data/`. Nothing in it is hand-authored: reflectivity fields go
in, and the display raster is built from the very array the segmenter
consumed, so the picture and the verdict cannot disagree.

## Segmentation and tracking

**Connected components are not objects.** Section 4.3 requires that
physically connected clouds still be assessed individually until they are no
longer distinguishable, and 4.3a-d mandate independence for the common
pairings. A raw component label dissolves the individuals at exactly the
moment the standard says not to. So identity comes from watershed basins
seeded on reflectivity cores, and physical connection is a *graph edge*
between objects sharing a 0 dBZ component. One component can hold three
pairwise-connected objects, which is 4.3e implemented literally.

Hysteresis seeds on cores rather than on every local bump, because a hard
0 dBZ cut makes objects blink in and out at range and every blink looks like
a split or a merge -- which would spuriously start a three-hour clock. A
component with no core still becomes one object, since thin anvil and cirrus
never reach the seed threshold but are still clouds.

Tracking is TITAN-style: advect by storm motion, match on volume overlap,
treat splits and merges as typed events. A split timestamp *is* the
detachment time feeding 4.1.6.1a and 4.1.5.1a, so association accuracy moves
real holds. Split children inherit the parent's top history and coldest-ever
top; merges take the union of provenance.

Two asymmetries, both safety properties rather than tuning choices:

- Connection is declared on first evidence, disconnection only after it has
  persisted. Declaring detachment late lengthens a hold; declaring
  reconnection late would let a still-attached anvil be assessed as
  detached.
- A track appearing at the domain edge or after a data gap carries unknown
  provenance. It cannot satisfy any "for the previous 3 hours" test, and
  history shorter than the window returns unknown rather than false.

## Geometry engine

`llcc/geometry.py` produces the numbers the evaluator consumes. Requires
numpy and scipy; the evaluator itself does not, so `tests.py` still runs
anywhere.

The corridor does not move during a count, so distance is precomputed as a
field over the grid: every query is a lookup plus a reduction over an
object's voxels. Two separate fields, because the standard uses two
metrics and mixing them is an easy and dangerous bug. LLCCR 12 uses both in
adjacent clauses -- 12a is slant distance to the anvil, 12b is horizontal
distance for the MRR test. `d_horiz` also serves lightning, since GLM has no
altitude and LLCCR 34b collapses slant to horizontal between projections.

MRR is a max filter, not a loop. Section 4.2.3a bounds the volume below by
the 0 C level and above by 20 km MSL regardless of the evaluation point's
altitude, so MRR is a **2D field**: one slab maximum down the column, then a
separable box maximum in x and y. Three linear passes give MRR at every grid
point at once. The half-width rounds up, never to nearest -- under-covering
would shrink MRR and make an "MRR < +7.5 dBZ" exception easier to satisfy.

Section 4.2.3d is applied as a validity mask rather than a check: points
within 10 nmi of a 35 dBZ core at or above the 0 C level, or within 10 nmi
of lightning in the last five minutes, are marked unusable, so an exception
resting on them is simply unavailable.

`tests_geometry.py` checks the fast paths against brute-force oracles. A
nested loop over every voxel pair is exact, and the naive nested-box maximum
is exact; both are far too slow for production but settle correctness
outright on small grids. Those two comparisons catch essentially every
indexing and anisotropy bug.

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
