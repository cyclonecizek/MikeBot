"""Emit the snapshot/verdict JSON contract for the static display.

Writes a replay bundle (every frame, for the stepper) plus the latest frame
as standalone snapshot.json and verdict.json, which is the shape a live
backend should publish.
"""
import argparse
import json
from pathlib import Path

from llcc.radar import synth_field
from llcc.replay import load_scenario
from llcc.serialize import (SCHEMA_VERSION, snapshot_to_dict, verdict_to_dict,
                            write_json)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("scenario")
    p.add_argument("-o", "--outdir", default="docs/data")
    p.add_argument("--assume-manifest-complete", action="store_true")
    args = p.parse_args()

    raw = json.loads(Path(args.scenario).read_text())
    scen = load_scenario(args.scenario)
    out = Path(args.outdir)

    frames = []
    for frame in scen.frames:
        snapshot, baseline, overridden, applications, diff = scen.run_pair(
            frame, assume_manifest_complete=args.assume_manifest_complete)
        obj_dicts = {oid: o.__dict__ for oid, o in snapshot.objects.items()}
        radar = synth_field(obj_dicts, frame.time.isoformat())
        snap_doc = snapshot_to_dict(snapshot, pad=scen.pad,
                                    azimuth_deg=scen.azimuth_deg, radar=radar)
        snap_doc["trajectory"] = raw.get("trajectory", [])
        frames.append({
            "snapshot": snap_doc,
            "verdict": verdict_to_dict(overridden, applications, diff),
            "baseline_verdict": verdict_to_dict(baseline),
        })

    write_json(out / "replay.json", {
        "schema": SCHEMA_VERSION,
        "kind": "replay",
        "name": scen.name,
        "pad": scen.pad,
        "azimuth_deg": scen.azimuth_deg,
        "manifest_gate_suppressed": args.assume_manifest_complete,
        "frames": frames,
    })
    write_json(out / "snapshot.json", frames[-1]["snapshot"])
    write_json(out / "verdict.json", frames[-1]["verdict"])
    print(f"wrote {len(frames)} frames to {out}/replay.json")
    print(f"wrote {out}/snapshot.json and {out}/verdict.json")


if __name__ == "__main__":
    main()
