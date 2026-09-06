"""Explain a verdict: which requirements block, and on what evidence.

An indeterminate result means some leaf of the evidence tree came back
unknown. This walks the trace and names those leaves, which is usually the
fastest way to tell a missing input from a real weather condition.

    python3 explain.py                      # docs/data/replay.json
    python3 explain.py --frame -1 --deep    # last frame, full traces
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def unknown_leaves(node, out, path=""):
    """Leaves of the evidence tree whose value is unknown."""
    label = f"{path} > {node['label']}" if path else node["label"]
    kids = node.get("children") or []
    if not kids:
        if node["value"] == "unknown":
            out.append((node["label"], node.get("detail", "")))
        return
    for kid in kids:
        unknown_leaves(kid, out, label)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="docs/data/replay.json")
    ap.add_argument("--frame", type=int, help="only this frame index")
    ap.add_argument("--deep", action="store_true", help="print full traces")
    args = ap.parse_args()

    bundle = json.loads(Path(args.path).read_text())
    frames = bundle["frames"]
    if args.frame is not None:
        frames = [frames[args.frame]]

    tally: Counter = Counter()
    causes: Counter = Counter()

    for frame in frames:
        snap, verdict = frame["snapshot"], frame["verdict"]
        blocking = [r for r in verdict["results"] if r["state"] != "GO"]
        objs = len([k for k in snap["objects"] if k != "DOMAIN"])
        print(f"\n{snap['time'][11:19]}Z  {verdict['state']}  "
              f"{objs} object(s)  {len(blocking)} blocking")

        for r in sorted(blocking, key=lambda x: x["requirement_id"]):
            name = r.get("rule_name", r["requirement_id"])
            tally[f"{name} ({r['requirement_id']}) {r['state']}"] += 1
            print(f"  {name:<28} {r['state']:<14} [{r['unit_id']}]"
                  f"   {r['requirement_id']} {r['section']}")
            if args.deep:
                print("    trigger:")
                print(_render(r["trigger"], 6))
                print("    exception:")
                print(_render(r["exception"], 6))
            else:
                for side in ("trigger", "exception"):
                    leaves: list = []
                    unknown_leaves(r[side], leaves)
                    for label, detail in leaves:
                        causes[label] += 1
                        note = f"  ({detail})" if detail else ""
                        print(f"      unknown in {side}: {label}{note}")

    print("\n" + "=" * 60)
    print("blocking by requirement:")
    for key, n in tally.most_common():
        print(f"  {n:>3}x  {key}")
    if causes:
        print("\nunknown evidence, most common first:")
        for label, n in causes.most_common(12):
            print(f"  {n:>3}x  {label}")


def _render(node, indent=0):
    pad = " " * indent
    line = f"{pad}[{node['value']}] {node['label']}"
    if node.get("detail"):
        line += f"  ({node['detail']})"
    out = [line]
    for kid in node.get("children") or []:
        out.append(_render(kid, indent + 2))
    return "\n".join(out)


if __name__ == "__main__":
    main()
