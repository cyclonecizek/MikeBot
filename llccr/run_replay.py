"""CLI: replay a scenario through the criterion evaluator."""
import argparse
from llcc.evaluate import format_verdict
from llcc.replay import expectations_report, load_scenario


def main() -> None:
    p = argparse.ArgumentParser(description="Replay an LLCC scenario.")
    p.add_argument("scenario")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print the full evidence trace for blocking requirements")
    p.add_argument("--assume-manifest-complete", action="store_true",
                   help="development only: suppress the Appendix A manifest gate")
    p.add_argument("--check", action="store_true",
                   help="compare against the scenario's declared expectations")
    args = p.parse_args()

    scen = load_scenario(args.scenario)
    print(f"{scen.name}\n{scen.pad}  azimuth {scen.azimuth_deg:.0f} deg  "
          f"field mills: {'yes' if scen.field_mills_available else 'no'}\n")

    if args.assume_manifest_complete:
        print("WARNING: Appendix A manifest gate suppressed. Unencoded "
              "requirements are being skipped, not evaluated.\n")

    for verdict in scen.run(assume_manifest_complete=args.assume_manifest_complete):
        print(format_verdict(verdict, verbose=args.verbose))
        print()

    if args.check:
        print("expectations:")
        lines = expectations_report(scen, args.scenario)
        print("\n".join(lines) if lines else "  none declared")


if __name__ == "__main__":
    main()
