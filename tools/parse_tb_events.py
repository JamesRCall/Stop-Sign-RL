"""Parse TensorBoard event files and export selected scalars to JSON."""

import argparse
import json
from pathlib import Path


def parse_events(event_path: Path, tags: list[str]) -> dict[str, list[dict]]:
    from tensorboard.backend.event_processing import event_accumulator

    ea = event_accumulator.EventAccumulator(str(event_path))
    ea.Reload()

    out = {}
    available = set(ea.Tags().get("scalars", []))
    for tag in tags:
        if tag not in available:
            continue
        rows = []
        for ev in ea.Scalars(tag):
            rows.append({"step": int(ev.step), "wall_time": float(ev.wall_time), "value": float(ev.value)})
        out[tag] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse TensorBoard scalar events to JSON.")
    ap.add_argument("--event", required=True, help="Path to events.out.tfevents.* file")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--tags", nargs="+", required=True, help="Scalar tags to export")
    args = ap.parse_args()

    event_path = Path(args.event)
    out_path = Path(args.out)
    data = parse_events(event_path, args.tags)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
