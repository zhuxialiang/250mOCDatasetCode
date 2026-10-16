"""
Balance_Samples.py -- balancing of the candidate window list.

Pipeline stage: after Find_4Class_Sample.py (which outputs `sample_records.csv`)
and before/around Sample_Crop.py. The balancing is carried out in two stages:

  Stage 1 (class balancing, raw -> balanced):
      Only the redundant PW and HZ classes are subsampled to equalize the
      share of valid pixels across classes; RH and RB windows are kept
      entirely. A target number of windows is defined per class, allocated to
      months by the largest remainder method, and within each month the
      windows with the highest valid-pixel ratio are retained. The output
      (18,706 windows) is the crop list used by Sample_Crop.py.

  Stage 2 (PW temporal balancing, after cropping):
      The dominant PW class is reduced from 3,683 to about 1,000 windows by
      equidistant sampling over the time-sorted candidates, preserving its
      yearly-monthly distribution. The subsequent pixel-count quality
      filtering (RH < 500, RB < 1,000, PW < 8,000, HZ < 1,500 valid pixels)
      and the NSOAS-calibration cleanup require the cropped samples and are
      not reproduced by this script; the authoritative final list
      (`sample_records_final.csv`, 13,034 windows) reflects all post-crop
      filters.

Note on exact reproducibility: the balancing was originally carried out with
the same rules plus manual adjustments. The published sample lists
(`sample_records_balanced.csv`, `sample_records_final.csv`) are authoritative;
this script reproduces the reproducible rules and serves as the reference
implementation.

Usage:
    python Balance_Samples.py --input ..\S2\sample_records.csv \
        --output ..\S2\sample_records_balanced.csv \
        --stage 1 --targets PW=3683,HZ=5116
    python Balance_Samples.py --input ..\S2\sample_records_balanced.csv \
        --output <intermediate_pw_sampled.csv> --stage 2 --targets PW=1000
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

import numpy as np


def load_records(csv_path):
    """Load the sample-record CSV."""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            m = re.search(r"(20\d{6})_(\d{6})", r["chl_file"])
            rows.append({
                "id": r["id"],
                "chl_file": r["chl_file"],
                "sample_type": r["sample_type"],
                "bbox": r["bbox"],
                "valid_ratio": float(r["valid_ratio"]),
                "ym": m.group(1)[:6] if m else "",
            })
    return rows


def largest_remainder(counts, total):
    """Allocate `total` items to groups proportionally to `counts`."""
    n = sum(counts.values())
    if n == 0:
        return {k: 0 for k in counts}
    base = {k: counts[k] * total / n for k in counts}
    floor = {k: int(v) for k, v in base.items()}
    rem = {k: v - int(v) for k, v in base.items()}
    leftover = total - sum(floor.values())
    for k in sorted(rem, key=lambda x: -rem[x])[:leftover]:
        floor[k] += 1
    return floor


def pick_topk_by_ratio(rows, take):
    """Keep the `take` windows with the highest valid-pixel ratio (stable)."""
    rows = sorted(rows, key=lambda r: (-r["valid_ratio"], r["id"]))
    return rows[:take]


def balance_stage1(rows, targets):
    """Stage 1: subsample PW/HZ by monthly quota + valid-ratio priority.

    RH and RB are kept entirely. For PW and HZ, the per-month target is obtained
    by allocating the class target with the largest remainder method, then the
    highest-valid-ratio windows of that month are retained.
    """
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["sample_type"]].append(r)

    kept = []
    for typ, trows in by_type.items():
        if typ in ("RH", "RB"):
            kept.extend(trows)                       # keep all
            continue
        target = targets[typ]
        if target is None:
            kept.extend(trows)
            continue
        mcnt = Counter(r["ym"] for r in trows)
        monthly_target = largest_remainder(mcnt, target)
        by_month = defaultdict(list)
        for r in trows:
            by_month[r["ym"]].append(r)
        for ym, mrows in by_month.items():
            kept.extend(pick_topk_by_ratio(mrows, monthly_target.get(ym, 0)))
    return kept


def balance_stage2(rows, targets):
    """Stage 2: equidistant (time-balanced) subsampling of the PW class.

    Only the PW class is subsampled here (default target 1,000 windows), by
    equidistant sampling over the time-sorted candidates to preserve the
    yearly-monthly distribution. RH/RB/HZ and classes without a target are
    kept entirely.
    """
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["sample_type"]].append(r)

    kept = []
    for typ, trows in by_type.items():
        target = targets.get(typ)
        if target is None:
            kept.extend(trows)
            continue
        trows = sorted(trows, key=lambda r: (r["ym"], r["id"]))
        idx = np.linspace(0, len(trows) - 1, min(target, len(trows))).astype(int)
        kept.extend(trows[i] for i in idx)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Input sample-record CSV")
    ap.add_argument("--output", required=True, help="Output CSV")
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2],
                    help="1 = class balancing (PW/HZ), 2 = PW temporal (equidistant) sampling")
    ap.add_argument("--targets", required=True,
                    help="Per-class target window counts, e.g. PW=3683,HZ=5116 (stage 1) or PW=1000 (stage 2)")
    args = ap.parse_args()

    targets = {}
    for kv in args.targets.split(","):
        k, v = kv.split("=")
        targets[k.strip()] = int(v.strip())

    rows = load_records(args.input)
    print(f"Input : {len(rows)} records")

    if args.stage == 1:
        out = balance_stage1(rows, targets)
    else:
        out = balance_stage2(rows, targets)

    # renumber ids
    for i, r in enumerate(sorted(out, key=lambda r: (r["sample_type"], r["ym"], r["chl_file"]))):
        r["id"] = str(i + 1)

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["id", "chl_file", "sample_type", "bbox", "valid_ratio"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows({k: r[k] for k in fieldnames} for r in out)

    print(f"Output: {len(out)} records -> {args.output}")
    for typ, cnt in Counter(r["sample_type"] for r in out).items():
        print(f"  {typ}: {cnt}")


if __name__ == "__main__":
    main()
