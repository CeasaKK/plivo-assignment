"""Diagnostics: hold-duration frontier, per-feature AUC, error analysis."""
import argparse
import csv
import os
import numpy as np

from eot_common import build_matrix
from eot_features import FEATURE_NAMES


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else float("nan")


def error_dump(data_dir, pred_csv, delay):
    """Show the cases that hurt the score at operating delay `delay`:
    long-hold FALSE ALARMS (high p) and eot MISSES (low p), with features."""
    X, y, groups, keys = build_matrix(data_dir)
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    durs = {(r["turn_id"], int(r["pause_index"])): float(r["pause_end"]) - float(r["pause_start"])
            for r in rows}
    preds = {}
    with open(pred_csv) as f:
        for r in csv.DictReader(f):
            preds[(r["turn_id"], int(r["pause_index"]))] = float(r["p_eot"])
    p = np.array([preds[k] for k in keys])
    show = ["e_decay_slope", "e_std_last300", "trail_sil_s", "f0_slope_st",
            "f0_range_st", "f0_last_minus_med_st", "voiced_ratio_last", "pause_index"]
    ci = [FEATURE_NAMES.index(n) for n in show]

    eot = [(p[i], keys[i], X[i]) for i in range(len(y)) if y[i] == 1]
    longhold = [(p[i], keys[i], X[i], durs[keys[i]]) for i in range(len(y))
                if y[i] == 0 and durs[keys[i]] > delay]
    print(f"\n=== operating delay={delay}s: long-holds(dur>{delay}) that would false-cut if they fire ===")
    print("  worst (highest-p) long-holds -- these force the threshold up:")
    for pv, k, xv, d in sorted(longhold, reverse=True)[:8]:
        feats = "  ".join(f"{n}={xv[j]:.2f}" for n, j in zip(show, ci))
        print(f"   p={pv:.2f} dur={d:.2f} {k}  {feats}")
    print("  eot MISSES (lowest-p eots) -- these cost 1.6s timeouts:")
    for pv, k, xv in sorted(eot)[:8]:
        feats = "  ".join(f"{n}={xv[j]:.2f}" for n, j in zip(show, ci))
        print(f"   p={pv:.2f}          {k}  {feats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--errors", help="oof pred csv to run error analysis on")
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    if args.errors:
        error_dump(args.data_dir, args.errors, args.delay)
        return

    # hold-duration frontier: how many turns have a hold longer than d?
    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    turn_maxhold = {}
    for r in rows:
        if r["label"] == "hold":
            dur = float(r["pause_end"]) - float(r["pause_start"])
            turn_maxhold[r["turn_id"]] = max(turn_maxhold.get(r["turn_id"], 0.0), dur)
    turns = len({r["turn_id"] for r in rows})
    mh = np.array(list(turn_maxhold.values()))
    print(f"turns={turns}  turns_with_holds={len(mh)}")
    print("hold-dur frontier: turns whose LONGEST hold exceeds d (these can be cut at delay d):")
    for d in [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]:
        n = int((mh > d).sum())
        print(f"   d={d:.1f}s -> {n:3d} turns ({100*n/turns:.0f}%) at risk")

    # per-feature univariate AUC (higher => feature ranks eot above hold)
    X, y, groups, keys = build_matrix(args.data_dir)
    # long-hold mask: eot(1) vs holds whose dur>0.5s (0). short holds excluded --
    # those are the ones that actually cause cutoffs, so this AUC drives the score.
    durs = {}
    for r in rows:
        durs[(r["turn_id"], int(r["pause_index"]))] = float(r["pause_end"]) - float(r["pause_start"])
    is_long_hold = np.array([(y[i] == 0 and durs[keys[i]] > 0.5) for i in range(len(y))])
    keep = (y == 1) | is_long_hold
    yl = y[keep]
    print(f"\nper-feature AUC: all-holds | long-holds-only(dur>0.5, n={int(is_long_hold.sum())}):")
    scored = []
    for i, name in enumerate(FEATURE_NAMES):
        a = auc(y, X[:, i])
        al = auc(yl, X[keep, i])
        scored.append((abs(al - 0.5), a, al, name))
    for strength, a, al, name in sorted(scored, reverse=True):
        print(f"   {name:22s} all={a:.3f}  long={al:.3f}  |long-signal|={strength:.3f}")


if __name__ == "__main__":
    main()
