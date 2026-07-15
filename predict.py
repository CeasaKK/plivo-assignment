"""Final deliverable: load a saved model and predict p_eot for each pause.

Matches baseline.py's CLI exactly:
    python predict.py --data_dir ../eot_data/english --out predictions.csv

This ONLY runs inference. It loads the model fit by fit_model.py (joblib) and
does NOT refit on the data it is predicting. Features are causal: each pause is
scored from audio[0:pause_start] via eot_features.extract_features, which never
reads pause_end. pause_end is not read here at all.
"""
import argparse
import csv
import os

import numpy as np
import joblib

from features import load_wav
from eot_features import extract_features, FEATURE_NAMES

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eot_model.joblib")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    artifact = joblib.load(args.model)
    models = artifact["models"]
    if artifact.get("feature_names") != FEATURE_NAMES:
        raise SystemExit("feature mismatch: model was trained with a different "
                         "feature set than eot_features.py currently produces.")

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, keys = [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        # causal: only pause_start and pause_index are read; pause_end is ignored
        X.append(extract_features(x, sr, float(r["pause_start"]),
                                  pause_index=float(r["pause_index"])))
        keys.append((r["turn_id"], r["pause_index"]))
    X = np.array(X)

    # ensemble = average of the saved models' probabilities
    p = np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pv in zip(keys, p):
            w.writerow([tid, pi, f"{pv:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
