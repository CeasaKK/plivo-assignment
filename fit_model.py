"""Fit the FINAL end-of-turn model and save it for predict.py.

We ship ONE JOINT model trained on English + Hindi together. Held-out (GroupKFold
OOF) scores showed the joint model beats per-language models on both tracks --
doubling the data regularizes a 26-feature model that otherwise overfits badly
(in-sample AUC ~0.99 vs OOF ~0.66-0.76), and the prosodic features generalize
across the two languages. See RUNLOG.md for the comparison.

The model is an average-probability ensemble of a gradient-boosted tree and a
logistic-regression pipeline (the ensemble is more stable at the operating point
than either alone).

    python fit_model.py --data_dirs ../eot_data/english ../eot_data/hindi \
                        --out eot_model.joblib
"""
import argparse

import numpy as np
import joblib

from eot_common import build_matrix, make_model
from eot_features import FEATURE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+",
                    default=["../eot_data/english", "../eot_data/hindi"])
    ap.add_argument("--out", default="eot_model.joblib")
    args = ap.parse_args()

    Xs, ys = [], []
    for d in args.data_dirs:
        X, y, groups, keys = build_matrix(d)   # no durations: none used as features
        Xs.append(X)
        ys.append(y)
        print(f"  loaded {d}: {len(y)} pauses ({int(y.sum())} eot / {int((1 - y).sum())} hold)")
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    models = []
    for kind in ("hgb", "logreg"):
        clf = make_model(kind)
        clf.fit(X, y)
        models.append(clf)
    artifact = {
        "models": models,          # probabilities are averaged at predict time
        "feature_names": FEATURE_NAMES,
        "trained_on": list(args.data_dirs),
    }
    joblib.dump(artifact, args.out)
    print(f"saved joint ensemble ({len(models)} models, {X.shape[1]} features, "
          f"{len(y)} pauses) -> {args.out}")


if __name__ == "__main__":
    main()
