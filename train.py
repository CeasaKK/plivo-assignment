"""Dev-loop trainer: writes HONEST out-of-fold predictions for score.py.

Unlike the starter skeleton (which refit on all data and predicted in-sample),
this writes cross-validated out-of-fold predictions using GroupKFold on turn_id
(a turn is never split across folds), so the scored mean-delay reflects true
generalization to unseen turns. The final deliverable is predict.py, which loads
a saved model (see fit_model.py) and only runs inference.

    python train.py --data_dir ../eot_data/english --out mine_en.csv
    python train.py --data_dir ../eot_data/english --out mine_en.csv --model logreg
"""
import argparse

import numpy as np
from sklearn.model_selection import GroupKFold

from eot_common import build_matrix, make_model, write_preds, long_hold_weights, fit_weighted
# The real feature work lives in eot_features.extract_features (shared by train,
# fit_model and predict so fit-time and inference-time features are identical).
from eot_features import FEATURE_NAMES, extract_features  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="hgb", choices=["hgb", "logreg"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--lh_weight", type=float, default=0.0,
                    help="extra sample weight on long holds (cost-sensitive; 0=off)")
    args = ap.parse_args()

    X, y, groups, keys, dur = build_matrix(args.data_dir, want_dur=True)

    # out-of-fold predictions: each pause is scored by a model that never saw its turn
    oof = np.zeros(len(y), dtype=float)
    gkf = GroupKFold(n_splits=args.folds)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        clf = make_model(args.model)
        w = long_hold_weights(y[tr], dur[tr], extra=args.lh_weight) if args.lh_weight else None
        fit_weighted(clf, X[tr], y[tr], w)
        oof[te] = clf.predict_proba(X[te])[:, 1]
        accs.append((clf.predict(X[te]) == y[te]).mean())
    print(f"[{args.model}] {args.folds}-fold OOF acc: {np.mean(accs):.3f} "
          f"(chance ~ {max(np.mean(y), 1 - np.mean(y)):.3f})")

    write_preds(args.out, keys, oof)
    print(f"wrote {len(keys)} out-of-fold predictions -> {args.out}")


if __name__ == "__main__":
    main()
