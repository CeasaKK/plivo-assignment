"""Fast config sweep: cache features once, evaluate OOF mean-delay for many
(model, feature-subset, weight) configs. Optimizes the ACTUAL scorer metric."""
import os
import numpy as np
from sklearn.model_selection import GroupKFold

from eot_common import build_matrix, make_model, long_hold_weights, fit_weighted
from eot_features import FEATURE_NAMES
from score import score as score_fn
import csv

CACHE = "feat_cache"
os.makedirs(CACHE, exist_ok=True)


def get_data(lang):
    p = os.path.join(CACHE, f"{lang}.npz")
    d = f"../eot_data/{lang}"
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["X"], z["y"], z["groups"], z["keys"], z["dur"]
    X, y, groups, keys, dur = build_matrix(d, want_dur=True)
    np.savez(p, X=X, y=y, groups=groups, keys=np.array(keys, dtype=object), dur=dur)
    return X, y, groups, keys, dur


def oof_preds(X, y, groups, dur, model, lh_weight, cols, folds=5):
    oof = np.zeros(len(y))
    Xs = X[:, cols] if cols is not None else X
    for tr, te in GroupKFold(n_splits=folds).split(Xs, y, groups):
        clf = make_model(model)
        w = long_hold_weights(y[tr], dur[tr], extra=lh_weight) if lh_weight else None
        fit_weighted(clf, Xs[tr], y[tr], w)
        oof[te] = clf.predict_proba(Xs[te])[:, 1]
    return oof


def mean_delay(lang, keys, oof):
    """Write a temp pred + run the official scorer; return (delay_ms, auc)."""
    tmp = os.path.join(CACHE, f"_p_{lang}.csv")
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["turn_id", "pause_index", "p_eot"])
        for k, p in zip(keys, oof):
            w.writerow([k[0], int(k[1]), f"{p:.4f}"])
    r = score_fn(f"../eot_data/{lang}/labels.csv", tmp)
    return r["latency"] * 1000, r["auc"], r["delay"] * 1000


def run(cols=None, models=("hgb", "logreg"), weights=(0,), tag=""):
    for lang in ("english", "hindi"):
        X, y, groups, keys, dur = get_data(lang)
        for m in models:
            for w in weights:
                oof = oof_preds(X, y, groups, dur, m, w, cols)
                dly, auc, op = mean_delay(lang, keys, oof)
                print(f"{tag:14s} {lang:8s} {m:7s} w={w}: AUC={auc:.3f} delay={dly:.0f}ms (op={op:.0f})")


if __name__ == "__main__":
    run(tag="ALL")
