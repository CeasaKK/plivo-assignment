"""Shared data loading / feature-matrix building for the EOT pipeline."""
import csv
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import load_wav
from eot_features import extract_features, FEATURE_NAMES


def build_matrix(data_dir, want_dur=False):
    """Return X, y, groups, keys for every pause in data_dir/labels.csv.

    If want_dur, also return pause durations (train-time only, for cost-sensitive
    weighting -- NEVER used as a feature; predict.py does not read it).
    """
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    cache = {}
    X, y, groups, keys, dur = [], [], [], [], []
    for r in rows:
        path = os.path.join(data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"]),
                                  pause_index=float(r["pause_index"])))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], int(r["pause_index"])))
        dur.append(float(r["pause_end"]) - float(r["pause_start"]))
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    if want_dur:
        return X, y, groups, keys, np.array(dur)
    return X, y, groups, keys


def long_hold_weights(y, dur, extra=3.0, thresh=0.5):
    """Sample weights: up-weight long holds (dur>thresh) -- the only pauses that
    can cause a false cutoff -- so the model works hardest to rank them low.
    Uses duration at TRAIN time only; not a feature."""
    w = np.ones(len(y), dtype=float)
    w[(y == 0) & (dur > thresh)] = 1.0 + extra
    return w


def make_model(kind="hgb"):
    """Classifier factory. Same object used at fit and inference time."""
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        )
    # gradient boosting: handles nonlinear feature interactions, no scaling needed
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=300, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=15,
        class_weight="balanced", random_state=0,
    )


def fit_weighted(clf, X, y, w):
    """Fit, routing sample_weight to the final step if clf is a Pipeline."""
    if w is None:
        clf.fit(X, y)
    elif hasattr(clf, "steps"):  # Pipeline: name the final estimator step
        clf.fit(X, y, **{clf.steps[-1][0] + "__sample_weight": w})
    else:
        clf.fit(X, y, sample_weight=w)
    return clf


def write_preds(out_path, keys, p):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
