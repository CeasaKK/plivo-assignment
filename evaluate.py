"""Honest held-out evaluation of the SHIPPED joint ensemble.

The final model (fit_model.py) is trained on English+Hindi together, so running
predict.py on that same data is in-sample and optimistic. This script reproduces
the true generalization numbers: GroupKFold over turns across BOTH languages
(a turn never spans folds), using the identical joint hgb+logreg ensemble, then
writes per-language out-of-fold predictions and scores them with the official
scorer. These are the numbers reported in RUNLOG.md.

    python evaluate.py            # writes oof_english.csv, oof_hindi.csv + scores
"""
import csv
import numpy as np
from sklearn.model_selection import GroupKFold

from eot_common import build_matrix, make_model, write_preds
from score import score as score_fn

DATA = {"english": "../eot_data/english", "hindi": "../eot_data/hindi"}


def main():
    Xs, ys, gs, ks, langs = [], [], [], [], []
    for lang, d in DATA.items():
        X, y, groups, keys = build_matrix(d)
        Xs.append(X); ys.append(y)
        gs += [f"{lang}:{g}" for g in groups]     # keep turn groups distinct per lang
        ks += keys
        langs += [lang] * len(y)
    X = np.vstack(Xs); y = np.concatenate(ys)
    g = np.array(gs); langs = np.array(langs)

    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        probs = []
        for kind in ("hgb", "logreg"):
            clf = make_model(kind)
            clf.fit(X[tr], y[tr])
            probs.append(clf.predict_proba(X[te])[:, 1])
        oof[te] = np.mean(probs, axis=0)

    print("Held-out (GroupKFold OOF) scores of the shipped joint ensemble:")
    for lang, d in DATA.items():
        m = langs == lang
        keys_l = [ks[i] for i in range(len(ks)) if m[i]]
        out = f"oof_{lang}.csv"
        write_preds(out, keys_l, oof[m])
        r = score_fn(f"{d}/labels.csv", out)
        print(f"  {lang:8s}: AUC={r['auc']:.3f}  mean_delay={r['latency']*1000:.0f} ms  "
              f"cutoff={r['cutoff']*100:.1f}%  (op: thr={r['threshold']}, delay={r['delay']*1000:.0f} ms)  -> {out}")


if __name__ == "__main__":
    main()
