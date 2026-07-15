# End-of-Turn (EOT) Detection for a Voice Agent

For every silence/pause in a conversation recording, predict `p_eot` — the probability
that the pause is a genuine **end of turn** (user is done) vs a **hold** (user paused
mid-thought and will resume). This is the decision layer a real-time voice agent uses to
choose between responding now and waiting.

**Hard constraint — causality:** for a pause at `pause_start`, only `audio[0 : pause_start]`
may be used. `pause_end` (and any duration derived from it) is future information and is
never read by a feature or by `predict.py` (verified by grep).

## Results (held-out, GroupKFold OOF over turns)

Scored by `score.py`: **mean response delay (ms)** at the best operating point that keeps
interrupted turns ≤ 5%. Lower is better.

| track | baseline (silence timer) | final | improvement |
|-------|--------------------------|-------|-------------|
| English | 1600 ms | **1116 ms** (AUC 0.69) | **−30%** |
| Hindi | 850 ms | **850 ms** (AUC 0.76) | 0% — baseline already near-optimal |

Oracle (perfect classifier) is 100 ms for both, so the whole gap is model quality. Full
experiment history, error analysis, and the joint-vs-per-language decision are in
[RUNLOG.md](RUNLOG.md).

## Approach in one paragraph

26 causal prosodic features from the pre-pause audio: energy-decay/trailing-silence
structure, pitch shape via **pYIN** (slope, range, final pitch re speaker register, and
declination toward the speaker's F0 floor over a longer context), final-syllable lengthening
and speaking-rate rhythm, spectral centroid, and pause position. Classifier is an
average-probability **ensemble of gradient boosting + logistic regression**, trained
**jointly on English + Hindi** — the decisive win, because a 26-feature model overfits badly
on ~200 turns (in-sample AUC ~0.99 vs OOF ~0.7) and doubling the data regularizes it while
also proving the features transfer across languages. English is energy/silence-driven; Hindi
is pitch/declination-driven; Hindi is floored at its baseline because its holds are short
enough that a silence timer is already near-optimal.

## Files

| file | role |
|------|------|
| `eot_features.py` | **the feature work** — `extract_features()`, one source of truth for train/fit/predict |
| `features.py` | provided audio utilities (loading, framing, energy, autocorr pitch) |
| `eot_common.py` | data-matrix building, model factory, weight helpers |
| `train.py` | dev loop: honest out-of-fold predictions per language |
| `analyze.py` | numerical error analysis (frontier, per-feature AUC, misclassification dumps) |
| `eval_sweep.py` | fast cached config sweeps |
| `evaluate.py` | held-out OOF evaluation of the shipped joint ensemble (numbers of record) |
| `fit_model.py` | fit the joint ensemble on all data → `eot_model.joblib` |
| `predict.py` | **final deliverable** — loads the saved model, inference only |
| `baseline.py`, `score.py` | provided silence-only baseline and official scorer |
| `eot_model.joblib` | saved joint ensemble artifact |
| `oof_english.csv`, `oof_hindi.csv` | held-out predictions scored in the results table |

## Running it

Place the handout `eot_data/` (with `english/` and `hindi/` subfolders) next to this repo,
then:

```bash
# baseline and scorer
python baseline.py --data_dir eot_data/english --out base.csv
python score.py    --data_dir eot_data/english --pred base.csv

# honest per-language dev loop (out-of-fold)
python train.py    --data_dir eot_data/english --out mine.csv
python score.py    --data_dir eot_data/english --pred mine.csv

# fit the final joint model, then inference-only prediction
python fit_model.py --data_dirs eot_data/english eot_data/hindi --out eot_model.joblib
python predict.py   --data_dir eot_data/english --out predictions.csv
python score.py     --data_dir eot_data/english --pred predictions.csv

# reproduce the held-out numbers of record
python evaluate.py
```

Note: running `predict.py` on the provided data is **in-sample** (the shipped model was
trained on it) and reports optimistic ~469 ms / ~331 ms. The held-out numbers in the table
above (from `evaluate.py`) are the honest expected performance on unseen turns.

Requires: `numpy scipy scikit-learn pandas librosa soundfile joblib`.
