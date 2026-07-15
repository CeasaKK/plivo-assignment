# EOT Detection — Run Log

**Metric:** mean response delay (ms) at the best operating point keeping interrupted
turns ≤ 5% (from `score.py`). **Lower is better.** AUC is a secondary diagnostic.

**Scoring honesty.** The starter `train.py` wrote *in-sample* predictions (refit on all
data, predict on the same data) — optimistic. From E1 on, evaluation uses **out-of-fold**
predictions via `GroupKFold` on `turn_id` (a turn never spans folds), so scored numbers
reflect generalization to unseen turns. Headline numbers below are held-out OOF.

**Data.** EN and HI each: 100 turns, 248 pauses (100 eot / 148 hold), one eot (final pause)
per turn. 16 kHz mono.

**Frontier facts that shape the whole problem:**
- Only *long* holds can cause a false cutoff: a hold with `dur ≤ delay` never fires before
  the user resumes. So the classifier only needs to rank eots above **long** holds; short
  holds firing is harmless. All feature work targets the eot-vs-long-hold boundary.
- HI holds are short (only ~5% of turns have a hold > 0.9 s), so an 850 ms silence timer is
  already near-optimal for HI. EN holds are longer (25% of turns have a hold > 0.7 s), so the
  naive timer must wait 1600 ms — leaving EN much more room to improve.
- **Oracle** (perfect p_eot): 100 ms for both languages. All remaining gap is model quality.

## Experiment table

| run | data | features / change (repr. model) | AUC | mean_delay_ms | cutoff_% |
|-----|------|--------------------------------|-----|---------------|----------|
| baseline | EN | p_eot=1.0 (silence timer) | 0.506 | 1600 | 0.0 |
| baseline | HI | p_eot=1.0 (silence timer) | 0.516 | 850 | 5.0 |
| starter (in-sample) | EN | energy mean, final pitch, seg len (3 feats) | 0.599 | 1190 | 5.0 |
| starter (in-sample) | HI | energy mean, final pitch, seg len (3 feats) | 0.634 | 850 | 5.0 |
| E1 | EN | 19 causal feats: F0 slope/range, energy decay, trailing silence, rhythm, pause pos (logreg, OOF) | 0.624 | 1249 | 5.0 |
| E1 | HI | same 19 feats (logreg, OOF) | 0.650 | 850 | 5.0 |
| E2 | EN | F0 de-noise (median-3 + octave repair) + e_std, trail_unvoiced, sylrate_ratio (logreg) | 0.632 | 1160 | 5.0 |
| E2 | HI | same (hgb) | 0.693 | 887 | 5.0 |
| E3 | EN | + spectral centroid last/slope (hgb) | 0.616 | 1165 | 5.0 |
| E3 | HI | + spectral centroid (hgb) | 0.703 | 872 | 5.0 |
| E4 | EN | **F0 via librosa pYIN** (robust pitch) (logreg) | 0.663 | 1220 | 5.0 |
| E4 | HI | F0 via pYIN (logreg) | 0.706 | 850 | 5.0 |
| E5 | EN | cost-sensitive long-hold weighting (train-time only) (hgb) | 0.649 | 1108 | 5.0 |
| E5 | HI | cost-sensitive weighting (hgb) | 0.697 | 887 | 5.0 |
| E6 | EN | **ensemble hgb+logreg** (avg proba) | 0.672 | 1112 | 5.0 |
| E6 | HI | ensemble hgb+logreg | 0.725 | 850 | 5.0 |
| E7 | EN | + pitch-declination (final vs F0 floor over 3.5 s ctx) — hurt EN (pitch-agnostic lang) | 0.659 | 1176 | 5.0 |
| E7 | HI | + pitch-declination — helped HI ranking | 0.723 | 850 | 5.0 |
| **E8 (final)** | **EN** | **JOINT model (EN+HI), ensemble, 26 feats** | **0.693** | **1116** | **5.0** |
| **E8 (final)** | **HI** | **JOINT model (EN+HI), ensemble, 26 feats** | **0.761** | **850** | **5.0** |

## Key findings from numerical error analysis (`analyze.py`)

The assignment's "listen to your errors" done numerically: after each run I dumped the
worst long-hold false-alarms (high p) and the worst eot misses (low p) with their feature
values (`analyze.py --errors <oof.csv> --delay D`).

1. **Energy decay + trailing silence are the most reliable, language-consistent eot cues.**
   `e_decay_slope` and `trail_sil_s` had the strongest long-hold AUC for EN (0.61 each).
2. **The autocorr pitch tracker was too noisy** — F0 slope was dead (EN AUC 0.512). Many
   "rising-pitch eots" in HI were tracker octave errors. Switching to **pYIN** revived the
   pitch channel: HI `f0_last_minus_med_st` AUC 0.46→0.33 (eot pitch clearly lower), and the
   declination feature `f0_final_vs_floor_st` reached AUC 0.33 for HI.
3. **The languages differ, as predicted.** EN is **energy/silence-driven** (pitch weak); HI is
   **pitch/declination-driven** (final pitch drops to the speaker's floor at true ends). This is
   why the language-specific pitch features help HI but add noise to EN.
4. **Overfitting was the real ceiling, not features.** In-sample AUC ≈ 0.99 (delay ~145 ms)
   vs OOF AUC ~0.66–0.76 — a huge gap on ~200 training points × 26 features. Regularization
   and feature pruning gave little; the decisive fix was **more data via joint training**.
5. **HI is floored at its 850 ms baseline.** Its fine frontier has a cliff: d=0.85 (fire on
   everything, 850 ms) vs d=0.80 (must raise the threshold → eot recall craters → 968–1024 ms).
   Beating 850 needs near-perfect eot/long-hold separation that these prosodic features don't
   reach. AUC improved a lot (0.52→0.76) but the metric can't cash it in — an honest negative
   result, not a bug.

## Feature directions covered (Section 6 checklist)

- **F0 slope over last voiced region** — `f0_slope_st` (pYIN, 600 ms trailing span). Helps HI.
- **Final-syllable lengthening** — `last_run_s`, `last_run_ratio` (final voiced run vs mean).
- **Energy decay rate into the pause** — `e_decay_slope`, `e_drop_peak`, `e_std_last300`. Top EN cue.
- **Speaking-rate / pause position** — `syllable_rate`, `sylrate_ratio`, `pause_index`, `pause_start_s`.
- **Voicing / silence-before-silence** — `voiced_ratio`, `voiced_ratio_last`, `trail_sil_s`, `trail_unvoiced_s`.
- Extra channels: spectral centroid (creak), pitch **declination** vs speaker floor over a 3.5 s context.

## Model decision: JOINT vs per-language

Held-out (OOF) head-to-head, identical 26 features + hgb+logreg ensemble:

| | per-language | joint (EN+HI) |
|---|---|---|
| EN | AUC 0.659, 1176 ms | **AUC 0.693, 1116 ms** |
| HI | AUC 0.723, 850 ms | **AUC 0.761, 850 ms** |

**Shipped: one JOINT model.** It beats per-language on both AUC and delay. The reason is the
overfitting finding above — doubling the training data (200→400 turns) regularizes the
26-feature model far better than any hyperparameter could, and the prosodic features are
normalized (semitones re register, relative dB) so they transfer across the two languages.
Joint training is also direct evidence the features *generalize cross-language*.

Cost-sensitive long-hold weighting and pitch-declination features use pause duration only at
**train time** (sample weights / floor context) — never as inference features. `predict.py`
and `extract_features()` never read `pause_end` (verified by grep).

## Final deliverable & numbers

- `fit_model.py` → trains the joint ensemble on EN+HI, `joblib.dump` → `eot_model.joblib`.
- `predict.py` → loads that artifact, inference only (no refit), CLI matches `baseline.py`.
- `evaluate.py` → reproduces the held-out OOF numbers of record below.

| track | baseline | **final (held-out OOF)** | improvement |
|-------|----------|--------------------------|-------------|
| English | 1600 ms | **1116 ms** (AUC 0.693) | **−30%** |
| Hindi | 850 ms | **850 ms** (AUC 0.761) | 0% (baseline is already near-optimal for HI) |

*Note:* running `predict.py` on the provided data is **in-sample** (the shipped model was
trained on it) and reports an optimistic ~469 ms (EN) / ~331 ms (HI). The **held-out** numbers
above (from `evaluate.py`) are the honest expected performance on unseen turns.
