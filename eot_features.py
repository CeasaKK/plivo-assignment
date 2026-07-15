"""Causal prosodic features for end-of-turn detection.

ONE source of truth for feature extraction, imported by train.py, fit_model.py
and predict.py so that fit-time and inference-time features are identical.

CAUSALITY: every feature is computed from audio[0 : pause_start] only. `pause_end`
is never read here (it is future information). The only per-pause metadata used is
`pause_index` (how many pauses have already occurred) and `pause_start` (elapsed
time) -- both are known to a live agent at decision time.
"""
import numpy as np
import librosa

from features import speech_before, frame_energy_db, f0_contour

HOP_S = 0.010  # f0_contour / frame_energy_db hop (10 ms)

FEATURE_NAMES = [
    # --- energy / trailing-silence structure ---
    "e_last150",         # mean energy dB, last 150 ms
    "e_decay_slope",     # dB/s slope over last 400 ms (into the pause)
    "e_drop_peak",       # window-max dB minus last-frame dB
    "e_last_minus_mean",  # last-frame dB minus window-mean dB
    "e_std_last300",     # energy dB std over last 300 ms (steady fade vs wobble)
    "trail_sil_s",       # trailing seconds already below (max-25 dB) before pause
    "trail_unvoiced_s",  # trailing seconds unvoiced after the last voiced run
    "voiced_ratio",      # fraction of frames voiced in window
    "voiced_ratio_last",  # fraction voiced in last 300 ms
    # --- pitch / F0 shape over the last voiced region ---
    "f0_slope_st",       # semitone/s slope over last voiced region (falling<0)
    "f0_final_delta_st",  # last-3 voiced vs prior-3 voiced, semitones
    "f0_last_minus_med_st",  # last voiced vs window median voiced, semitones
    "f0_range_st",       # max-min voiced F0 in window, semitones
    "f0_med_hz",         # median voiced F0 (speaker register)
    # --- rhythm / final-syllable lengthening / speaking rate ---
    "last_run_s",        # duration of final contiguous voiced run
    "last_run_ratio",    # final voiced-run length / mean voiced-run length
    "syllable_rate",     # energy-peak (syllable) rate per second in window
    "sylrate_ratio",     # syllable rate in last 750 ms / first 750 ms (slowing<1)
    "n_voiced_runs",     # number of voiced runs in window
    # --- spectral (new information channel; turn-final creak lowers centroid) ---
    "spec_cent_last",    # spectral centroid (Hz) over last 200 ms of speech
    "spec_cent_slope",   # spectral-centroid slope over last 400 ms (Hz/s)
    # --- pitch declination over a longer (3.5 s) context ---
    "f0_final_vs_floor_st",  # final pitch above the speaker's F0 floor (low=at end)
    "f0_ctx_decl_st",    # trailing-median vs context-median pitch (declination)
    # --- position / context ---
    "pause_index",       # holds seen so far this turn
    "pause_start_s",     # elapsed time into the turn
    "speech_ctx_s",      # amount of speech context available (<=1.5 s)
]

N_FEATURES = len(FEATURE_NAMES)


def _hz_to_st(f, ref):
    """semitones of f relative to ref (both Hz, >0)."""
    return 12.0 * np.log2(f / ref)


def _voiced_runs(voiced_mask):
    """List of (start, end_exclusive) index ranges where voiced_mask is True."""
    runs = []
    i, n = 0, len(voiced_mask)
    while i < n:
        if voiced_mask[i]:
            j = i
            while j < n and voiced_mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def pyin_f0(seg, sr, hop_s=HOP_S):
    """Robust F0 contour via probabilistic YIN (0.0 where unvoiced).

    pYIN tracks pitch far more reliably than raw autocorrelation (fewer octave
    errors, principled voicing). Computed on audio[0:pause_start] only.
    """
    hop = int(round(hop_s * sr))
    try:
        f0, vflag, _ = librosa.pyin(
            seg, sr=sr, fmin=65.0, fmax=400.0, frame_length=1024,
            hop_length=hop, fill_na=0.0)
    except Exception:
        return _clean_f0(f0_contour(seg, sr))
    f0 = np.nan_to_num(f0, nan=0.0)
    f0[~np.asarray(vflag, dtype=bool)] = 0.0
    return f0.astype(np.float32)


def _clean_f0(f0):
    """Median-smooth a raw F0 contour and repair octave jumps, in-place-safe.

    autocorr_f0 is noisy (octave halving/doubling, isolated spurious lags). We
    (1) fix voiced frames that are ~2x/0.5x their voiced neighbours' median,
    (2) apply a length-3 median filter over voiced frames. Unvoiced stays 0.
    """
    f = f0.astype(np.float64).copy()
    v = f > 0
    idx = np.where(v)[0]
    if len(idx) < 3:
        return f
    med = np.median(f[v])
    for i in idx:  # octave repair toward the running register
        if f[i] > 1.6 * med:
            f[i] /= 2.0
        elif f[i] < 0.6 * med:
            f[i] *= 2.0
    # median-3 over the voiced samples only (keeps unvoiced gaps as 0)
    vals = f[idx]
    sm = vals.copy()
    for k in range(1, len(vals) - 1):
        sm[k] = np.median(vals[k - 1:k + 2])
    f[idx] = sm
    return f


def _robust_slope(y):
    """Least-squares slope of y vs frame index, in units of y per frame."""
    n = len(y)
    if n < 2:
        return 0.0
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    denom = np.dot(t, t)
    if denom <= 0:
        return 0.0
    return float(np.dot(t, y - y.mean()) / denom)


def extract_features(x, sr, pause_start, pause_index=0.0):
    """Return an N_FEATURES vector from audio strictly before `pause_start`."""
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    speech_ctx = len(seg) / sr
    if len(seg) < sr // 10:  # <100 ms of context: nothing reliable to say
        f = np.zeros(N_FEATURES, dtype=np.float32)
        f[FEATURE_NAMES.index("pause_index")] = pause_index
        f[FEATURE_NAMES.index("pause_start_s")] = float(pause_start)
        f[FEATURE_NAMES.index("speech_ctx_s")] = speech_ctx
        return f

    e = frame_energy_db(seg, sr)          # ~10 ms hop
    f0 = pyin_f0(seg, sr)                 # ~10 ms hop, 0 = unvoiced, robust pitch
    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]

    # ---- energy structure ----
    n_last150 = max(1, int(0.150 / HOP_S))
    n_last300 = max(1, int(0.300 / HOP_S))
    n_last400 = max(2, int(0.400 / HOP_S))
    e_last150 = float(e[-n_last150:].mean())
    e_decay_slope = _robust_slope(e[-n_last400:]) / HOP_S  # dB per second
    e_max = float(e.max())
    e_drop_peak = e_max - float(e[-1])
    e_last_minus_mean = float(e[-1]) - float(e.mean())
    e_std_last300 = float(np.std(e[-n_last300:]))
    # trailing silence: frames from the end that are below (max-25 dB)
    sil_thresh = e_max - 25.0
    trail = 0
    for k in range(len(e) - 1, -1, -1):
        if e[k] < sil_thresh:
            trail += 1
        else:
            break
    trail_sil_s = trail * HOP_S
    voiced_ratio = float(voiced_mask.mean())
    voiced_ratio_last = float(voiced_mask[-n_last300:].mean())

    # ---- pitch shape over the last voiced region ----
    f0_med = float(np.median(voiced)) if len(voiced) else 0.0
    vidx = np.where(voiced_mask)[0]
    if len(voiced) >= 2 and f0_med > 0:
        st = _hz_to_st(f0[vidx], f0_med)          # semitones re speaker register
        n = len(voiced_mask)
        # slope over the last 600 ms of voiced frames (trailing intonation)
        w600 = vidx >= (n - int(0.600 / HOP_S))
        f0_slope_st = _robust_slope(st[w600]) / HOP_S if w600.sum() >= 2 else 0.0
        # trailing-vs-preceding pitch delta: last 300 ms voiced vs the 300-600 ms before
        recent = st[vidx >= (n - int(0.300 / HOP_S))]
        prev = st[(vidx >= (n - int(0.600 / HOP_S))) & (vidx < (n - int(0.300 / HOP_S)))]
        f0_final_delta_st = float(recent.mean() - prev.mean()) if len(recent) and len(prev) else 0.0
        f0_last_minus_med_st = float(st[-3:].mean())  # last voiced pitch re register
        f0_range_st = float(st.max() - st.min())
    else:
        f0_slope_st = f0_final_delta_st = f0_last_minus_med_st = f0_range_st = 0.0

    # ---- rhythm / final-syllable lengthening / speaking rate ----
    runs = _voiced_runs(voiced_mask)
    if runs:
        run_lens = np.array([(b - a) for a, b in runs], dtype=np.float64)
        last_run_s = float(run_lens[-1] * HOP_S)
        mean_run = float(run_lens.mean())
        last_run_ratio = float(run_lens[-1] / mean_run) if mean_run > 0 else 1.0
        n_voiced_runs = float(len(runs))
        # trailing unvoiced: frames after the last voiced run to the window end
        trail_unvoiced_s = float((len(voiced_mask) - runs[-1][1]) * HOP_S)
    else:
        last_run_s = last_run_ratio = 0.0
        n_voiced_runs = 0.0
        trail_unvoiced_s = float(len(voiced_mask) * HOP_S)

    # syllable rate: energy peaks above (max-15 dB), local maxima, per second
    win_s = len(seg) / sr
    thr = e_max - 15.0

    def _peak_rate(ev):
        p = sum(1 for k in range(1, len(ev) - 1)
                if ev[k] > thr and ev[k] >= ev[k - 1] and ev[k] > ev[k + 1])
        return p / (len(ev) * HOP_S) if len(ev) > 2 else 0.0

    syllable_rate = _peak_rate(e)
    # slowing-down: syllable rate in last 750 ms vs first 750 ms of the window
    half = int(0.750 / HOP_S)
    if len(e) >= 2 * 3:
        r_late = _peak_rate(e[-half:])
        r_early = _peak_rate(e[:half])
        sylrate_ratio = float(r_late / r_early) if r_early > 0 else 1.0
    else:
        sylrate_ratio = 1.0

    # ---- spectral tilt / centroid over the trailing speech ----
    hop = int(round(HOP_S * sr))
    cent = librosa.feature.spectral_centroid(y=seg, sr=sr, n_fft=512, hop_length=hop)[0]
    # keep only frames above the silence floor (centroid on silence is noise)
    ce = frame_energy_db(seg, sr)
    m = min(len(cent), len(ce))
    cent, ce = cent[:m], ce[:m]
    speechy = ce > (e_max - 30.0)
    n_c200 = max(1, int(0.200 / HOP_S))
    n_c400 = max(2, int(0.400 / HOP_S))
    tail_c = cent[-n_c200:][speechy[-n_c200:]]
    spec_cent_last = float(tail_c.mean()) if len(tail_c) else float(cent[-n_c200:].mean())
    c4 = cent[-n_c400:]
    s4 = speechy[-n_c400:]
    spec_cent_slope = _robust_slope(c4[s4]) / HOP_S if s4.sum() >= 2 else 0.0

    # ---- pitch declination over a longer context: is the final pitch at the
    #      bottom of the speaker's recent range? (true ends bottom out) ----
    f0_final_vs_floor_st = f0_ctx_decl_st = 0.0
    seg_long = speech_before(x, sr, pause_start, window_s=3.5)
    if len(seg_long) > len(seg) + sr // 5 and len(voiced):
        f0l = pyin_f0(seg_long, sr)
        vl = f0l[f0l > 0]
        if len(vl) >= 5:
            floor = float(np.percentile(vl, 10))
            last_pitch = float(np.median(f0[voiced_mask][-3:]))
            if floor > 0 and last_pitch > 0:
                f0_final_vs_floor_st = float(_hz_to_st(last_pitch, floor))
            ctx_med = float(np.median(vl))
            tail_ct = int(0.5 / HOP_S)
            tail = f0l[-tail_ct:][f0l[-tail_ct:] > 0]
            if ctx_med > 0 and len(tail):
                f0_ctx_decl_st = float(_hz_to_st(np.median(tail), ctx_med))

    return np.array([
        e_last150, e_decay_slope, e_drop_peak, e_last_minus_mean, e_std_last300,
        trail_sil_s, trail_unvoiced_s, voiced_ratio, voiced_ratio_last,
        f0_slope_st, f0_final_delta_st, f0_last_minus_med_st, f0_range_st, f0_med,
        last_run_s, last_run_ratio, syllable_rate, sylrate_ratio, n_voiced_runs,
        spec_cent_last, spec_cent_slope, f0_final_vs_floor_st, f0_ctx_decl_st,
        float(pause_index), float(pause_start), speech_ctx,
    ], dtype=np.float32)
