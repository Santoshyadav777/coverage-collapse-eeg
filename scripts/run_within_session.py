"""Within-session calibration audit. Produces section 5.1 and figures 3 and 4.

Under the within-session protocol of section 4.2 the model is fitted on the
first 40% of every recording, its softmax temperature is fitted on the 40% to
55% block, and all measurements are made from 55% to the end. Evaluation
windows are binned by position within their own recording, and the same
predictions are then aggregated two ways: pooled over all windows of a bin, and
per recording and averaged with equal weight. Nothing changes between the two
numbers except the order of averaging.

    python scripts/run_within_session.py seedvig
    python scripts/run_within_session.py sadt
"""
import sys
import os
import numpy as np
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_audit import (LOADERS, SESSION_BINS, MIN_PER_CELL, TEMPERATURE_FITS,
                         EVALUATE_FROM, LATE_START, load_seedvig,
                         within_session_predictions, session_bin_table, ece,
                         class_conditional_ece, sign_test)


TRAIN_END_WARN = 0.40


def load_streams(corpus):
    """The unit this analysis runs on, which is not the cross-subject unit.

    SEED-VIG is kept at the level of its 23 recordings, because a recording is
    the unit within which time-on-task drift accumulates, while the
    cross-subject analysis uses the 21 subjects. SADT is loaded at its usual
    subject level, aggregating a subject's sessions, because time-on-task
    accumulates across them in the released recordings.
    """
    if corpus == 'seedvig':
        return load_seedvig(by_recording=True)
    return LOADERS[corpus]()


def main(corpus='seedvig', backbone='mlp', nbins=None, seed=0, temp_fit='logt',
         evaluate_from=None):
    if corpus == 'drozy':
        raise SystemExit(
            'DROZY carries one KSS rating per session rather than a continuous\n'
            'annotation, so it has no time-resolved label and cannot support the\n'
            'within-session analysis. See section 3.1 of the paper. Use seedvig\n'
            'or sadt here, and DROZY with the cross-subject scripts.')
    if corpus not in SESSION_BINS:
        raise SystemExit(f'unknown corpus {corpus!r}; expected one of '
                         f'{sorted(SESSION_BINS)}')
    if temp_fit not in TEMPERATURE_FITS:
        raise SystemExit(f'unknown temperature fit {temp_fit!r}; expected one of '
                         f'{sorted(TEMPERATURE_FITS)}')
    nbins = int(nbins) if nbins else SESSION_BINS[corpus]
    seed = int(seed)
    start = (EVALUATE_FROM[corpus] if evaluate_from is None
             else (LATE_START if evaluate_from == 'late' else float(evaluate_from)))
    recordings, D, C, B = load_streams(corpus)

    probs, bins, temperature, late, dropped = within_session_predictions(
        recordings, nbins, backbone, D, C, B, seed,
        temperature_fit=TEMPERATURE_FITS[temp_fit], evaluate_from=start)
    label = 'deciles' if nbins == 10 else 'quintiles' if nbins == 5 else f'{nbins} bins'
    print(f"corpus={corpus}  backbone={backbone}  {label}  seed={seed}  "
          f"temperature fit={temp_fit}")
    print(f"evaluation block starts at {start:.0%} of each recording"
          + ("  (bins before 40% are in sample)" if start < TRAIN_END_WARN else ""))
    print(f"streams kept {len(late)}, dropped {len(dropped)}"
          + (f" -> {dropped}" if dropped else ""))
    print(f"temperature T = {temperature:.3f}\n")

    t = session_bin_table(probs, bins, late, nbins)
    pooled, per = t['pooled'], t['per_recording']

    print(f"{'bin':>4}{'n rec':>7}{'pooled ECE':>12}{'per-rec ECE':>13}"
          f"{'acc':>8}{'balECE':>9}{'Brier':>8}{'ECE|fatig':>11}{'frac fatig':>12}")
    for b in range(nbins):
        print(f"{b + 1:>4}{t['n_recordings'][b]:>7}{pooled['ece'][b]:>12.4f}"
              f"{per['ece'][b]:>13.4f}{pooled['accuracy'][b]:>8.4f}"
              f"{pooled['balanced_ece'][b]:>9.4f}{pooled['brier'][b]:>8.4f}"
              f"{pooled['ece_fatigued'][b]:>11.4f}"
              f"{pooled['fatigued_fraction'][b]:>12.3f}")

    print("\n--- Spearman against session position ---")
    pos = np.arange(1, nbins + 1)
    for name, v in (('pooled ECE', pooled['ece']),
                    ('per-recording ECE', per['ece']),
                    ('accuracy', pooled['accuracy']),
                    ('balanced ECE', pooled['balanced_ece']),
                    ('ECE|fatigued', pooled['ece_fatigued']),
                    ('fatigued fraction', pooled['fatigued_fraction'])):
        rho, pv = spearmanr(pos, v)
        print(f"  {name:<20} rho = {rho:+.3f}   p = {pv:.4f}")

    gap = per['ece'] - pooled['ece']
    wins = int((gap > 0).sum())
    print(f"\nper-recording exceeds pooled in {wins} of {nbins} bins"
          f"  (one-sided sign test p = {sign_test(wins, nbins):.3e})")
    print(f"widest gap {np.nanmax(gap):+.4f} at bin {int(np.nanargmax(gap)) + 1}")
    ratio = pooled['ece_fatigued'] / pooled['ece']
    print(f"widest ECE|fatigued / pooled {np.nanmax(ratio):.2f}x at bin "
          f"{int(np.nanargmax(ratio)) + 1}")

    print("\n--- paired per-recording, first against last bin ---")
    for name in ('ece', 'balanced_ece', 'brier', 'ece_fatigued'):
        vals = t['per_recording_values'][name]
        a, b = np.array(vals[0]), np.array(vals[nbins - 1])
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b = a[ok], b[ok]
        if len(a) < 3:
            print(f"  {name:<14} n = {len(a)}, too few for a paired test")
            continue
        pv = wilcoxon(b, a).pvalue
        print(f"  {name:<14} n = {len(a):<4} {a.mean():.4f} -> {b.mean():.4f}"
              f"   median {np.median(b - a):+.4f}   p = {pv:.3e}")

    print("\n--- per-recording class gap over the whole evaluation block ---")
    fat, agg, frac = [], [], []
    for r, (_, y) in late.items():
        p = probs[r]
        f = class_conditional_ece(p, y, 1)
        if (not np.isnan(f) and (y == 1).sum() >= MIN_PER_CELL
                and (y == 0).sum() >= MIN_PER_CELL):
            fat.append(f)
            agg.append(ece(p, y))
            frac.append(float((y == 1).mean()))
    fat, agg, frac = map(np.array, (fat, agg, frac))
    if len(fat) >= 3:
        r_ = fat / agg
        rho, p_rho = spearmanr(frac, r_)
        med = np.median(frac)
        print(f"  recordings with both classes >= {MIN_PER_CELL}: {len(fat)}")
        print(f"  mean ECE|fatigued {fat.mean():.4f} against aggregate "
              f"{agg.mean():.4f}   Wilcoxon p = {wilcoxon(fat, agg).pvalue:.4f}")
        print(f"  median ratio {np.median(r_):.3f}   range {r_.min():.3f} to "
              f"{r_.max():.3f}   ratio of means {fat.mean() / agg.mean():.2f}")
        print(f"  Spearman(fatigued fraction, ratio) rho = {rho:+.3f}  p = {p_rho:.4f}")
        print(f"  median ratio: rarer-fatigued half {np.median(r_[frac <= med]):.2f}"
              f"   commoner half {np.median(r_[frac > med]):.2f}")
    else:
        print(f"  only {len(fat)} recordings carry both classes at "
              f">= {MIN_PER_CELL} windows")

    cells = empty = saturated = 0
    for r, (_, y) in late.items():
        for b in range(nbins):
            m = bins[r] == b
            if not m.sum():
                continue
            cells += 1
            f = (y[m] == 1).mean()
            empty += int(f == 0)
            saturated += int(f == 1)
    print(f"\nrecording-bin cells {cells}   no fatigued window "
          f"{100 * empty / cells:.1f}%   all fatigued {100 * saturated / cells:.1f}%")

    np.save(f'within_session_{corpus}_{backbone}.npy',
            np.c_[pooled['ece'], per['ece'], pooled['accuracy'],
                  pooled['ece_fatigued']])
    print(f"saved -> within_session_{corpus}_{backbone}.npy")


if __name__ == '__main__':
    main(*sys.argv[1:7])
