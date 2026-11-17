"""Cross-subject calibration degradation and zero-shot conformal coverage.
Produces section 5.2, tables 3 and 4.

The threshold here comes from the training subjects' calibration fold, so the
deployment subject supplies no labels at all. That is the difference from the
static arm of table 5, which gets a 50-window labelled prefix.

    python scripts/run_cross_subject.py seedvig
    python scripts/run_cross_subject.py sadt mlp,xgb
"""
import sys
import os
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_audit import (LOADERS, MODELS, SEEDS, ALPHA, subject_key,
                         zero_shot_loso)

# Table 3 (delta balanced ECE) and table 4 (zero-shot coverage) as published,
# so a run says straight away whether it reproduces them.
PUBLISHED_DELTA = {
    'seedvig': dict(mlp=.207, cnn=.154, eegnet=.178, conformer=.196, xgb=.218),
    'drozy': dict(mlp=.192, cnn=.209, eegnet=.190, conformer=.176, xgb=.156),
}
PUBLISHED_COVERAGE = {
    'seedvig': dict(mlp=.718, cnn=.816, eegnet=.801, conformer=.746, xgb=.535),
    'drozy': dict(mlp=.726, cnn=.782, eegnet=.843, conformer=.761, xgb=.686),
}


def main(corpus='seedvig', backbones=None):
    names = backbones.split(',') if backbones else MODELS
    subjects, D, C, B = LOADERS[corpus]()
    ids = sorted(subjects, key=subject_key)
    print(f"corpus={corpus}  subjects={len(ids)}  seeds={len(SEEDS)}  "
          f"target={1 - ALPHA:.2f}\n")

    print(f"{'backbone':>11}{'balECE in':>11}{'balECE out':>12}{'delta':>9}"
          f"{'coverage':>11}{'set size':>10}{'frac<0.85':>11}{'sd':>8}{'min':>8}")
    print('-' * 91)
    rows = {}
    for name in names:
        inside, outside, cov, size = [], [], [], []
        per_subj = collections.defaultdict(list)
        for held, cc_in, cc_out, c, sz in zero_shot_loso(
                subjects, name, D, C, B):
            if not np.isnan(cc_in):
                inside.append(cc_in)
            if not np.isnan(cc_out):
                outside.append(cc_out)
            cov.append(c)
            size.append(sz)
            per_subj[held].append(c)
        by_subject = np.array([np.mean(per_subj[s]) for s in ids if per_subj[s]])
        rows[name] = dict(
            cc_in=float(np.mean(inside)), cc_out=float(np.mean(outside)),
            delta=float(np.mean(outside) - np.mean(inside)),
            cov=float(np.mean(cov)), size=float(np.mean(size)),
            frac=float((by_subject < 0.85).mean()),
            sd=float(by_subject.std()), mn=float(by_subject.min()))
        r = rows[name]
        print(f"{name:>11}{r['cc_in']:>11.4f}{r['cc_out']:>12.4f}{r['delta']:>+9.4f}"
              f"{r['cov']:>11.3f}{r['size']:>10.2f}{r['frac']:>11.2f}"
              f"{r['sd']:>8.3f}{r['mn']:>8.3f}")

    if corpus in PUBLISHED_DELTA:
        print("\n--- against the published tables 3 and 4 ---")
        print(f"{'backbone':>11}{'delta':>10}{'published':>11}{'diff':>8}"
              f"{'coverage':>11}{'published':>11}{'diff':>8}")
        worst = 0.0
        for name in names:
            if name not in PUBLISHED_DELTA[corpus]:
                continue
            r = rows[name]
            pd_ = PUBLISHED_DELTA[corpus][name]
            pc_ = PUBLISHED_COVERAGE[corpus][name]
            worst = max(worst, abs(r['delta'] - pd_), abs(r['cov'] - pc_))
            print(f"{name:>11}{r['delta']:>+10.3f}{pd_:>+11.3f}"
                  f"{r['delta'] - pd_:>+8.3f}{r['cov']:>11.3f}{pc_:>11.3f}"
                  f"{r['cov'] - pc_:>+8.3f}")
        print(f"\nlargest disagreement {worst:+.3f}. Seed and split randomness "
              f"accounts for a few\nhundredths; past about 0.03 the protocol "
              f"itself differs from the published run.")
    else:
        print(f"\nNo published column for {corpus} to check against: tables 3 "
              f"and 4 report its\nvalues from this script rather than the other "
              f"way round.")

    np.save(f'cross_subject_{corpus}.npy',
            np.array([[rows[n]['delta'], rows[n]['cov'], rows[n]['size']]
                      for n in names]))
    print(f"saved -> cross_subject_{corpus}.npy")


if __name__ == '__main__':
    main(*sys.argv[1:3])
