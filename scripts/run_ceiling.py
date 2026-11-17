"""Per-subject attainability ceiling. Produces table S1 and figure 7.

For each held-out subject: the largest score in the fixed warm-up multiset, the
coverage that ceiling implies, the best any ceiling-bound arm attains, and DACR
for contrast. No ceiling-bound arm can lie above its own ceiling, and the
figure is the check of that.

The published table compares the ceiling against max(static, ACI), both at the
plain 1 - alpha level, which is what the cells behind it used. Pass arms=all to
include DtACI and Mondrian as well, at the corrected level where they use it.

    python scripts/run_ceiling.py seedvig mlp
    python scripts/run_ceiling.py sadt mlp all
"""
import sys
import os
import collections
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_audit import (LOADERS, ALPHA, loso_streams, subject_key,
                         attainability_ceiling, required_quantile, static_cov,
                         aci_cov, dtaci_cov, mondrian_cov, dacr_cov)


def main(corpus='seedvig', backbone='mlp', arms='published'):
    subjects, D, C, B = LOADERS[corpus]()
    print(f"corpus={corpus}  backbone={backbone}  subjects={len(subjects)}  "
          f"arms={arms}  target={1 - ALPHA:.2f}\n")

    rows = []
    for seed, held, po, yo in loso_streams(subjects, backbone, D, C, B):
        s_max, ceil_s = attainability_ceiling(po, yo)
        _, q_needed, shortfall = required_quantile(po, yo)
        bound = [static_cov(po, yo, corrected=False), aci_cov(po, yo)]
        if arms == 'all':
            bound += [dtaci_cov(po, yo), mondrian_cov(po, yo)]
        rows.append((held, s_max, ceil_s, q_needed, shortfall,
                     max(bound), dacr_cov(po, yo)))

    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r[1:])

    print(f"{'subj':>8}{'s_max':>9}{'Ceil(S)':>10}{'q_needed':>10}{'shortfall':>11}"
          f"{'best bound':>12}{'gap':>9}{'DACR':>9}{'binds':>7}")
    CEIL, BEST, FREE, SHORT, binds = [], [], [], [], 0
    for s in sorted(by, key=subject_key):
        s_max, ceil_s, q_needed, shortfall, best, free = np.array(
            by[s], dtype=float).mean(0)
        b = ceil_s < (1 - ALPHA)
        binds += int(b)
        CEIL.append(ceil_s); BEST.append(best); FREE.append(free)
        SHORT.append(shortfall)
        print(f"{s:>8}{s_max:9.4f}{ceil_s:10.4f}{q_needed:10.4f}{shortfall:>+11.4f}"
              f"{best:12.4f}{ceil_s - best:9.4f}{free:9.4f}"
              f"{'yes' if b else 'no':>7}")

    CEIL, BEST, FREE, SHORT = map(np.array, (CEIL, BEST, FREE, SHORT))
    n = len(CEIL)
    print(f"\n--- summary ({corpus}) ---")
    print(f"subject streams                 : {n}")
    print(f"bound held (best <= Ceil)       : {int((BEST <= CEIL + 1e-9).sum())} / {n}")
    print(f"largest violation (want <= 0)   : {np.max(BEST - CEIL):+.4f}")
    print(f"mean gap below own ceiling      : {np.mean(CEIL - BEST):+.4f}")
    print(f"exactly attained (gap < 1e-6)   : {int((CEIL - BEST < 1e-6).sum())} / {n}")
    print(f"ceiling binds (Ceil < {1 - ALPHA:.2f})      : {binds} / {n}")
    print(f"warm-up short of what is needed : "
          f"{int((SHORT > 0).sum())} / {n}   mean {SHORT.mean():+.4f}")
    if (CEIL < 1 - ALPHA).any():
        m = CEIL < 1 - ALPHA
        print(f"  binding subjects, Ceil(S)     : {CEIL[m].min():.3f} to {CEIL[m].max():.3f}")
        print(f"  binding subjects, coverage    : {BEST[m].min():.3f} to {BEST[m].max():.3f}")
    print(f"DACR (ceiling-free) coverage    : {FREE.mean():.3f} "
          f"sd {FREE.std(ddof=1):.3f}")
    print(f"DACR above its own ceiling      : {int((FREE > CEIL).sum())} / {n}"
          f"   (that is allowed: it is not thresholding the fixed multiset)")

    np.save(f'ceiling_{corpus}_{backbone}.npy', np.c_[CEIL, BEST, FREE, SHORT])
    print(f"saved -> ceiling_{corpus}_{backbone}.npy")


if __name__ == '__main__':
    main(*sys.argv[1:4])
