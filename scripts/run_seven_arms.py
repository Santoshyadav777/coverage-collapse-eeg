"""Seven-arm recalibration comparison. Produces tables 5 and 7.

    python scripts/run_seven_arms.py seedvig
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_audit import (LOADERS, MODELS, ARMS, ALPHA, WARM, ROLLW, SEEDS,
                         per_subject, paired_wilcoxon)


def main(corpus='seedvig'):
    subjects, D, C, B = LOADERS[corpus]()
    print(f"corpus={corpus}  subjects={len(subjects)}  D={D} ({C} channels x {B} bands)")
    print(f"target={1 - ALPHA:.2f}  WARM={WARM}  ROLLW={ROLLW}  seeds={SEEDS}\n")

    for backbone in MODELS:
        per, ids = per_subject(subjects, backbone, D, C, B, ARMS)
        n = len(per['DACR'])
        print(f"=== {backbone.upper()}  (n={n} subjects x {len(SEEDS)} seeds) ===")
        print(f"{'arm':>11}{'coverage':>11}{'sd':>9}{'min':>9}{'frac<0.85':>11}{'vs DACR p':>12}")
        for name, _ in ARMS:
            v = per[name]
            p = '-' if name == 'DACR' else f"{paired_wilcoxon(per['DACR'], v)[1]:.2e}"
            print(f"{name:>11}{v.mean():>11.3f}{v.std():>9.3f}{v.min():>9.3f}"
                  f"{(v < 0.85).mean():>11.2f}{p:>12}")
        for other in ('NexCP', 'Mond+roll'):
            d = per['DACR'] - per[other]
            print(f"  DACR - {other:<10}: median {np.median(d):+.4f}  "
                  f"lower variance on {int((d > 0).sum())}/{n}")
        print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'seedvig')
