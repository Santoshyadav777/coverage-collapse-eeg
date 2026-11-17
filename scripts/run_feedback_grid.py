"""Sparse and noisy deployment feedback. Produces table 8 and figure 9.

    python scripts/run_feedback_grid.py seedvig mlp
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_audit import LOADERS, ALPHA, loso_streams, dacr_sparse_feedback

INTERVALS = [1, 2, 5, 10, 30, 60]     # kappa: a label every kappa-th window
NOISE = [0.00, 0.05, 0.10, 0.20]      # label-flip probability


def main(corpus='seedvig', backbone='mlp'):
    subjects, D, C, B = LOADERS[corpus]()
    streams = [(po, yo) for _, _, po, yo in
               loso_streams(subjects, backbone, D, C, B)]
    print(f"corpus={corpus}  backbone={backbone}  streams={len(streams)}  "
          f"target={1 - ALPHA:.2f}\n")

    print("--- feedback interval, clean labels ---")
    print(f"{'kappa':>7}{'coverage':>11}{'set size':>11}")
    for k in INTERVALS:
        out = [dacr_sparse_feedback(po, yo, interval=k) for po, yo in streams]
        cov, size, _ = np.array(out).mean(0)
        print(f"{k:>7}{cov:>11.3f}{size:>11.3f}")

    print("\n--- label noise, every window ---")
    print(f"{'noise':>7}{'coverage':>11}{'set size':>11}")
    for e in NOISE:
        out = [dacr_sparse_feedback(po, yo, interval=1, noise=e,
                                    rng=np.random.RandomState(i))
               for i, (po, yo) in enumerate(streams)]
        cov, size, _ = np.array(out).mean(0)
        print(f"{e:>7.2f}{cov:>11.3f}{size:>11.3f}")

    print("\n--- full grid, coverage / mean set size ---")
    print(f"{'kappa':>7}" + "".join(f"{e:>18.2f}" for e in NOISE))
    for k in INTERVALS:
        cells = []
        for e in NOISE:
            out = [dacr_sparse_feedback(po, yo, interval=k, noise=e,
                                        rng=np.random.RandomState(i))
                   for i, (po, yo) in enumerate(streams)]
            cov, size, _ = np.array(out).mean(0)
            cells.append(f"{cov:>10.3f} /{size:>6.2f}")
        print(f"{k:>7}" + "".join(cells))


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['seedvig', 'mlp']))
