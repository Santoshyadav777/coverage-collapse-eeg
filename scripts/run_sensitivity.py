"""Window size and target tracking. Produces section 5.4.

Two ablations on the one hyperparameter DACR has. Sweeping the rolling window
over {30, 50, 75, 100, 150} tests whether the method needs a tuned operating
point; varying the requested coverage over {0.80, 0.90, 0.95} tests whether it
delivers the level it is asked for rather than one value that happens to sit
near 0.90.

    python scripts/run_sensitivity.py seedvig
    python scripts/run_sensitivity.py seedvig mlp,cnn
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_audit import (LOADERS, SEEDS, ALPHA, ROLLW, loso_streams,
                         static_cov, dacr_cov)

WINDOWS = [30, 50, 75, 100, 150]
TARGETS = [0.80, 0.90, 0.95]


def main(corpus='seedvig', backbones='mlp,cnn'):
    names = backbones.split(',')
    subjects, D, C, B = LOADERS[corpus]()

    for name in names:
        streams = [(po, yo) for _, _, po, yo in
                   loso_streams(subjects, name, D, C, B)]
        print(f"=== {name.upper()}  corpus={corpus}  "
              f"{len(SEEDS)} seeds, {len(streams)} streams ===")
        print(f"static baseline coverage "
              f"{np.mean([static_cov(p, y) for p, y in streams]):.3f}\n")

        print(f"  window sweep at a {1 - ALPHA:.2f} target")
        print(f"  {'W':>6}{'coverage':>11}{'sd':>8}{'set size':>11}")
        covs = []
        for w in WINDOWS:
            out = np.array([dacr_cov(p, y, window=w, return_size=True)
                            for p, y in streams])
            covs.append(out[:, 0].mean())
            print(f"  {w:>6}{out[:, 0].mean():>11.3f}{out[:, 0].std():>8.3f}"
                  f"{out[:, 1].mean():>11.2f}")
        covs = np.array(covs)
        print(f"  range {covs.min():.3f} to {covs.max():.3f}, a spread of "
              f"{covs.max() - covs.min():.3f} for a "
              f"{max(WINDOWS) // min(WINDOWS)}-fold change in window\n")

        print(f"  target tracking at W = {ROLLW}")
        print(f"  {'requested':>11}{'achieved':>11}{'sd':>8}{'error':>9}"
              f"{'set size':>11}")
        worst = 0.0
        for target in TARGETS:
            out = np.array([dacr_cov(p, y, alpha=1 - target, return_size=True)
                            for p, y in streams])
            err = out[:, 0].mean() - target
            worst = max(worst, abs(err))
            print(f"  {target:>11.2f}{out[:, 0].mean():>11.3f}"
                  f"{out[:, 0].std():>8.3f}{err:>+9.3f}{out[:, 1].mean():>11.2f}")
        print(f"  tracks the request to within {worst:.3f} at every level\n")


if __name__ == '__main__':
    main(*sys.argv[1:3])
