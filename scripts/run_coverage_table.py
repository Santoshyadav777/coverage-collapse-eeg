"""Coverage at a 0.90 target across five backbones and three corpora.
Produces table 7.

Three arms only: static one-shot recalibration on the labelled prefix, ACI
level adaptation, and DACR rolling re-estimation. Means over three seeds.

    python scripts/run_coverage_table.py
    python scripts/run_coverage_table.py seedvig,sadt
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from drift_audit import (LOADERS, MODELS, SEEDS, ALPHA, per_subject,
                         static_cov, aci_cov, dacr_cov)

ARMS = [('static', static_cov), ('ACI', aci_cov), ('DACR', dacr_cov)]


def main(corpora=None, backbones=None):
    corpora = corpora.split(',') if corpora else ['seedvig', 'drozy', 'sadt']
    names = backbones.split(',') if backbones else MODELS

    print(f"target {1 - ALPHA:.2f}, means over {len(SEEDS)} seeds\n")
    header = f"{'backbone':>11}"
    for c in corpora:
        header += f"{c + ' static':>16}{'ACI':>8}{'DACR':>8}"
    print(header)
    print('-' * len(header))

    table = {}
    counts = {}
    for name in names:
        line = f"{name:>11}"
        for corpus in corpora:
            subjects, D, C, B = LOADERS[corpus]()
            per, ids = per_subject(subjects, name, D, C, B, ARMS)
            counts[corpus] = len(ids)
            for arm, _ in ARMS:
                table[(name, corpus, arm)] = float(per[arm].mean())
            line += (f"{per['static'].mean():>16.3f}"
                     f"{per['ACI'].mean():>8.3f}{per['DACR'].mean():>8.3f}")
        print(line)

    print()
    for corpus in corpora:
        print(f"{corpus}: {counts[corpus]} subject streams")

    print(f"\n--- mean over the {len(names)} backbones ---")
    print(f"{'corpus':>11}{'static':>10}{'ACI':>8}{'DACR':>8}"
          f"{'DACR - best fixed':>20}")
    for corpus in corpora:
        v = {arm: np.mean([table[(n, corpus, arm)] for n in names])
             for arm, _ in ARMS}
        best_fixed = max(v['static'], v['ACI'])
        print(f"{corpus:>11}{v['static']:>10.3f}{v['ACI']:>8.3f}"
              f"{v['DACR']:>8.3f}{v['DACR'] - best_fixed:>+20.3f}")

    np.save('coverage_table.npy',
            np.array([[table[(n, c, a)] for c in corpora for a, _ in ARMS]
                      for n in names]))
    print("\nsaved -> coverage_table.npy")


if __name__ == '__main__':
    main(*sys.argv[1:3])
