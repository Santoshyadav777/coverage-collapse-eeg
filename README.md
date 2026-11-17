# Coverage collapse and deceptive calibration in EEG fatigue monitoring

Reference implementation for:

> S Kumar and V K Jha, "Coverage collapse and deceptive calibration in EEG fatigue
> monitoring: a cross-dataset audit under within-session and cross-subject shift",
> *Journal of Neural Engineering* (2026).
> [https://doi.org/10.1088/1741-2552/aebe6a](https://doi.org/10.1088/1741-2552/aebe6a)

This is a reference implementation, not the original analysis code. The work
was carried out across a long research notebook; this module is a clean,
readable consolidation of the method, the protocols and the metrics, meant to
show what was done and to be built on.

The exact code behind the published numbers is available from the corresponding
author on reasonable request.

The paper audits whether an EEG fatigue monitor's stated confidence survives two
shifts: drift within a single recording, and transfer to an unseen person. It
measures calibration pooled, per recording and per class; compares seven
conformal recalibration strategies under one protocol; and derives a ceiling on
the coverage attainable by any rule that thresholds a calibration set fixed at
deployment.

## Data

**This repository contains no EEG data.** Two of the three corpora are released
under signed licence agreements and cannot be redistributed in raw or derived
form. Obtain each from its provider and point the loaders at your own copy.

| Corpus | Channels | Rate | Features | Label | Access |
|---|---|---|---|---|---|
| SEED-VIG | 17 | 200 Hz | 85-dim differential entropy | PERCLOS > 0.5, continuous | Signed licence from the BCMI Laboratory, Shanghai Jiao Tong University: https://bcmi.sjtu.edu.cn/home/seed/seed-vig.html |
| DROZY | 5 | 512 Hz | 25-dim log band power | KSS >= 7, one per session | Signed end-user licence from the University of Liège: http://www.drozy.ulg.ac.be/ |
| SADT (Cao et al. 2019) | 30 | 500 Hz | 150-dim log band power | response time, per event | Openly available from figshare: https://doi.org/10.6084/m9.figshare.6427334 |

The DROZY host is `www.drozy.ulg.ac.be`; some printed references to it drop a
dot. The address in this table is the working one.

Set the paths before running anything:

```bash
export SEEDVIG_ROOT=/path/to/SEED-VIG
export DROZY_ROOT=/path/to/DROZY
export SADT_ROOT=/path/to/SADT
```

The SADT loader uses the raw 500 Hz recordings described in the accompanying
*Scientific Data* descriptor, not the pre-processed release.

Which unit a corpus is loaded at depends on the analysis, and the two must not
be conflated. SEED-VIG holds 23 recordings from 21 subjects: the cross-subject
scripts use the 21 subjects, because a subject is what transfer is across,
while `run_within_session.py` loads the 23 recordings through
`load_seedvig(by_recording=True)`, because a recording is what time-on-task
drift accumulates inside. SADT is loaded at subject level, aggregating a
subject's sessions, which gives 11 streams. DROZY has one KSS rating per
session and therefore no time-resolved label, so it supports the cross-subject
analysis only; `run_within_session.py` refuses it rather than producing a
meaningless answer.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A GPU is optional. `drift_audit.py` falls back to CPU, which is slower for the
Conformer backbone but gives the same numbers.

To check the install without any EEG data:

```bash
python drift_audit.py
```

That trains all five backbones on random features, runs all seven arms and the
metrics on a synthetic drifting stream, runs the feedback ablations, walks the
within-session protocol end to end, and asserts that no ceiling-bound arm
exceeds its own ceiling and that the short recording is dropped. It confirms
the install, not the paper.

## Layout

```
drift_audit.py      the whole pipeline in one module
scripts/            thin runners, one per block of results
```

`drift_audit.py` holds, in order: the three corpus loaders, the five backbones
with their fit, predict and temperature-scaling wrappers, the calibration
metrics, the seven recalibration arms, the attainability ceiling, the
deployment-feedback ablations, the two evaluation protocols, and the
statistical tests.

## Reproducing the results

| Script | Produces |
|---|---|
| `scripts/run_within_session.py <corpus> [backbone] [bins] [seed] [temp_fit] [evaluate_from]` | Section 5.1, figures 3 and 4, table S2 |
| `scripts/run_cross_subject.py <corpus> [backbones]` | Section 5.2, tables 3 and 4 |
| `scripts/run_seven_arms.py <corpus>` | Section 5.3, table 5, figure 8 |
| `scripts/run_ceiling.py <corpus> [backbone]` | Section 5.3, table S1, figure 7 |
| `scripts/run_coverage_table.py [corpora] [backbones]` | Section 5.4, table 7 |
| `scripts/run_sensitivity.py <corpus> [backbones]` | Section 5.4, window and target sweeps |
| `scripts/run_feedback_grid.py <corpus> [backbone]` | Section 5.5, table 8, figure 9 |

For example:

```bash
python scripts/run_within_session.py seedvig
python scripts/run_within_session.py seedvig mlp 10 42 logt late   # uniform window
python scripts/run_within_session.py sadt
python scripts/run_cross_subject.py seedvig
python scripts/run_seven_arms.py sadt
python scripts/run_ceiling.py drozy mlp
python scripts/run_coverage_table.py
python scripts/run_sensitivity.py seedvig mlp
python scripts/run_feedback_grid.py seedvig mlp
```

Corpus names are `seedvig`, `drozy` and `sadt`. Backbone names are `mlp`,
`cnn`, `eegnet`, `conformer` and `xgb`. `run_within_session.py` defaults per
corpus: deciles over the whole recording for SEED-VIG, quintiles from 55%
onward for SADT. Its last two arguments override the temperature fit and the
evaluation window.

`run_cross_subject.py` carries reference values for SEED-VIG and DROZY and
prints its own output beside them, so a run says immediately how close the
environment lands.

The tuned-backbone comparison has no runner. It needs a hyperparameter search
on a tuning split disjoint from the evaluation subjects; the backbones here accept those search keys
(`h1`, `h2`, `f1`, `f2`, `F1`, `Dm`, `dm`, `nh`, `L`, `dropout`, `lr`,
`epochs`) through `fit_pred`'s `cfg`, and `fit_pred` rejects a key the chosen
backbone does not take rather than ignoring it. Tables 1, 2, 6, S2 and S3 are
literature, configuration and timing tables rather than measured results.

## Protocol

Two protocols are used, one for each shift, and they differ in what is held
out.

The cross-subject protocol is leave-one-subject-out with three seeds. Each
training subject contributes 85% of its windows, and the standardisation
statistics come from the training split alone, so nothing from the held-out
subject reaches the model. The first 50 windows of the held-out stream are the
labelled warm start; every arm is scored on what follows. A held-out stream of
80 windows or fewer is skipped. This is what `loso_streams` yields, and what
the arms, the ceiling and the feedback grid run on.

The within-session protocol holds the model fixed and asks how its calibration
behaves as one recording proceeds. Each recording is cut by position in time
and never shuffled: the first 40% is the training pool, a random 15% of that
pool is discarded, the block from 40% to 55% carries the softmax temperature,
and every measurement is made from 55% to the end. The gap leaves no evaluation
window next to a training window. Every recording contributes to all three
blocks. Evaluation windows are then binned by position within their own
recording, so bin 1 holds the earliest deployment windows of every recording.
A recording joins a bin's per-recording mean only when it supplies at least 20
windows to that bin, and a recording whose evaluation block cannot fill every
bin is dropped and reported. This is `within_session_predictions` and
`session_bin_table`.

The pooled and per-recording figures come from the same predictions. Only the
order of averaging differs.

The seven arms consume the same nonconformity scores from the same frozen
backbone, so any difference between them is a difference of thresholding rule.
Four of them threshold a multiset fixed at deployment and are bound by the
ceiling; three admit deployment scores into the multiset and are not.

| Arm | Admits deployment scores | State grows with session |
|---|---|---|
| Static split CP | no | no |
| ACI | no | no |
| DtACI | no | no |
| Mondrian | no | no |
| NexCP | yes | yes |
| Mondrian + rolling | yes | no |
| DACR | yes | no |

Three arms take their threshold at the finite-sample corrected level
`ceil((n+1)(1-alpha))/n` of equation 9: static split CP, Mondrian + rolling and
DACR. The other four do not, for reasons given in the comment above the arms in
`drift_audit.py`: ACI and DtACI adapt the level itself and are defined on the
plain empirical quantile at the current alpha, NexCP gets its finite-sample
validity from the test point held at infinity, and Mondrian is kept as
published. Unifying the convention changes four of the seven arms while the
code still runs.

Defaults, all in the constants block of `drift_audit.py`: target coverage 0.90,
warm start 50 windows, rolling window 50, ACI step 0.05, NexCP decay 0.99,
DtACI experts {0.001, 0.005, 0.02, 0.08, 0.32} with eta 2.72 and sigma 1/500,
ECE with 15 bins throughout, which is what every ECE value quoted in the
paper uses; the reliability diagrams of figure 5 are drawn with 10 bins for
legibility, not for estimation.

## Configurable conventions

A few details of the pipeline can reasonably be set more than one way, and this
module makes each of them an explicit switch rather than a hidden decision. The
defaults are the settings behind the numbers reported in the paper. Each switch
is documented where it is defined, with the reasoning and with what changes if
you flip it.

| Switch | Default | Alternative |
|---|---|---|
| `EVALUATE_FROM` | per corpus: whole recording for SEED-VIG, 55% onward for SADT | `LATE_START` for both |
| `STATIC_CORRECTED` | per corpus | one level for both |
| `rt_bounds` | unfiltered response times | `RT_BOUNDS`, a 0.15 to 10 s exclusion |
| `BANDS_SADT` | 1-4, 4-8, 8-13, 13-30, 30-45 Hz | `BANDS`, the SEED-VIG and DROZY edges |
| `temperature_fit` | `fit_temperature`, optimising log T | `fit_temperature_direct`, optimising T |
| `balanced_ece(strict=)` | average over classes present | NaN unless every class is present |

Two magnitudes, measured rather than assumed, for the two that move numbers
most. Putting ACI, DtACI and Mondrian on the corrected quantile level raises
them by 0.013 to 0.022, which leaves the gap to DACR at 0.121 where it was
0.134. Temperature is one global scalar applied to every window, so sweeping it
from 0.25 to 20 leaves the per-recording figure above the pooled one in 9 or 10
of 10 bins: the levels move, the ordering does not.

None of the switches changes the direction of any result.

## Notes

Numbers vary slightly with the PyTorch and CUDA version even at a fixed seed.
The qualitative results in the paper, the direction of every effect and the
separation between the ceiling-bound and ceiling-free arms, do not depend on
that variation.

The work was carried out in a notebook; this module is a consolidation of it,
with the duplicated definitions, exploratory cells and hard-coded figure data
left out.

Every one of the 99 notebook cells was read before this module was settled,
which is how the four differences above were found; three of them are invisible
to a search and only show up in which variable a later cell actually consumes.

What was then checked numerically against the original cells, on synthetic
streams chosen to hit the awkward branches (lengths from 90 to 900 windows,
flipped predicted-class sides, one-class warm-up windows, buffers short enough
to trigger the Mondrian fallback): all seven recalibration arms, ECE, both
class-conditional ECEs, balanced ECE, Brier, signed bias, s_max, Ceil(S), the
Wei labelling with its alert baseline and 90 s global response time, the
sparse-feedback and label-noise ablations, both temperature fits, and the whole
within-session pipeline including the split, the dropped-recording list, the
temperature, the bin indices and the per-bin pooled and per-recording figures.
All of these agree to the last bit against the SADT within-session cell. Against
the SEED-VIG one the pooled figure agrees to 1.2e-7, because that cell pools its
recordings in a different order and floating-point addition is not associative.
Results are quoted to four decimals.

`python drift_audit.py` also runs the four reductions the notebook checks in a
sanity cell of its own, since each one collapses an arm onto a simpler arm it
has to agree with: DtACI with a single expert must equal ACI at that step size,
NexCP at rho = 1 must equal unweighted full-history CP, Mondrian with one cell
must equal static split CP, and DACR with its buffer frozen must equal static
split CP. If one of those breaks, the arm is at fault and not the data.

What has not been checked that way: the three corpus loaders, since that needs
the data. The channel sets, sampling rates, thresholds, window length and
labelling rules follow sections 3.1 and 3.2, and the loaders now fail loudly on
a missing channel or an unexpected feature dimension rather than carrying a
silent mismatch forward, but they have not been run against the corpora here.
Run one script on real data and check it against the paper before relying on
them.

Section 5.6 (table 9) has no runner here; see Reproducing the results.

## Citation

```bibtex
@article{kumar2026coverage,
  author  = {Kumar, Santosh and Jha, Vijay Kumar},
  title   = {Coverage collapse and deceptive calibration in {EEG} fatigue
             monitoring: a cross-dataset audit under within-session and
             cross-subject shift},
  journal = {Journal of Neural Engineering},
  year    = {2026},
  doi     = {10.1088/1741-2552/aebe6a}
}
```

## License

MIT, see `LICENSE`. The licence covers this code only, not the datasets.
