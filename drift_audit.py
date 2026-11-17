"""
drift_audit.py
==============

Calibration and conformal-coverage audit of EEG fatigue monitors under
within-session drift and cross-subject transfer.

Reference implementation for:
    S Kumar and V K Jha, "Coverage collapse and deceptive calibration in EEG
    fatigue monitoring: a cross-dataset audit under within-session and
    cross-subject shift", Journal of Neural Engineering (2026).
    https://doi.org/10.1088/1741-2552/aebe6a

This single module holds the whole pipeline: the three corpus loaders, the
five backbones, the calibration metrics, the seven recalibration arms, the
attainability ceiling, and the deployment-feedback ablations. The scripts in
scripts/ are thin runners over it.

No EEG data ships with this repository. SEED-VIG and DROZY are released under
signed licence agreements and cannot be redistributed. Set the three paths in
the Paths section (or the matching environment variables) to your own copies.
"""

import os
import re
import glob
import inspect
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.signal import welch
from scipy.stats import wilcoxon, binomtest

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_TRAPZ = getattr(np, 'trapezoid', getattr(np, 'trapz', None))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SEEDVIG_ROOT = os.environ.get('SEEDVIG_ROOT', './data/SEED-VIG')
DROZY_ROOT = os.environ.get('DROZY_ROOT', './data/DROZY')
SADT_ROOT = os.environ.get('SADT_ROOT', './data/SADT')


# ---------------------------------------------------------------------------
# Protocol constants (section 4 of the paper)
# ---------------------------------------------------------------------------

EPOCHS = 80
LR = 1e-3
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4

ALPHA = 0.10              # target miscoverage, so the target coverage is 0.90
WARM = 50                 # labelled warm-start prefix, in windows
ROLLW = 50                # rolling-window length for quantile re-estimation
GAMMA = 0.05              # ACI step size

SEEDS = [0, 1, 2]
MODELS = ['mlp', 'cnn', 'eegnet', 'conformer', 'xgb']

TRAIN_FRACTION = 0.85     # of each training subject's windows
MIN_STREAM = WARM + 30    # a held-out subject shorter than this is skipped

# Within-session protocol of section 4.2. Each recording is cut by position in
# time and never shuffled: the first 40% is the training pool, a random 15% of
# that pool is discarded, 40% to 55% is the calibration block for the softmax
# temperature, and 55% to the end is where every within-session measurement is
# made. The gap keeps no evaluation window temporally adjacent to a training
# window. Every recording contributes to all three blocks, because the question
# is drift inside a recording rather than transfer between subjects.
TRAIN_END = 0.40
EARLY_HOLDOUT = 0.15      # of the training pool, discarded
CALIB_BLOCK = (0.40, 0.55)
LATE_START = 0.55
MIN_PER_CELL = 20         # a recording joins a bin's mean only at this many

# Where the binned evaluation block starts, as a fraction of the recording.
# Section 4.2 describes LATE_START for both corpora, and the SADT run does
# that. The SEED-VIG run binned the whole recording instead, so its first four
# deciles fall inside the 0 to 40% training window and are measured in sample.
# EVALUATE_FROM records what each published column actually used; pass
# evaluate_from=LATE_START to put both corpora on the protocol as written.
EVALUATE_FROM = {'seedvig': 0.0, 'sadt': LATE_START}
# Bins over session time, per corpus. SEED-VIG gets deciles because PERCLOS is
# annotated continuously; SADT gets quintiles because its labels are locked to
# lane-departure events and so carry an order of magnitude fewer units. DROZY
# is absent on purpose: one KSS rating covers a whole session, so it has no
# time-resolved label and cannot support this analysis at all.
SESSION_BINS = {'seedvig': 10, 'sadt': 5}

NEXCP_RHO = 0.99
DTACI_GAMMAS = [0.001, 0.005, 0.02, 0.08, 0.32]
DTACI_ETA = 2.72
DTACI_SIGMA = 1 / 500.

ECE_BINS = 15             # M in equation (3); the reliability diagrams use 10

# ---------------------------------------------------------------------------
# Corpus loaders
#
# Every loader returns (subjects, D, C, B):
#   subjects : {subject_id: (X, y)}, X float32 (n, D), y int (n,)
#   D        : feature dimension, equal to C * B
#   C, B     : channel count and band count
# ---------------------------------------------------------------------------

WIN_S = 8.0               # window length in seconds, all corpora

# Delta, theta, alpha, beta, gamma. Section 3.2 describes one band set for all
# three corpora, and these are the edges used for SEED-VIG and DROZY.
BANDS = [(1, 4), (4, 8), (8, 14), (14, 31), (31, 50)]

# SADT was run with slightly different edges for alpha, beta and gamma, and
# these are the ones behind its published numbers, so they are kept. The
# difference is in the edges only: five bands over the same 1 to 50 Hz span,
# the same 8 s window, the same construction. Results that turn on band edges
# at this scale are not reported in the paper, and the ceiling and the coverage
# comparison are properties of the score multiset rather than of the feature
# edges. Setting BANDS_SADT = BANDS makes all three corpora literally match
# section 3.2, at the cost of no longer reproducing the published SADT values.
BANDS_SADT = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def load_seedvig(root=None, by_recording=False):
    """SEED-VIG: 17 channels x 5 bands of differential entropy, PERCLOS > 0.5.

    The corpus holds 23 recordings from 21 subjects, and the two analyses use
    different units. by_recording=False groups a subject's recordings into one
    stream and gives the 21 subjects the cross-subject analysis transfers
    across. by_recording=True keeps the 23 recordings separate, which is the
    unit the within-session analysis needs, because time-on-task drift
    accumulates inside a recording.
    """
    root = root or SEEDVIG_ROOT
    feat = os.path.join(root, 'EEG_Feature_5Bands')
    lab = os.path.join(root, 'perclos_labels')
    if not os.path.isdir(feat):
        feat = [d for d in glob.glob(os.path.join(root, '**', 'EEG_Feature_5Bands'),
                                     recursive=True) if 'Forehead' not in d][0]
        lab = glob.glob(os.path.join(root, '**', 'perclos_labels'), recursive=True)[0]

    by = {}
    for path in sorted(glob.glob(os.path.join(feat, '*.mat'))):
        name = os.path.basename(path)
        lab_file = os.path.join(lab, name)
        if not os.path.exists(lab_file):
            continue
        de = loadmat(path)['de_LDS']
        perclos = loadmat(lab_file)['perclos'].reshape(-1).astype(np.float32)
        axis = int(np.argmin([abs(s - len(perclos)) for s in de.shape]))
        de = np.moveaxis(de, axis, 0)
        X = de.reshape(de.shape[0], -1).astype(np.float32)
        n = min(len(X), len(perclos))
        key = name[:-4] if by_recording else name.split('_')[0]
        by.setdefault(key, [[], []])
        by[key][0].append(X[:n])
        by[key][1].append((perclos[:n] > 0.5).astype(int))

    streams = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in by.items()}
    if not streams:
        raise RuntimeError(f'no SEED-VIG feature/label pairs found under {root}')
    dim = next(iter(streams.values()))[0].shape[1]
    if dim != 85:
        raise RuntimeError(f'expected 85 features (17 channels x 5 bands), got {dim}')
    return streams, 85, 17, 5


def _log_band_power(sig, fs, bands, win_s=WIN_S):
    """Log band power per channel per band over consecutive windows."""
    w = int(win_s * fs)
    n = sig.shape[1] // w
    out = []
    for i in range(n):
        seg = sig[:, i * w:(i + 1) * w]
        fr, P = welch(seg, fs=fs, nperseg=min(w, 1024), axis=1)
        row = []
        for c in range(seg.shape[0]):
            for lo, hi in bands:
                m = (fr >= lo) & (fr < hi)
                row.append(np.log(_TRAPZ(P[c, m], fr[m]) + 1e-12))
        out.append(row)
    return np.array(out, dtype=np.float32)


def load_drozy(root=None, kss_threshold=7):
    """DROZY: 5 channels x 5 bands of log band power, one KSS rating per session.

    The EDF files are read at their native 512 Hz. KSS is reported once per
    session, not continuously, so every window of a session carries the same
    label. That is why DROZY supports the cross-subject analysis and not the
    within-session one.
    """
    import mne
    root = root or DROZY_ROOT
    channels = ['Fz', 'Cz', 'C3', 'C4', 'Pz']
    kss = np.loadtxt(os.path.join(root, 'KSS.txt'))

    by = {}
    for edf in sorted(glob.glob(os.path.join(root, 'psg', '*.edf'))):
        m = re.match(r'(\d+)-(\d+)', os.path.basename(edf))
        if not m:
            continue
        subj, sess = int(m.group(1)), int(m.group(2))
        raw = mne.io.read_raw_edf(edf, preload=True, verbose='ERROR')
        missing = [c for c in channels if c not in raw.ch_names]
        if missing:
            raise RuntimeError(
                f'{os.path.basename(edf)} is missing channel(s) {missing}; the '
                f'25-dimensional feature needs all of {channels}')
        # picks= returns the rows in the order given, so the 25 columns line up
        # the same way for every session regardless of EDF channel order.
        feats = _log_band_power(raw.get_data(picks=channels),
                                raw.info['sfreq'], BANDS)
        y = int(kss[subj - 1, sess - 1] >= kss_threshold)
        by.setdefault(str(subj), [[], []])
        by[str(subj)][0].append(feats)
        by[str(subj)][1].append(np.full(len(feats), y))

    if not by:
        raise RuntimeError(f'no DROZY EDF files found under {root}/psg')
    subjects = {s: (np.concatenate(v[0]), np.concatenate(v[1])) for s, v in by.items()}
    return subjects, 25, 5, 5


# --- SADT (sustained-attention driving task, Cao et al. 2019) ---------------

DEV_CODES = ['251', '252']     # lane-deviation onset, left and right
RSP_CODE = '253'               # response onset
GLOBAL_WIN = 90.0              # seconds, per Wei et al.
RT_LO, RT_HI = 1.5, 2.5        # alert and fatigue multiples of the alert RT
MIN_PER_CLASS = 50             # session inclusion rule

# Section 3.1 says response times below 0.15 s are excluded as beneath the
# human visual-motor limit and those above 10 s as task disengagement. The runs
# behind the published SADT numbers did not apply that filter: the notebook
# computed the filtered set in a diagnostic cell and then labelled and
# extracted features from the unfiltered response times. load_sadt therefore
# defaults to unfiltered, and rt_bounds=RT_BOUNDS applies the stated filter.
#
# It is not cosmetic. The alert response time is the 5th percentile of the
# session's own response times, so dropping sub-0.15 s trials can lift that
# baseline, which moves both the 1.5x and 2.5x thresholds and so relabels
# trials and can change which sessions clear the 50-per-class floor.
RT_BOUNDS = (0.15, 10.0)


def sadt_reaction_times(raw):
    """(onset_s, rt_s, duration_s) from one loaded SADT session."""
    import mne
    fs = raw.info['sfreq']
    duration = raw.n_times / fs
    events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
    inv = {v: k for k, v in event_id.items()}
    seq = [(s / fs, inv.get(c, '?')) for s, _, c in events]

    onset, rt = [], []
    for i, (t, name) in enumerate(seq):
        if name in DEV_CODES:
            for t2, n2 in seq[i + 1:]:
                if n2 == RSP_CODE:
                    onset.append(t)
                    rt.append(t2 - t)
                    break
                if n2 in DEV_CODES:        # the response was missed
                    break
    return np.array(onset), np.array(rt), duration


def wei_labels(onset, rt):
    """Wei et al. labelling. Returns (label, alert_rt, global_rt).

    The subject's alert response time is the 5th percentile of the session's
    local response times. A trial is alert when the local and the 90 s global
    response time both fall below 1.5 times it, fatigued when both exceed 2.5
    times it, and transitional otherwise, which the binary formulation
    discards (label -1).

    The response times reaching here are unfiltered: see RT_BOUNDS and the
    rt_bounds argument of load_sadt.
    """
    alert = np.percentile(rt, 5)
    g = np.empty_like(rt)
    for i, t in enumerate(onset):
        m = (onset >= t - GLOBAL_WIN) & (onset <= t)
        g[i] = rt[m].mean() if m.sum() else rt[i]
    is_alert = (rt < RT_LO * alert) & (g < RT_LO * alert)
    is_fatigued = (rt > RT_HI * alert) & (g > RT_HI * alert)
    label = np.full(len(rt), -1)
    label[is_alert] = 0
    label[is_fatigued] = 1
    return label, alert, g


def differential_entropy(sig, fs, bands=BANDS_SADT):
    """Differential entropy per channel per band, flattened to (C * B,)."""
    f, P = welch(sig, fs=fs, nperseg=min(int(fs * 2), sig.shape[-1]), axis=-1)
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        p = _TRAPZ(P[..., m], f[m], axis=-1)
        out.append(np.log(np.maximum(p, 1e-12)))
    return np.stack(out, -1).reshape(-1)


def load_sadt(root=None, return_times=False, rt_bounds=None):
    """SADT: 30 channels x 5 bands, 8 s window ending at deviation onset.

    The window ends at the onset so no post-onset activity enters the features.

    rt_bounds=None reproduces the published numbers. rt_bounds=RT_BOUNDS
    applies the response-time exclusion section 3.1 describes, before the alert
    baseline is taken; see the note at RT_BOUNDS.
    """
    import mne
    mne.set_log_level('ERROR')
    root = root or SADT_ROOT
    sets = sorted(f for f in glob.glob(os.path.join(root, '**', '*.set'), recursive=True)
                  if os.path.isfile(f))

    X_by, T_by, S_by = {}, {}, {}
    for path in sets:
        tag = os.path.basename(path)[:-4]
        raw = mne.io.read_raw_eeglab(path, preload=True, verbose='ERROR')
        onset, rt, _ = sadt_reaction_times(raw)
        if rt_bounds is not None:
            keep_rt = (rt >= rt_bounds[0]) & (rt <= rt_bounds[1])
            onset, rt = onset[keep_rt], rt[keep_rt]
        if len(rt) == 0:
            del raw
            continue
        label, _, _ = wei_labels(onset, rt)
        if (label == 0).sum() < MIN_PER_CLASS or (label == 1).sum() < MIN_PER_CLASS:
            del raw
            continue

        fs = raw.info['sfreq']
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        sig = raw.get_data(picks=picks)
        n_t = sig.shape[1]

        X, y, T = [], [], []
        for i in np.where(label >= 0)[0]:
            b = int(onset[i] * fs)
            a = b - int(WIN_S * fs)
            if a < 0 or b > n_t:
                continue
            X.append(differential_entropy(sig[:, a:b], fs))
            y.append(int(label[i]))
            T.append(onset[i])
        del raw, sig
        if not X:
            continue

        s = tag.split('_')[0]
        X_by.setdefault(s, [[], []])
        T_by.setdefault(s, [])
        S_by.setdefault(s, [])
        X_by[s][0].append(np.asarray(X, np.float32))
        X_by[s][1].append(np.asarray(y))
        T_by[s].append(np.asarray(T))
        S_by[s].append(np.full(len(y), tag, dtype=object))

    subjects = {s: (np.concatenate(v[0]), np.concatenate(v[1])) for s, v in X_by.items()}
    D = next(iter(subjects.values()))[0].shape[1]
    B = len(BANDS_SADT)
    if return_times:
        times = {s: np.concatenate(v) for s, v in T_by.items()}
        sessions = {s: np.concatenate(v) for s, v in S_by.items()}
        return subjects, D, D // B, B, times, sessions
    return subjects, D, D // B, B


LOADERS = {'seedvig': load_seedvig, 'drozy': load_drozy, 'sadt': load_sadt}


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, D, C, B, k=2, h1=128, h2=64, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, k))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, D, C, B, k=2, f1=16, f2=32, dropout=DROPOUT):
        super().__init__()
        self.C, self.B = C, B
        self.net = nn.Sequential(
            nn.Conv2d(1, f1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(f1, f2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 2)), nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(f2 * 8, 64), nn.ReLU(), nn.Linear(64, k))

    def forward(self, x):
        return self.net(x.view(-1, 1, self.C, self.B))


class EEGNet(nn.Module):
    def __init__(self, D, C, B, k=2, F1=8, Dm=2, dropout=DROPOUT):
        super().__init__()
        self.C, self.B = C, B
        kt = min(3, B)
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kt), padding=(0, kt // 2), bias=False),
            nn.BatchNorm2d(F1))
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * Dm, (C, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * Dm), nn.ELU(), nn.Dropout(dropout))
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * Dm, F1 * Dm, (1, kt), padding=(0, kt // 2),
                      groups=F1 * Dm, bias=False),
            nn.Conv2d(F1 * Dm, F1 * Dm, 1, bias=False),
            nn.BatchNorm2d(F1 * Dm), nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.fc = nn.Linear(F1 * Dm, k)

    def forward(self, x):
        h = x.view(-1, 1, self.C, self.B)
        return self.fc(self.separable(self.depthwise(self.temporal(h))))


class Conformer(nn.Module):
    def __init__(self, D, C, B, k=2, dm=32, nh=4, L=2, dropout=DROPOUT):
        super().__init__()
        self.C, self.B = C, B
        # nn.MultiheadAttention requires the head count to divide the model
        # width. The default 32 / 4 does, but a tuned pair from the search
        # space of table S2 need not, so step the heads down until it does.
        while dm % nh:
            nh -= 1
        kt = min(3, B)
        self.embed = nn.Sequential(
            nn.Conv2d(1, dm, (1, kt), padding=(0, kt // 2)), nn.ELU(),
            nn.Conv2d(dm, dm, (C, 1)), nn.ELU())
        layer = nn.TransformerEncoderLayer(dm, nh, dm * 2, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, L)
        self.fc = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, k))

    def forward(self, x):
        h = self.embed(x.view(-1, 1, self.C, self.B)).squeeze(2).transpose(1, 2)
        return self.fc(self.encoder(h).mean(1))


NETS = {'mlp': MLP, 'cnn': CNN, 'eegnet': EEGNet, 'conformer': Conformer}


def fit_pred(name, Xtr, ytr, seed, D, C, B, cfg=None, n_classes=None):
    """Train one backbone. cfg overrides the defaults for the tuned replication.

    cfg keys the chosen backbone does not accept are rejected rather than
    ignored, so a typo or a key meant for another backbone fails here instead
    of silently leaving the default in place.

    n_classes defaults to what ytr contains, which is 2 for the binary
    formulation and 3 for the graded ordinal one of section 5.6.
    """
    cfg = cfg or {}
    k = int(n_classes if n_classes is not None else ytr.max() + 1)
    if name == 'xgb':
        import xgboost as xgb
        model = xgb.XGBClassifier(
            n_estimators=cfg.get('n_estimators', 300),
            max_depth=cfg.get('max_depth', 4),
            learning_rate=cfg.get('learning_rate', 0.1),
            subsample=cfg.get('subsample', 0.8),
            colsample_bytree=cfg.get('colsample_bytree', 0.8),
            eval_metric='logloss', random_state=seed, n_jobs=4, verbosity=0)
        model.fit(Xtr, ytr)
        return ('xgb', model)

    torch.manual_seed(seed)
    accepted = set(inspect.signature(NETS[name].__init__).parameters) - {'self', 'D', 'C', 'B'}
    net_cfg = {key: v for key, v in cfg.items() if key in accepted}
    unknown = set(cfg) - accepted - {'lr', 'weight_decay', 'epochs'}
    if unknown:
        raise TypeError(f"{name} does not accept {sorted(unknown)}; "
                        f"it accepts {sorted(accepted)}")
    net_cfg.setdefault('k', k)
    model = NETS[name](D, C, B, **net_cfg).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), cfg.get('lr', LR),
                           weight_decay=cfg.get('weight_decay', WEIGHT_DECAY))
    Xt = torch.tensor(Xtr).to(DEVICE)
    yt = torch.tensor(ytr).long().to(DEVICE)
    for _ in range(cfg.get('epochs', EPOCHS)):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(Xt), yt).backward()
        opt.step()
    return ('net', model)


def proba(fitted, X):
    """Class probabilities for either backbone kind."""
    kind, model = fitted
    if kind == 'xgb':
        p = model.predict_proba(X)
        return p if p.shape[1] == 2 else np.c_[1 - p, p]
    model.eval()
    with torch.no_grad():
        z = model(torch.tensor(X).to(DEVICE)).cpu().numpy()
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logits(fitted, X):
    """Pre-softmax outputs. Temperature scaling needs these, not probabilities."""
    kind, model = fitted
    if kind == 'xgb':
        raise TypeError('xgb exposes no logits; temperature scaling needs a net')
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(DEVICE)).cpu()


def softmax(z):
    """Row-wise softmax, in the dtype it is given.

    The dtype is deliberately preserved rather than promoted to float64: the
    backbones emit float32 and proba() stays in float32, so promoting here
    would make the same quantity differ in the last few digits depending on
    which path produced it.
    """
    z = np.asarray(z)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(z, y, max_iter=60, lr=0.1):
    """One temperature on held-out logits, by LBFGS on the NLL.

    Post-hoc temperature scaling, fitted on the calibration block of the
    within-session protocol. It rescales confidence without changing any
    prediction, so accuracy is untouched while ECE moves.

    The optimisation runs on log T rather than T, which keeps T positive and
    puts the search on a scale where a single LBFGS step range covers the
    useful range. Fitting T directly needs a much smaller step and many more
    iterations to reach the same optimum. See fit_temperature_direct.
    """
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y)).long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def fit_temperature_direct(z, y, max_iter=200, lr=0.01, min_temperature=0.05):
    """The temperature fit used for the SEED-VIG within-session figures.

    It optimises T directly from T = 1 with a small step and floors the result,
    and on this objective it stops well short of the optimum: against a grid
    search over the same NLL it can be off by more than 1.0 in T, where
    fit_temperature lands within 0.04. It is kept so the published SEED-VIG
    decile values can be reproduced exactly, and is not the recommended fit.

    Temperature is one global scalar applied to every window of every
    recording, so it cannot change which way the pooled and per-recording
    aggregations of section 5.1 differ: sweeping T from 0.25 to 20 leaves the
    per-recording figure above the pooled one in 9 or 10 of 10 bins. The
    values do move, substantially, so the two fits give different ECE levels
    while the direction of every effect is the same.
    """
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y)).long()
    t = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([t], lr=lr, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=min_temperature))


TEMPERATURE_FITS = {'logt': fit_temperature, 'direct': fit_temperature_direct}


# ---------------------------------------------------------------------------
# Calibration metrics (section 3.4)
# ---------------------------------------------------------------------------

def _bin_masks(conf, M=ECE_BINS):
    edges = np.linspace(0, 1, M + 1)
    return [((conf >= 0) & (conf <= edges[1])) if m == 0
            else ((conf > edges[m]) & (conf <= edges[m + 1]))
            for m in range(M)]


def ece(p, y, M=ECE_BINS):
    """Pooled expected calibration error, equation (3)."""
    if len(y) == 0:
        return np.nan
    conf, pred = p.max(1), p.argmax(1)
    acc = (pred == y).astype(float)
    total = 0.0
    for sel in _bin_masks(conf, M):
        if sel.sum():
            total += sel.mean() * abs(acc[sel].mean() - conf[sel].mean())
    return float(total)


def class_conditional_ece(p, y, k, M=ECE_BINS):
    """ECE restricted to the windows whose true class is k, equation (5)."""
    m = y == k
    return ece(p[m], y[m], M) if m.sum() else np.nan


def balanced_ece(p, y, K=2, M=ECE_BINS, strict=False):
    """Class-balanced ECE, equation (4): the unweighted mean over classes.

    strict=False averages over whichever classes are present, so a window with
    only one class still yields a number. That is what the within-session bins
    need, since a late bin can hold one class only.

    strict=True returns NaN unless every class is present. The cross-subject
    degradation of section 5.2 uses that, because there a value computed from
    one class is not comparable to one computed from two.
    """
    v = [class_conditional_ece(p, y, k, M) for k in range(K)]
    if strict and any(np.isnan(v)):
        return np.nan
    return float(np.nanmean(v)) if not all(np.isnan(v)) else np.nan


def brier(p, y):
    """Multiclass Brier score, equation (6).

    Proper, unlike any variant of ECE, so where the two disagree about the
    direction of a trend this one settles it (section 3.4).
    """
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1
    return float(((p - oh) ** 2).sum(1).mean())


def nll(p, y, eps=1e-12):
    """Negative log likelihood, equation (7). Also proper."""
    return float(-np.log(np.clip(p[np.arange(len(y)), y], eps, 1)).mean())


def signed_bias(p, y):
    """Mean confidence minus accuracy. Its sign is what cancels under pooling."""
    return float(p.max(1).mean() - (p.argmax(1) == y).mean())


def accuracy(p, y):
    return float((p.argmax(1) == y).mean())


# ---------------------------------------------------------------------------
# Conformal prediction: the seven recalibration arms (section 3.6, table 5)
#
# Every arm consumes the same nonconformity scores s(x, y) = 1 - p_y(x) from
# the same frozen backbone under the same labelled warm-start prefix, so any
# difference between them is a difference of thresholding rule alone.
# ---------------------------------------------------------------------------

def scores(po, yo):
    """s_t = 1 - p_{y_t}(x_t), equation (8)."""
    return 1 - po[np.arange(len(yo)), yo]


def quantile(s, level):
    """Empirical quantile with the upper rule, so the level is never undershot."""
    return np.quantile(np.asarray(s), np.clip(level, 0, 1), method='higher')


def conformal_level(n, alpha=ALPHA):
    """The quantile level for a calibration multiset of size n, equation (9).

    The (n + 1) correction is what makes the guarantee finite-sample rather
    than asymptotic. When alpha < 1 / (n + 1) the level exceeds 1 and the
    quantile convention returns the largest score, which is the trivial step
    of section 5.3.
    """
    return min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)


def conformal_quantile(s, alpha=ALPHA):
    """Threshold for a calibration multiset at the corrected level."""
    s = np.asarray(s)
    return quantile(s, conformal_level(len(s), alpha))


# Not every arm takes its threshold at the corrected level, and the split is
# deliberate. Do not unify it.
#
#   corrected, conformal_quantile    static, Mondrian + rolling, DACR
#       These threshold a multiset at a constant alpha, which is the setting
#       equation (9) is stated for.
#
#   plain 1 - alpha_t, quantile      ACI, DtACI
#       These adapt the level itself. Gibbs and Candes define them on the
#       plain empirical quantile of the calibration set at the current
#       alpha_t; applying the correction on top of an adapted level is a
#       different estimator, not a better one.
#
#   plain 1 - alpha, weighted        NexCP
#       Barber et al. obtain finite-sample validity from the test point held
#       at +infinity in the weighted multiset, not from an (n + 1) term.
#
#   plain 1 - alpha                  Mondrian
#       Kept as published, to match table 5.
#
# Changing any of these moves the arm away from the published numbers while
# the code still runs and still looks right.


# --- ceiling-bound arms: the calibration multiset is fixed at deployment ----

STATIC_CORRECTED = {'seedvig': False, 'drozy': False, 'sadt': True}


def static_cov(po, yo, alpha=ALPHA, corrected=True):
    """Static split conformal prediction.

    corrected=True takes the threshold at the finite-sample level of equation
    (9); corrected=False takes it at a plain 1 - alpha. With WARM = 50 and
    alpha = 0.10 that is the 0.92 against the 0.90 empirical quantile, one
    order statistic apart, so the two differ by a little in coverage.

    The published table 5 is not uniform here, which STATIC_CORRECTED records:
    its SEED-VIG column was produced with the plain level and its SADT column
    with the corrected one. Every other arm used one convention throughout.
    Pass corrected=STATIC_CORRECTED[corpus] to reproduce the table, and leave
    the default to compare arms on one footing.
    """
    s = scores(po, yo)
    q = (conformal_quantile(s[:WARM], alpha) if corrected
         else quantile(s[:WARM], 1 - alpha))
    return float((s[WARM:] <= q).mean())


def aci_cov(po, yo, alpha=ALPHA, gamma=GAMMA):
    """Adaptive conformal inference: the level moves, the multiset does not."""
    s = scores(po, yo)
    warm = s[:WARM]
    a = alpha
    cov = []
    for st in s[WARM:]:
        q = quantile(warm, 1 - a)
        c = int(st <= q)
        cov.append(c)
        a = np.clip(a + gamma * (alpha - (1 - c)), 1e-3, 1 - 1e-3)
    return float(np.mean(cov))


def aci_trivial_escape(po, yo, alpha=ALPHA, gamma=GAMMA):
    """ACI under the +infinity convention of section 3.5, which the published
    arms do not use. Returns (coverage, trivial_fraction, min_alpha).

    When the level falls below 1 / (n_cal + 1) the corrected quantile index
    exceeds n_cal, so the rule returns +infinity and the prediction set is the
    whole label space: the step is covered by emitting both classes. This
    counts those steps, and the escape feeds back, because a step covered that
    way raises alpha again through the same update. Computing the fraction
    without letting it feed back gives a different alpha trajectory and a
    different answer, so the two cannot be separated.

    Section 5.3 reports the fraction an unclipped implementation would reach.
    The arms themselves clip the level and cap the index instead, which is
    what aci_cov and dtaci_cov do and what every published coverage figure
    used.
    """
    s = scores(po, yo)
    warm = s[:WARM]
    threshold = 1.0 / (len(warm) + 1)
    a = alpha
    cov, trivial, a_min = [], 0, a
    for st in s[WARM:]:
        if a < threshold:
            trivial += 1
            cov.append(1)
        else:
            cov.append(int(st <= quantile(warm, 1 - a)))
        a = np.clip(a + gamma * (alpha - (1 - cov[-1])), 1e-3, 1 - 1e-3)
        a_min = min(a_min, a)
    return float(np.mean(cov)), trivial / len(cov), float(a_min)


def dtaci_cov(po, yo):
    """DtACI, Gibbs and Candes: an expert pool over ACI learning rates."""
    s = scores(po, yo)
    warm = s[:WARM]
    G = np.array(DTACI_GAMMAS)
    K = len(G)
    a_k = np.full(K, ALPHA)
    w_k = np.ones(K)
    cov = []
    for t in range(WARM, len(yo)):
        p_k = w_k / w_k.sum()
        a_bar = float(np.clip((p_k * a_k).sum(), 1e-3, 1 - 1e-3))
        q = quantile(warm, 1 - a_bar)
        cov.append(1 - int(s[t] > q))
        q_k = np.array([quantile(warm, 1 - a) for a in a_k])
        err = (s[t] > q_k).astype(float)
        loss = ALPHA * (ALPHA - err) - np.minimum(0, ALPHA - err)
        w_k = w_k * np.exp(-DTACI_ETA * loss)
        w_k = (1 - DTACI_SIGMA) * w_k / w_k.sum() + DTACI_SIGMA / K
        a_k = np.clip(a_k + G * (ALPHA - err), 1e-3, 1 - 1e-3)
    return float(np.mean(cov))


def mondrian_cov(po, yo, K=2):
    """Class-conditional (Mondrian) split CP, partitioned on the predicted label."""
    s = scores(po, yo)
    pred = po.argmax(1)
    qs = {}
    for k in range(K):
        m = pred[:WARM] == k
        qs[k] = (quantile(s[:WARM][m], 1 - ALPHA) if m.sum() >= 10
                 else quantile(s[:WARM], 1 - ALPHA))
    return float(np.mean([int(s[t] <= qs[pred[t]]) for t in range(WARM, len(yo))]))


# --- ceiling-free arms: deployment scores enter the calibration multiset ----

def _weighted_quantile(s, w, level):
    order = np.argsort(s)
    s, w = s[order], w[order]
    c = np.cumsum(w) / w.sum()
    return s[min(np.searchsorted(c, level, side='left'), len(s) - 1)]


def nexcp_cov(po, yo):
    """Weighted CP for non-exchangeable data, Barber et al.

    Age-decaying weights over the accumulating deployment stream, with the
    test point placed at +infinity. The state grows with session length.
    """
    s = scores(po, yo)
    hist = list(s[:WARM])
    cov = []
    for t in range(WARM, len(yo)):
        n = len(hist)
        w = np.append(NEXCP_RHO ** np.arange(n, 0, -1), 1.0)
        v = np.append(np.array(hist), np.inf)
        cov.append(int(s[t] <= _weighted_quantile(v, w, 1 - ALPHA)))
        hist.append(s[t])
    return float(np.mean(cov))


def mondrian_roll_cov(po, yo, K=2):
    """Class-conditional partition plus rolling re-estimation."""
    s = scores(po, yo)
    pred = po.argmax(1)
    buf = {k: list(s[:WARM][pred[:WARM] == k][-ROLLW:]) for k in range(K)}
    glob = list(s[:WARM][-ROLLW:])
    cov = []
    for t in range(WARM, len(yo)):
        k = pred[t]
        b = buf[k] if len(buf[k]) >= 10 else glob
        cov.append(int(s[t] <= conformal_quantile(b)))
        buf[k].append(s[t])
        buf[k] = buf[k][-ROLLW:]
        glob.append(s[t])
        glob = glob[-ROLLW:]
    return float(np.mean(cov))


def dacr_cov(po, yo, window=ROLLW, alpha=ALPHA, return_size=False):
    """DACR: the corrected quantile over a trailing window of deployment scores.

    The state does not grow with session length, which is what separates it
    from NexCP on cost (table 6). window and alpha are exposed for the two
    sensitivity sweeps of section 5.4, and are the paper's defaults otherwise.
    """
    s = scores(po, yo)
    buf = list(s[:WARM][-window:])
    cov, size = [], []
    for t in range(WARM, len(yo)):
        q = conformal_quantile(buf, alpha)
        cov.append(int(s[t] <= q))
        if return_size:
            size.append(prediction_set_size(po[t], q))
        buf.append(s[t])
        buf = buf[-window:]
    if return_size:
        return float(np.mean(cov)), float(np.mean(size))
    return float(np.mean(cov))


ARMS = [
    ('static', static_cov),
    ('ACI', aci_cov),
    ('NexCP', nexcp_cov),
    ('DtACI', dtaci_cov),
    ('Mondrian', mondrian_cov),
    ('Mond+roll', mondrian_roll_cov),
    ('DACR', dacr_cov),
]

CEILING_BOUND = ['static', 'ACI', 'DtACI', 'Mondrian']
CEILING_FREE = ['NexCP', 'Mond+roll', 'DACR']


# ---------------------------------------------------------------------------
# Attainability ceiling (section 3.8, equations 15 and 16)
# ---------------------------------------------------------------------------

def attainability_ceiling(po, yo):
    """(s_max, Ceil(S)) of equation (16).

    No rule that thresholds a multiset fixed at deployment can exceed Ceil(S),
    because its threshold is always a member of that multiset and so never
    exceeds s_max. The bound is a property of the score multiset and the
    deployment distribution, not of the adaptation rule, which is why it
    applies to all four ceiling-bound arms at once. Ceil(S) < 1 - alpha means
    the nominal target is unreachable for every one of them.
    """
    s = scores(po, yo)
    s_max = float(s[:WARM].max())
    return s_max, float((s[WARM:] <= s_max).mean())


def required_quantile(po, yo, alpha=ALPHA):
    """(s_max, q_needed, shortfall) for one stream.

    q_needed is the 1 - alpha quantile of the deployment scores themselves,
    which is the threshold a rule would have to reach to hit the target on
    this stream. shortfall is how far the fixed multiset falls short of it.
    A positive shortfall means no rule thresholding that multiset can get
    there, which is the mechanism behind the corpus difference discussed in
    section 6.2: the fraction of subjects whose warm-up set cannot supply the
    required quantile is what predicts how well level adaptation does.
    """
    s = scores(po, yo)
    s_max = float(s[:WARM].max())
    q_needed = float(np.quantile(s[WARM:], 1 - alpha))
    return s_max, q_needed, q_needed - s_max


def prediction_set_size(po_t, q):
    """Number of labels in C(x_t) = {y' : s(x_t, y') <= q}."""
    return int(((1 - po_t) <= q).sum())


# ---------------------------------------------------------------------------
# Deployment-feedback ablations (section 5.5, table 8)
# ---------------------------------------------------------------------------

def dacr_sparse_feedback(po, yo, interval=1, noise=0.0, rng=None):
    """DACR when the deployment label arrives only every interval-th window
    and is flipped with probability noise.

    Returns (coverage, mean set size, mean quantile).
    """
    rng = rng or np.random.RandomState(0)
    K = po.shape[1]
    buf = list(scores(po[:WARM], yo[:WARM])[-ROLLW:])
    q = conformal_quantile(buf)
    cov, size, qs = [], [], []
    for t in range(WARM, len(yo)):
        size.append(prediction_set_size(po[t], q))
        cov.append(int((1 - po[t, yo[t]]) <= q))
        qs.append(q)
        if (t - WARM) % interval == 0:
            y_obs = yo[t]
            if noise > 0 and rng.rand() < noise:
                y_obs = rng.choice([k for k in range(K) if k != yo[t]])
            buf.append(1 - po[t, y_obs])
            buf = buf[-ROLLW:]
            q = conformal_quantile(buf)
    return float(np.mean(cov)), float(np.mean(size)), float(np.mean(qs))


def dacr_asymmetric_noise(po, yo, p_a2f=0.0, p_f2a=0.0, burst=0, guard=None, rng=None):
    """DACR under direction-specific and bursty label corruption.

    p_a2f  : probability that an alert label is reported as fatigued
    p_f2a  : probability that a fatigued label is reported as alert
    burst  : if positive, corruption arrives in runs of this length
    guard  : None, or 'trim' to drop the top 5% of buffer scores before the
             quantile

    Returns (coverage, mean set size, mean quantile).
    """
    rng = rng or np.random.RandomState(0)
    buf = list(scores(po[:WARM], yo[:WARM])[-ROLLW:])

    def q_of(b):
        b = np.asarray(b)
        if guard == 'trim' and len(b) >= 20:
            b = b[b <= np.quantile(b, 0.95)]
        return conformal_quantile(list(b))

    q = q_of(buf)
    cov, size, qs = [], [], []
    burst_left = 0
    for t in range(WARM, len(yo)):
        size.append(prediction_set_size(po[t], q))
        cov.append(int((1 - po[t, yo[t]]) <= q))
        qs.append(q)

        if burst > 0:
            if burst_left > 0:
                corrupt = True
                burst_left -= 1
            else:
                rate = p_a2f if yo[t] == 0 else p_f2a
                corrupt = rng.rand() < rate / burst
                if corrupt:
                    burst_left = burst - 1
        else:
            rate = p_a2f if yo[t] == 0 else p_f2a
            corrupt = rng.rand() < rate

        y_obs = (1 - yo[t]) if corrupt else yo[t]
        buf.append(1 - po[t, y_obs])
        buf = buf[-ROLLW:]
        q = q_of(buf)
    return float(np.mean(cov)), float(np.mean(size)), float(np.mean(qs))


# ---------------------------------------------------------------------------
# Leave-one-subject-out protocol (section 4.2)
# ---------------------------------------------------------------------------

def subject_key(s):
    """Sort subject ids numerically whether or not they carry a letter prefix."""
    d = ''.join(ch for ch in str(s) if ch.isdigit())
    return int(d) if d else 0


def loso_streams(subjects, backbone, D, C, B, seeds=SEEDS, cfg=None):
    """Train leave-one-subject-out and yield the held-out subject's stream.

    Yields (seed, held_out_id, po, yo). The training data of each training
    subject is subsampled to TRAIN_FRACTION, and standardisation statistics
    come from the training split only, so no information from the held-out
    subject reaches the model.
    """
    ids = sorted(subjects, key=subject_key)
    for seed in seeds:
        rng = np.random.RandomState(seed)
        for held in ids:
            Xtr, ytr = [], []
            for s in ids:
                if s == held:
                    continue
                X, y = subjects[s]
                idx = rng.permutation(len(X))
                a = int(TRAIN_FRACTION * len(X))
                Xtr.append(X[idx[:a]])
                ytr.append(y[idx[:a]])
            if not Xtr:
                continue
            Xtr = np.concatenate(Xtr)
            ytr = np.concatenate(ytr)

            mu = Xtr.mean(0, keepdims=True)
            sd = Xtr.std(0, keepdims=True) + 1e-6
            standardise = lambda Z: ((Z - mu) / sd).astype(np.float32)

            fitted = fit_pred(backbone, standardise(Xtr), ytr, seed, D, C, B, cfg)
            Xo, yo = subjects[held]
            if len(yo) <= MIN_STREAM:
                continue
            yield seed, held, proba(fitted, standardise(Xo)), yo


# ---------------------------------------------------------------------------
# Within-session protocol (section 4.2)
# ---------------------------------------------------------------------------

def within_session_split(recordings, nbins, seed=0, min_per_cell=MIN_PER_CELL,
                         evaluate_from=LATE_START):
    """Cut each recording by position in time, per section 4.2.

    Returns (Xtr, ytr, Xcal, ycal, late, dropped). late maps a recording id to
    its evaluation block; dropped lists the recordings whose evaluation block
    is too short to fill nbins cells, with the length that disqualified them.
    Nothing is shuffled across the time axis: the only randomness is which 15%
    of the training pool is discarded.

    evaluate_from sets where the evaluation block begins. LATE_START is the
    protocol as section 4.2 states it and is what the SADT column used.
    evaluate_from=0.0 bins the whole recording, which is what the SEED-VIG
    column used; its early bins then sit inside the training window and are
    measured in sample, so an early-against-late contrast on that setting is
    partly a train-against-test gap rather than drift alone. See EVALUATE_FROM.
    """
    rng = np.random.RandomState(seed)
    Xtr, ytr, Xcal, ycal = [], [], [], []
    late, dropped = {}, []
    # Iterate in the loader's own order, which is sorted file order and so is
    # deterministic. The order is load-bearing: it fixes the draw sequence that
    # decides which 15% of each training pool is discarded.
    for r in recordings:
        X, y = np.asarray(recordings[r][0]), np.asarray(recordings[r][1])
        T = len(y)
        end = int(TRAIN_END * T)
        c0, c1 = int(CALIB_BLOCK[0] * T), int(CALIB_BLOCK[1] * T)
        l0 = int(evaluate_from * T)
        if (T - l0) < nbins * min_per_cell:
            dropped.append((r, T - l0))
            continue
        pool = np.arange(end)
        held = rng.choice(pool, max(1, int(EARLY_HOLDOUT * len(pool))), replace=False)
        keep = np.setdiff1d(pool, held)
        Xtr.append(X[keep])
        ytr.append(y[keep])
        Xcal.append(X[c0:c1])
        ycal.append(y[c0:c1])
        late[r] = (X[l0:], y[l0:])
    if not late:
        raise RuntimeError(
            'no recording survived: every evaluation block is shorter than '
            f'{nbins} x {min_per_cell} windows. Lower nbins or min_per_cell.')
    return (np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xcal), np.concatenate(ycal), late, dropped)


def within_session_predictions(recordings, nbins, backbone='mlp', D=None, C=None,
                               B=None, seed=0, min_per_cell=MIN_PER_CELL,
                               cfg=None, temperature_fit=fit_temperature,
                               evaluate_from=LATE_START):
    """Fit on the early blocks, temperature-scale on the calibration block, and
    return temperature-scaled predictions over the evaluation blocks.

    Returns (probs, bins, temperature, late, dropped), where probs and bins are
    keyed by recording id and bins[r] gives each evaluation window's bin index
    by its position within its own recording, so bin 0 holds the earliest
    deployment windows of every recording and the last bin the latest.
    """
    Xtr, ytr, Xcal, ycal, late, dropped = within_session_split(
        recordings, nbins, seed, min_per_cell, evaluate_from)
    if D is None:
        D = Xtr.shape[1]

    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    standardise = lambda Z: ((Z - mu) / sd).astype(np.float32)

    fitted = fit_pred(backbone, standardise(Xtr), ytr, seed, D, C, B, cfg)
    temperature = temperature_fit(logits(fitted, standardise(Xcal)), ycal)

    probs, bins = {}, {}
    for r, (X, y) in late.items():
        probs[r] = softmax(logits(fitted, standardise(X)).numpy() / temperature)
        bins[r] = (np.arange(len(y)) * nbins // len(y)).clip(0, nbins - 1)
    return probs, bins, temperature, late, dropped


def session_bin_table(probs, bins, late, nbins, min_per_cell=MIN_PER_CELL):
    """Pooled and per-recording metrics for each session-time bin.

    Every window of a bin enters the pooled figure. A recording enters the
    per-recording mean of a bin only when it supplies at least min_per_cell
    windows to it, which is the filter of section 4.2. The two differ by the
    order of averaging and by nothing else.
    """
    keys = ('ece', 'balanced_ece', 'brier', 'ece_fatigued')
    fns = (ece, balanced_ece, lambda p, y: brier(p, y),
           lambda p, y: class_conditional_ece(p, y, 1))
    pooled_p = [[] for _ in range(nbins)]
    pooled_y = [[] for _ in range(nbins)]
    per = {k: [[] for _ in range(nbins)] for k in keys}

    for r, (_, y) in late.items():
        p, q = probs[r], bins[r]
        for b in range(nbins):
            m = q == b
            if not m.sum():
                continue
            pooled_p[b].append(p[m])
            pooled_y[b].append(y[m])
            if m.sum() >= min_per_cell:
                for k, fn in zip(keys, fns):
                    per[k][b].append(fn(p[m], y[m]))

    out = {'n_recordings': [len(per['ece'][b]) for b in range(nbins)]}
    P = [np.concatenate(pooled_p[b]) for b in range(nbins)]
    Y = [np.concatenate(pooled_y[b]) for b in range(nbins)]
    out['pooled'] = {k: np.array([fn(P[b], Y[b]) for b in range(nbins)])
                     for k, fn in zip(keys, fns)}
    out['pooled']['accuracy'] = np.array([accuracy(P[b], Y[b]) for b in range(nbins)])
    out['pooled']['fatigued_fraction'] = np.array(
        [float((Y[b] == 1).mean()) for b in range(nbins)])
    out['per_recording'] = {
        k: np.array([float(np.nanmean(per[k][b])) if per[k][b] else np.nan
                     for b in range(nbins)]) for k in keys}
    out['per_recording_values'] = per
    return out


def zero_shot_loso(subjects, backbone, D, C, B, seeds=SEEDS, alpha=ALPHA, cfg=None):
    """Cross-subject protocol with the 70/15/15 split of section 4.2.

    This is the other half of the cross-subject protocol, and it is not what
    loso_streams does. Here the training subjects are split 70% for training,
    15% for conformal calibration and 15% for an in-distribution test fold, and
    the conformal threshold comes from the training-subject calibration fold
    rather than from a prefix of the deployment stream. That is what makes this
    zero-shot, and what separates it from the static arm of table 5: static
    sees 50 labelled windows from the deployment subject, this sees none.

    Yields one record per (seed, held-out subject):
      held, ccece_in, ccece_out, coverage, set_size
    where the two class-conditional values are measured on the in-distribution
    fold and on the held-out subject with the same trained model, so their
    difference is degradation and not a difference of model.
    """
    ids = sorted(subjects, key=subject_key)
    for seed in seeds:
        rng = np.random.RandomState(seed)
        for held in ids:
            parts = {k: [[], []] for k in ('tr', 'cal', 'id')}
            for s in ids:
                if s == held:
                    continue
                X, y = subjects[s]
                idx = rng.permutation(len(X))
                a, b = int(0.70 * len(X)), int(0.85 * len(X))
                for key, sl in (('tr', idx[:a]), ('cal', idx[a:b]), ('id', idx[b:])):
                    parts[key][0].append(X[sl])
                    parts[key][1].append(y[sl])
            if not parts['tr'][0]:
                continue
            Xtr, ytr = np.concatenate(parts['tr'][0]), np.concatenate(parts['tr'][1])
            Xcal, ycal = np.concatenate(parts['cal'][0]), np.concatenate(parts['cal'][1])
            Xid, yid = np.concatenate(parts['id'][0]), np.concatenate(parts['id'][1])

            mu = Xtr.mean(0, keepdims=True)
            sd = Xtr.std(0, keepdims=True) + 1e-6
            standardise = lambda Z: ((Z - mu) / sd).astype(np.float32)
            fitted = fit_pred(backbone, standardise(Xtr), ytr, seed, D, C, B, cfg)

            ccece_in = balanced_ece(proba(fitted, standardise(Xid)), yid, strict=True)

            s_cal = scores(proba(fitted, standardise(Xcal)), ycal)
            q_hat = conformal_quantile(s_cal, alpha)          # equation (9)

            Xo, yo = subjects[held]
            p_out = proba(fitted, standardise(Xo))
            ccece_out = balanced_ece(p_out, yo, strict=True)
            s_out = scores(p_out, yo)
            coverage = float((s_out <= q_hat).mean())          # equation (11)
            set_size = float(((1 - p_out) <= q_hat).sum(1).mean())   # equation (12)

            yield held, ccece_in, ccece_out, coverage, set_size


def per_subject(subjects, backbone, D, C, B, fns, seeds=SEEDS, cfg=None):
    """Run a set of named stream functions and average each over seeds.

    fns is a list of (name, fn) where fn(po, yo) returns a scalar. The return
    is {name: array over subjects}.
    """
    acc = {name: collections.defaultdict(list) for name, _ in fns}
    for _, held, po, yo in loso_streams(subjects, backbone, D, C, B, seeds, cfg):
        for name, fn in fns:
            acc[name][held].append(fn(po, yo))
    ids = sorted(acc[fns[0][0]], key=subject_key)
    return {name: np.array([np.mean(acc[name][s]) for s in ids]) for name, _ in fns}, ids


# ---------------------------------------------------------------------------
# Statistics (section 4.6)
# ---------------------------------------------------------------------------

def paired_wilcoxon(a, b, alternative='greater'):
    """One-sided paired Wilcoxon signed-rank test. Returns (W, p)."""
    return wilcoxon(np.asarray(a), np.asarray(b), alternative=alternative)


def sign_test(successes, n, alternative='greater'):
    """One-sided sign test, used for the all-bins result of section 5.1."""
    return binomtest(successes, n, 0.5, alternative=alternative).pvalue


# ---------------------------------------------------------------------------
# Self-test
#
#     python drift_audit.py
#
# Runs every backbone and every arm on synthetic data, so the module can be
# checked on a machine that has no EEG data. It does not reproduce the paper;
# it only shows that nothing is broken.
# ---------------------------------------------------------------------------

def self_test(verbose=True):
    rng = np.random.RandomState(0)
    D, C, B, n = 85, 17, 5, 400

    # A stream that is confident throughout but accurate only in its first half.
    conf = np.clip(rng.beta(6, 2, n), 0.5, 0.999)
    po = np.c_[conf, 1 - conf]
    yo = np.where(np.arange(n) < n // 2, 0, rng.binomial(1, 0.5, n))

    if verbose:
        print("backbones")
    X = rng.randn(64, D).astype(np.float32)
    y = rng.binomial(1, 0.5, 64)
    for name in MODELS:
        quick = {'n_estimators': 8} if name == 'xgb' else {'epochs': 2}
        try:
            fitted = fit_pred(name, X, y, 0, D, C, B, cfg=quick)
            p = proba(fitted, X)
            assert p.shape == (64, 2), p.shape
            assert np.allclose(p.sum(1), 1, atol=1e-5)
            if verbose:
                print(f"  {name:<10} ok   mean confidence {p.max(1).mean():.3f}")
        except ImportError as exc:
            if verbose:
                print(f"  {name:<10} skipped ({exc})")

    if verbose:
        print("\nrecalibration arms, target %.2f" % (1 - ALPHA))
    cov = {}
    for name, fn in ARMS:
        cov[name] = fn(po, yo)
        if verbose:
            print(f"  {name:<11} {cov[name]:.3f}")

    s_max, ceil_s = attainability_ceiling(po, yo)
    worst = max(abs(cov[m] - ceil_s) if cov[m] > ceil_s else 0 for m in CEILING_BOUND)
    assert worst <= 1e-9, f"a ceiling-bound arm exceeded Ceil(S) by {worst}"

    # The four reductions the notebook checks in its own sanity cell. Each one
    # collapses an arm onto a simpler arm it must agree with, so any of them
    # breaking points at the arm rather than at the data.
    if verbose:
        print("\nreductions")

    saved = globals()['DTACI_GAMMAS']
    try:
        for g in (0.01, 0.05, 0.20):
            globals()['DTACI_GAMMAS'] = [g]
            a, b = dtaci_cov(po, yo), aci_cov(po, yo, gamma=g)
            assert abs(a - b) < 0.02, f"DtACI[{g}] {a:.4f} != ACI[{g}] {b:.4f}"
        if verbose:
            print("  DtACI with one expert matches ACI at that step size")
    finally:
        globals()['DTACI_GAMMAS'] = saved

    saved = globals()['NEXCP_RHO']
    try:
        globals()['NEXCP_RHO'] = 1.0
        s = scores(po, yo)
        hist, c = list(s[:WARM]), []
        for t in range(WARM, len(yo)):
            c.append(int(s[t] <= conformal_quantile(hist)))
            hist.append(s[t])
        a, b = nexcp_cov(po, yo), float(np.mean(c))
        assert abs(a - b) < 0.02, f"NexCP[rho=1] {a:.4f} != full history {b:.4f}"
        if verbose:
            print("  NexCP at rho = 1 matches unweighted full-history CP")
    finally:
        globals()['NEXCP_RHO'] = saved

    s = scores(po, yo)
    q = quantile(s[:WARM], 1 - ALPHA)
    one_cell = float(np.mean([int(s[t] <= q) for t in range(WARM, len(yo))]))
    plain_static = static_cov(po, yo, corrected=False)
    assert abs(one_cell - plain_static) < 5e-3, 'Mondrian with one cell != static'
    if verbose:
        print("  Mondrian with a single cell matches static split CP")

    buf = list(s[:WARM][-ROLLW:])
    frozen = float(np.mean([int(s[t] <= conformal_quantile(buf))
                            for t in range(WARM, len(yo))]))
    assert abs(frozen - static_cov(po, yo)) < 0.03, 'frozen DACR != static'
    if verbose:
        print("  DACR with the buffer frozen matches static split CP")
    if verbose:
        print(f"\nceiling     s_max {s_max:.4f}   Ceil(S) {ceil_s:.4f}")
        print("  every ceiling-bound arm at or below Ceil(S): yes")
        print("  ceiling-free arms: "
              + ", ".join(f"{m} {cov[m]:.3f}" for m in CEILING_FREE))

    if verbose:
        print("\nmetrics")
        print(f"  ECE {ece(po, yo):.4f}   balanced ECE {balanced_ece(po, yo):.4f}")
        print(f"  Brier {brier(po, yo):.4f}   NLL {nll(po, yo):.4f}"
              f"   signed bias {signed_bias(po, yo):+.4f}")

    if verbose:
        print("\nfeedback ablations")
    for k in (1, 5, 30):
        c, sz, _ = dacr_sparse_feedback(po, yo, interval=k)
        if verbose:
            print(f"  kappa {k:<3} coverage {c:.3f}  set size {sz:.3f}")
    for e in (0.0, 0.20):
        c, sz, _ = dacr_sparse_feedback(po, yo, interval=1, noise=e,
                                        rng=np.random.RandomState(1))
        if verbose:
            print(f"  noise {e:<4.2f} coverage {c:.3f}  set size {sz:.3f}")

    if verbose:
        print("\nwithin-session protocol")
    recs = {}
    for i in range(1, 9):
        T = int(rng.choice([300, 520]))
        yr = (rng.rand(T) < np.linspace(0.15, 0.7, T)).astype(int)
        Xr = (rng.randn(T, D) + yr[:, None] * 0.7).astype(np.float32)
        recs[str(i)] = (Xr, yr)
    recs['99'] = (rng.randn(60, D).astype(np.float32), rng.binomial(1, 0.5, 60))
    nb = 5
    pr, bn, temp, late, dropped = within_session_predictions(
        recs, nb, 'mlp', D, C, B, seed=0, cfg={'epochs': 5})
    assert [r for r, _ in dropped] == ['99'], dropped
    assert '99' not in late
    tab = session_bin_table(pr, bn, late, nb)
    assert all(len(v) == nb for v in tab['pooled'].values())
    if verbose:
        print(f"  kept {len(late)} recordings, dropped {[r for r, _ in dropped]}"
              f", temperature {temp:.3f}")
        print("  pooled ECE by bin      "
              + " ".join(f"{v:.3f}" for v in tab['pooled']['ece']))
        print("  per-recording ECE      "
              + " ".join(f"{v:.3f}" for v in tab['per_recording']['ece']))

    p15 = sign_test(15, 15)
    assert abs(p15 - 3.0517578125e-05) < 1e-12
    if verbose:
        print(f"\nstatistics\n  sign test 15 of 15: p = {p15:.3e}")
        print("\nself-test passed")
    return True


if __name__ == '__main__':
    self_test()
