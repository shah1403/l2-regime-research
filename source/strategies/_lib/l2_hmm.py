"""Gaussian HMM regime detector -- the causal, soft-probability alternative to
the hand-thresholded VR/autocorrelation detector.

Design choices that make it honest and useful:
  * observations per snapshot: [drift-z, log-vol-ratio] -- drift-to-noise over a
    trailing window (trend axis) and fast/slow vol ratio (activity axis).
    Plain returns would only discover volatility clusters, not trend-vs-MR.
  * K=3 diagonal-covariance states fit by EM.
  * decoding is FILTERED (forward pass only): P(state_t | obs_1..t). Viterbi or
    smoothed posteriors peek at the future and quietly leak -- the classic HMM
    backtest bug.
  * walk-forward: fit on day k, decode day k+1. No same-day fitting.
  * state -> label mapping from the FITTING day's stats: highest |mean drift-z|
    state = DIR; of the rest, the one with more negative return autocorr = MR;
    remainder = FLAT.

reg_hmm() returns labels in the same {1,-1,0} vocabulary as compute_regime so it
drops into l2_mmdir unchanged.
"""
from __future__ import annotations

import numpy as np

DRIFT_W = 60
VOL_FAST, VOL_SLOW = 30, 300
K = 3
EM_ITERS = 40
SEED = 7


def features(mid):
    r = np.diff(mid, prepend=mid[0])
    def roll_mean(x, w):
        c = np.cumsum(np.insert(x, 0, 0.0))
        out = np.full(len(x), np.nan)
        out[w - 1:] = (c[w:] - c[:-w]) / w
        return out
    mu = roll_mean(r, DRIFT_W)
    v = roll_mean(r * r, DRIFT_W)
    sd = np.sqrt(np.maximum(v - mu**2, 1e-18))
    drift_z = mu * np.sqrt(DRIFT_W) / (sd + 1e-12)
    vf = np.sqrt(roll_mean(r * r, VOL_FAST))
    vs = np.sqrt(roll_mean(r * r, VOL_SLOW))
    lvr = np.log((vf + 1e-12) / (vs + 1e-12))
    X = np.column_stack([drift_z, lvr])
    X[~np.isfinite(X)] = 0.0
    return X, r


class GaussHMM:
    def __init__(self, k=K, seed=SEED):
        self.k = k; self.rng = np.random.default_rng(seed)

    def _emis(self, X):
        # log N(x | mu_k, diag sig_k)
        ll = np.zeros((len(X), self.k))
        for j in range(self.k):
            z = (X - self.mu[j]) / self.sig[j]
            ll[:, j] = -0.5 * (z**2).sum(1) - np.log(self.sig[j]).sum() \
                       - X.shape[1] / 2 * np.log(2 * np.pi)
        return ll

    def fit(self, X):
        n, d = X.shape
        q = np.quantile(X[:, 0], np.linspace(0.1, 0.9, self.k))
        mu0 = np.zeros((self.k, d)); mu0[:, 0] = q   # dimension-generic init seed
        self.mu = mu0 + 0.01 * self.rng.standard_normal((self.k, d))
        self.sig = np.ones((self.k, d))
        self.A = np.full((self.k, self.k), 0.02 / (self.k - 1))
        np.fill_diagonal(self.A, 0.98)
        self.pi = np.full(self.k, 1 / self.k)
        for _ in range(EM_ITERS):
            logB = self._emis(X)
            la, lb = np.zeros((n, self.k)), np.zeros((n, self.k))
            lA = np.log(self.A)
            la[0] = np.log(self.pi) + logB[0]
            for t in range(1, n):     # forward
                m = la[t - 1][:, None] + lA
                la[t] = logB[t] + np.logaddexp.reduce(m, axis=0)
            for t in range(n - 2, -1, -1):   # backward
                m = lA + (logB[t + 1] + lb[t + 1])[None, :]
                lb[t] = np.logaddexp.reduce(m, axis=1)
            lg = la + lb
            lg -= np.logaddexp.reduce(lg, axis=1, keepdims=True)
            g = np.exp(lg)
            xi = np.zeros((self.k, self.k))
            for t in range(n - 1):
                m = la[t][:, None] + lA + (logB[t + 1] + lb[t + 1])[None, :]
                m -= np.logaddexp.reduce(m.ravel())
                xi += np.exp(m)
            self.pi = g[0] / g[0].sum()
            self.A = xi / np.maximum(xi.sum(1, keepdims=True), 1e-12)
            for j in range(self.k):
                w = g[:, j][:, None]; W = np.maximum(w.sum(), 1e-9)
                self.mu[j] = (w * X).sum(0) / W
                self.sig[j] = np.sqrt(np.maximum(
                    (w * (X - self.mu[j])**2).sum(0) / W, 1e-6))
        self._gamma = g
        return self

    def filter(self, X):
        """Causal P(state_t | obs_1..t)."""
        logB = self._emis(X); n = len(X)
        lp = np.log(self.pi) + logB[0]
        out = np.zeros((n, self.k))
        lp -= np.logaddexp.reduce(lp); out[0] = np.exp(lp)
        lA = np.log(self.A)
        for t in range(1, n):
            lp = logB[t] + np.logaddexp.reduce(lp[:, None] + lA, axis=0)
            lp -= np.logaddexp.reduce(lp)
            out[t] = np.exp(lp)
        return out


def label_states(model, X_fit, r_fit):
    """Map HMM states to {1:DIR, -1:MR, 0:FLAT} using fitting-day stats only."""
    g = model._gamma
    hard = g.argmax(1)
    drift_mag = np.abs(model.mu[:, 0])
    dir_state = int(np.argmax(drift_mag))
    ac = {}
    for j in range(K):
        m = hard == j
        x = r_fit[m]
        if m.sum() > 50 and x.std() > 0:
            x = x - x.mean()
            ac[j] = float((x[:-1] * x[1:]).sum() / (x[:-1]**2).sum())
        else:
            ac[j] = 0.0
    rest = [j for j in range(K) if j != dir_state]
    mr_state = min(rest, key=lambda j: ac[j])
    lab = {dir_state: 1, mr_state: -1}
    for j in rest:
        if j != mr_state: lab[j] = 0
    return lab


def reg_hmm(mid_fit, mid_decode, p_min=0.6):
    """Fit on day k (mid_fit), causally decode day k+1 (mid_decode)."""
    Xf, rf = features(mid_fit)
    Xd, _ = features(mid_decode)
    m = GaussHMM().fit(Xf)
    lab = label_states(m, Xf, rf)
    P = m.filter(Xd)
    hard = P.argmax(1)
    conf = P[np.arange(len(P)), hard]
    reg = np.array([lab[s] for s in hard])
    reg[conf < p_min] = 0                       # uncertain -> FLAT (stand aside)
    return reg, P, lab
