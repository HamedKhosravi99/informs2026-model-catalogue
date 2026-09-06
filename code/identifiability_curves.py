"""Corrected prefix-length curve + a three-way oracle attribution.

The first prefix-length experiment was confounded: the evaluation window moved with
L, so longer prefixes were scored on a quieter (easier, shorter) future. Here the
evaluation window is held FIXED at hours 121-215 for every L, so the curve measures
information gained from observation length and nothing else.

Adds a TIMING oracle (best per-county time shift), giving a three-way attribution of
error into scale, shape and phase -- the analogue of amplitude/phase decompositions
used in spatial weather verification, applied to entity-transfer forecasting.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/hkhosravi7/Documents/GitHub/DM_compettion')
from src.data import load_train, osi_trajectory, PRED_HOURS
from src.validate import score_trajectory

train = load_train()
truth = osi_trajectory(train)
P = pd.read_csv('oof/ens_prod.csv')
piv = P.pivot_table(index='fipsCode', columns='hour', values='osi_pred')
fips = piv.index.to_numpy()
Y = truth.loc[fips, PRED_HOURS].to_numpy()
H = piv.reindex(columns=PRED_HOURS).to_numpy()
EPS = 1e-9

print("=" * 76)
print("E2b  PREFIX-LENGTH CURVE with a FIXED evaluation window (hours 121-215)")
print("=" * 76)
FUT = np.arange(121, 216)
fut = truth.loc[fips, FUT].to_numpy()
a_fut = fut.sum(1)
sh_fut = fut / (a_fut[:, None] + EPS)
print(f"  {'prefix L':>9s} {'r(scale)':>10s} {'R2(scale)':>10s} {'r(shape)':>10s}")
for L in (12, 24, 36, 48, 60, 72, 84, 96, 108, 120):
    pre = truth.loc[fips, :L - 1].to_numpy()
    a_obs = pre.sum(1)
    m = (a_obs > 0) & (a_fut > 0)
    r_scale = np.corrcoef(np.log1p(a_obs[m] * 100), np.log1p(a_fut[m] * 100))[0, 1]
    n = 48
    idx = np.linspace(0, pre.shape[1] - 1, n).astype(int)
    so = pre[m][:, idx]
    so = so / (np.linalg.norm(so, axis=1, keepdims=True) + EPS)
    idx2 = np.linspace(0, fut.shape[1] - 1, n).astype(int)
    sf = sh_fut[m][:, idx2]
    sf = sf / (np.linalg.norm(sf, axis=1, keepdims=True) + EPS)
    print(f"  {L:9d} {r_scale:10.3f} {r_scale**2:10.3f} {float((so*sf).sum(1).mean()):10.3f}")
print("  (evaluation window identical for every row, so differences are information,")
print("   not a change of task difficulty)")
print()
print("  WARNING (audit 2026-08-14): the r(shape) column above is NOT interpretable")
print("  as a skill measure. It is a cosine between two non-negative vectors, which")
print("  is bounded well above zero by construction (random non-negative baseline")
print("  ~0.75), and the level says nothing about matching. Against a shuffled-county")
print("  control - each county's prefix paired with a RANDOM other county's future")
print("  shape - the excess is +0.002 / -0.016 / -0.018 / +0.013 / +0.041 / +0.030")
print("  at L = 12/24/48/72/96/120: tiny, and negative at two lengths. Own-county")
print("  prefix shape barely beats a random county's. Do NOT cite this column as")
print("  evidence that shape is predictable; the support for that claim is I05")
print("  archetype routing (0.01022, 0.0008 from its oracle), which is a real")
print("  held-out forecast comparison. See REPORT 5ak.")


def rmse_traj(M):
    d = pd.DataFrame(M, index=fips, columns=PRED_HOURS).stack().reset_index()
    d.columns = ['fipsCode', 'hour', 'osi_pred']
    return score_trajectory(d, truth).loc['mean', 'rmse']


print("\n" + "=" * 76)
print("E4  THREE-WAY ORACLE ATTRIBUTION: scale, shape, phase")
print("=" * 76)
base = rmse_traj(H)
A_t, A_p = Y.sum(1), H.sum(1)
S_t, S_p = Y / (A_t[:, None] + EPS), H / (A_p[:, None] + EPS)

# phase oracle: allow each county's prediction to slide to its best time offset.
# EDGE-PADDED, not np.roll: a circular roll wraps the quiet tail (mean truth
# 0.00035) into the wave-1 decay head (mean truth 0.01935), so every nonzero
# shift paid a large artificial seam penalty and the argmin collapsed to 0.
# That bug produced the "timing is free, 2% of MSE" claim; the physical shift
# gives 16.8% (audit, 2026-08-14).
def _shift(row, s):
    out = np.empty_like(row)
    if s > 0:
        out[:s], out[s:] = row[0], row[:-s]
    elif s < 0:
        out[s:], out[:s] = row[-1], row[-s:]
    else:
        out[:] = row
    return out


best = H.copy()
shifts = []
for i in range(len(fips)):
    e, s = min(((((_shift(H[i], s) - Y[i]) ** 2).sum(), s) for s in range(-12, 13)))
    shifts.append(s)
    best[i] = _shift(H[i], s)
r_phase = rmse_traj(best)
print(f"  our ensemble                        RMSE {base:.5f}")
print(f"  + oracle SCALE (true county mass)   RMSE {rmse_traj(A_t[:, None]*S_p):.5f}"
      f"   ({100*(1-(rmse_traj(A_t[:,None]*S_p)/base)**2):.0f}% of MSE attributable)")
print(f"  + oracle SHAPE (true profile)       RMSE {rmse_traj(A_p[:, None]*S_t):.5f}"
      f"   ({100*(1-(rmse_traj(A_p[:,None]*S_t)/base)**2):.0f}% of MSE attributable)")
print(f"  + oracle PHASE (best +/-12h shift)  RMSE {r_phase:.5f}"
      f"   ({100*(1-(r_phase/base)**2):.0f}% of MSE attributable)")
sh = np.array(shifts)
print(f"  optimal shifts: median {np.median(sh):+.0f}h, "
      f"|shift|<=2h for {100*(np.abs(sh)<=2).mean():.0f}% of counties")
print("\n  Reading: if one component dominated, its oracle would remove most of the")
print("  error. Whether it does is the diagnostic's answer for where to spend effort.")
