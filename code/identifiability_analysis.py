"""Core experiments for the proposed contribution:

  "What can a short prefix identify?" -- decomposing entity-transfer forecasting
  error into a SHAPE component (transferable across entities) and a SCALE component
  (entity-idiosyncratic), and measuring how much of each a length-L prefix reveals.

E1  Oracle decomposition: replace our predicted scale with the true scale, and
    separately our predicted shape with the true shape, to attribute error.
E2  Prefix-length information curve: how well can a length-L prefix predict the
    future scale vs the future shape?
E3  Identifiability-derived shrinkage: shrink the predicted scale toward the
    population scale by exactly the amount the measured predictability warrants
    (a James-Stein-style correction whose intensity is estimated, not tuned).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/hkhosravi7/Documents/GitHub/DM_compettion')
from src.data import load_train, osi_trajectory, FREEZE_H, PRED_HOURS
from src.validate import make_folds, score_trajectory

train = load_train()
truth = osi_trajectory(train)
P = pd.read_csv('oof/ens_prod.csv')
piv = P.pivot_table(index='fipsCode', columns='hour', values='osi_pred')
fips = piv.index.to_numpy()
Y = truth.loc[fips, PRED_HOURS].to_numpy()          # true future trajectories
H = piv.reindex(columns=PRED_HOURS).to_numpy()      # our predicted trajectories

EPS = 1e-9
A_true = Y.sum(1)                                   # true scale (trajectory mass)
A_pred = H.sum(1)                                   # predicted scale
S_true = Y / (A_true[:, None] + EPS)                # true shape (sums to 1)
S_pred = H / (A_pred[:, None] + EPS)


def rmse_traj(M):
    d = pd.DataFrame(M, index=fips, columns=PRED_HOURS).stack().reset_index()
    d.columns = ['fipsCode', 'hour', 'osi_pred']
    return score_trajectory(d, truth).loc['mean', 'rmse']


print("=" * 74)
print("E1  ORACLE DECOMPOSITION -- where does the remaining error actually live?")
print("=" * 74)
base = rmse_traj(H)
o_scale = rmse_traj(A_true[:, None] * S_pred)        # true scale, our shape
o_shape = rmse_traj(A_pred[:, None] * S_true)        # our scale, true shape
print(f"  our ensemble                              RMSE {base:.5f}")
print(f"  + oracle SCALE (true mass, our shape)     RMSE {o_scale:.5f}"
      f"   ({100*(base-o_scale)/base:+.1f}% of error removed)")
print(f"  + oracle SHAPE (our mass, true shape)     RMSE {o_shape:.5f}"
      f"   ({100*(base-o_shape)/base:+.1f}% of error removed)")
print(f"  + both (perfect)                          RMSE 0.00000")
print("\n  Reading: whichever oracle removes more error is the component our models"
      "\n  are failing at, and therefore where any further modelling effort must go.")

print("\n" + "=" * 74)
print("E2  PREFIX-LENGTH INFORMATION CURVE -- what does a length-L prefix reveal?")
print("=" * 74)
full = truth.loc[fips]
print(f"  {'prefix L':>9s} {'r(scale)':>10s} {'R2(scale)':>10s} {'r(shape)':>10s}"
      f"   what a longer observation window buys")
for L in (12, 24, 36, 48, 60, 72, 96, 120):
    pre = full.loc[:, :L - 1].to_numpy()
    a_obs = pre.sum(1)                                        # observed mass
    # future window starts after the prefix so the two never overlap
    fut = full.loc[:, L + 1:215].to_numpy()
    a_fut = fut.sum(1)
    m = (a_obs > 0) & (a_fut > 0)
    r_scale = np.corrcoef(np.log1p(a_obs[m] * 100), np.log1p(a_fut[m] * 100))[0, 1]
    # shape similarity: cosine between the observed profile and the future profile
    # resampled to a common length, averaged over counties
    def resample(M, n=48):
        idx = np.linspace(0, M.shape[1] - 1, n).astype(int)
        return M[:, idx]
    so = resample(pre[m]); sf = resample(fut[m])
    so = so / (np.linalg.norm(so, axis=1, keepdims=True) + EPS)
    sf = sf / (np.linalg.norm(sf, axis=1, keepdims=True) + EPS)
    r_shape = float((so * sf).sum(1).mean())
    print(f"  {L:9d} {r_scale:10.3f} {r_scale**2:10.3f} {r_shape:10.3f}")
print("\n  r(scale): how well the prefix's outage mass predicts the future mass"
      "\n  r(shape): mean cosine similarity between observed and future profiles")

print("\n" + "=" * 74)
print("E3  IDENTIFIABILITY-DERIVED SCALE SHRINKAGE (nested)")
print("=" * 74)
folds = list(make_folds(train))
fold_of = {f: k for k, (_, va) in enumerate(folds) for f in va}
fold = np.array([fold_of[f] for f in fips])

nested = H.copy()
lams = []
for k in range(len(folds)):
    tr, te = fold != k, fold == k
    # how much of the variation in log-scale do we actually get right on train folds?
    lo_p, lo_t = np.log1p(A_pred[tr] * 100), np.log1p(A_true[tr] * 100)
    r = np.corrcoef(lo_p, lo_t)[0, 1]
    lam = max(0.0, min(1.0, r ** 2))            # shrink by the measured reliability
    lams.append(lam)
    mu = lo_t.mean()
    lo_new = mu + lam * (np.log1p(A_pred[te] * 100) - mu)
    A_new = np.expm1(lo_new) / 100
    nested[te] = S_pred[te] * A_new[:, None]
r_shrunk = rmse_traj(nested)
print(f"  measured scale reliability lambda = R^2(log predicted, log true) per fold:"
      f" {[round(l,3) for l in lams]}")
print(f"  our ensemble                        RMSE {base:.5f}")
print(f"  + identifiability-derived shrinkage RMSE {r_shrunk:.5f}  ({r_shrunk-base:+.5f})")
out = pd.DataFrame(nested, index=fips, columns=PRED_HOURS).stack().reset_index()
out.columns = ['fipsCode', 'hour', 'osi_pred']
out['osi_pred'] = out['osi_pred'].clip(lower=0)
out.to_csv('oof/ens_shrunkscale.csv', index=False)
print("  (saved oof/ens_shrunkscale.csv for the paired bootstrap)")
