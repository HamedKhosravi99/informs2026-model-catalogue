"""Falsifiable tests of a record-driven fragility model of storm outages.

HYPOTHESIS.  Each county holds a finite population of components with latent failure
thresholds theta ~ F_c.  A component fails the first time the forcing exceeds its
threshold.  Therefore cumulative damage is

        C_c(t) = N_c * F_c( M_c(t) ),      M_c(t) = max_{s<=t} u_c(s)

a MONOTONE function of the RUNNING MAXIMUM of the forcing, not of its current value.

T1  Damage occurs only at records: dC ~ 0 when the running max does not advance.
T2  Depletion is a difference of the same CDF: second-wave damage depends on
    F(M_2) - F(M_1), so a second wave that does not exceed the first breaks nothing.
T3  Predictability is governed by SUPPORT OVERLAP: the prefix identifies F_c only on
    [0, M(prefix)].  Counties whose future forcing exceeds their prefix maximum need
    F_c outside the identified region, and should be predicted worse.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/hkhosravi7/Documents/GitHub/DM_compettion')
from src.data import load_train, osi_trajectory, FREEZE_H, PRED_HOURS

tr = load_train()
tr = tr.sort_values(['fipsCode', 'hour'])
g = tr.groupby('fipsCode')

# gross new damage per hour, in customers, and the running max of gust
dmg = g['outageCount'].diff().clip(lower=0).fillna(0)
tr['dmg'] = dmg
tr['runmax'] = g['gust'].cummax()
tr['dM'] = g['runmax'].diff().fillna(0)

print("=" * 76)
print("T1  DOES DAMAGE OCCUR ONLY WHEN THE FORCING SETS A RECORD?")
print("=" * 76)
rec = tr['dM'] > 0.01
frac_rows = rec.mean()
dmg_at_rec = tr.loc[rec, 'dmg'].sum()
dmg_total = tr['dmg'].sum()
print(f"  hours where the running max advances: {100*frac_rows:.1f}% of all county-hours")
print(f"  share of ALL gross damage occurring in those hours: {100*dmg_at_rec/dmg_total:.1f}%")
print(f"  mean damage | record advancing : {tr.loc[rec,'dmg'].mean():8.1f} customers/h")
print(f"  mean damage | no new record    : {tr.loc[~rec,'dmg'].mean():8.1f} customers/h")
print(f"  ratio: {tr.loc[rec,'dmg'].mean()/max(tr.loc[~rec,'dmg'].mean(),1e-9):.1f}x")
# how much of the damage in non-record hours is just the tail of a recent record?
tr['hrs_since_record'] = tr.groupby('fipsCode')['dM'].transform(
    lambda s: (~(s > 0.01)).cumsum() - (~(s > 0.01)).cumsum().where(s > 0.01).ffill().fillna(0))
near = (~rec) & (tr['hrs_since_record'] <= 3)
print(f"  share of damage within 3h of a record (records + lagged response): "
      f"{100*(dmg_at_rec + tr.loc[near,'dmg'].sum())/dmg_total:.1f}%")

print("\n" + "=" * 76)
print("T2  IS THE SECOND WAVE A DIFFERENCE OF THE SAME FRAGILITY CURVE?")
print("=" * 76)
w1 = tr[(tr.hour >= 48) & (tr.hour < 108)]
w2 = tr[(tr.hour >= 108) & (tr.hour < 180)]
m1 = w1.groupby('fipsCode')['gust'].max()
m2 = w2.groupby('fipsCode')['gust'].max()
d1 = w1.groupby('fipsCode')['dmg'].sum()
d2 = w2.groupby('fipsCode')['dmg'].sum()
N = g['customersTracked'].median()
df = pd.DataFrame({'m1': m1, 'm2': m2, 'd1': d1, 'd2': d2, 'N': N}).dropna()
exceed = df.m2 > df.m1
print(f"  counties whose wave-2 peak gust EXCEEDS wave-1 peak: {exceed.sum()} of {len(df)}")
print(f"    wave-2 damage rate (as % of customers), exceeding counties : "
      f"{100*(df.loc[exceed,'d2']/df.loc[exceed,'N']).mean():.3f}%")
print(f"    wave-2 damage rate,                 non-exceeding counties : "
      f"{100*(df.loc[~exceed,'d2']/df.loc[~exceed,'N']).mean():.3f}%")
r = (df.loc[exceed, 'd2'] / df.loc[exceed, 'N']).mean() / max(
    (df.loc[~exceed, 'd2'] / df.loc[~exceed, 'N']).mean(), 1e-12)
print(f"    ratio: {r:.1f}x   <- theory predicts >> 1 (records break new components)")
# does the size of the exceedance predict wave-2 damage, given wave-1 damage?
sub = df[df.d1 > 0].copy()
sub['exc'] = (sub.m2 - sub.m1).clip(lower=0)
sub['rate2'] = sub.d2 / sub.N
print(f"  corr(exceedance magnitude, wave-2 damage rate) = "
      f"{sub[['exc','rate2']].corr().iloc[0,1]:+.3f}")
print(f"  corr(wave-2 ABSOLUTE gust,  wave-2 damage rate) = "
      f"{sub[['m2','rate2']].corr().iloc[0,1]:+.3f}   <- what a standard model uses")

print("\n" + "=" * 76)
print("T3  IS PREDICTION ERROR GOVERNED BY FORCING-SUPPORT OVERLAP?")
print("=" * 76)
truth = osi_trajectory(tr)
P = pd.read_csv('oof/ens_prod.csv')
piv = P.pivot_table(index='fipsCode', columns='hour', values='osi_pred')
fips = piv.index
Y = truth.loc[fips, PRED_HOURS].to_numpy()
H = piv.reindex(columns=PRED_HOURS).to_numpy()
per_county_rmse = np.sqrt(np.nanmean((H - Y) ** 2, axis=1))
gp = tr[tr.hour <= FREEZE_H].groupby('fipsCode')['gust'].max().reindex(fips)   # prefix max
gf = tr[tr.hour > FREEZE_H].groupby('fipsCode')['gust'].max().reindex(fips)    # future max
ratio = (gf / gp).to_numpy()
out_of_support = ratio > 1.0
print(f"  counties whose FUTURE peak gust exceeds their PREFIX peak: "
      f"{out_of_support.sum()} of {len(fips)}")
print(f"    mean per-county RMSE, in-support     : {per_county_rmse[~out_of_support].mean():.5f}")
print(f"    mean per-county RMSE, out-of-support : {per_county_rmse[out_of_support].mean():.5f}")
if out_of_support.sum() > 3:
    print(f"    ratio: {per_county_rmse[out_of_support].mean()/per_county_rmse[~out_of_support].mean():.2f}x")
print(f"  corr(future/prefix gust ratio, per-county RMSE) = "
      f"{np.corrcoef(ratio, per_county_rmse)[0,1]:+.3f}")
# control: is it just that windier counties are harder?
print(f"  control corr(future ABSOLUTE max gust, per-county RMSE) = "
      f"{np.corrcoef(gf.to_numpy(), per_county_rmse)[0,1]:+.3f}")
