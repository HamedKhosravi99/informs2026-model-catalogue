"""R6 — low-dimensional transition-bias correction, gated by the R5 diagnosis.

R5 found a strong systematic bias in the FLOWS, not in the final OSI: damage is
predicted at 0.66-0.70x of truth in both the teacher-forced and closed-loop regimes,
while restoration swings from 1.13x (teacher-forced) to 0.79x (closed loop). This is
the pattern R6 is designed for -- correct the transition process, not the output
residual (which already failed, section 5m).

Exactly two parameters, both learned nested inside each outer fold by rolling the model
forward on the OUTER-TRAINING counties only and matching cumulative flows:

    damage_t  <- c_dmg * damage_t
    restore_t <- c_res * restore_t

No county-specific terms. Adoption requires stable coefficients across folds.
"""
import numpy as np
import pandas as pd

from src.data import FREEZE_H, PRED_HOURS, load_train, mask_after_freeze, osi_trajectory
from src.validate import make_folds, score_trajectory
from src.stockflow import (StockFlowModel, _weather_matrices, _static_matrix,
                           _state_block, reconstruct_osi, WCOLS)


def rollout(m, df, c_dmg=1.0, c_res=1.0):
    W = _weather_matrices(df)
    S, fips, _ = _static_matrix(df)
    Pobs = np.nan_to_num(df.pivot_table(index="fipsCode", columns="hour", values="P_t")
                         .sort_index().reindex(columns=np.arange(216)).to_numpy())
    n = len(fips)
    P = np.zeros((n, 216)); P[:, :FREEZE_H + 1] = Pobs[:, :FREEZE_H + 1]
    d = np.diff(P[:, :FREEZE_H + 1], axis=1, prepend=P[:, :1])
    UP = np.zeros((n, 216)); DN = np.zeros((n, 216))
    UP[:, :FREEZE_H + 1] = np.clip(d, 0, None); DN[:, :FREEZE_H + 1] = np.clip(-d, 0, None)
    cum = UP[:, :FREEZE_H + 1].sum(1)
    last = np.where(UP[:, :FREEZE_H + 1] > 0, np.arange(FREEZE_H + 1), -99).max(1)
    hrs = np.where(last < 0, 99.0, FREEZE_H - last).astype(float)
    uh, dh = UP[:, FREEZE_H - 5:FREEZE_H + 1].copy(), DN[:, FREEZE_H - 5:FREEZE_H + 1].copy()
    for t in range(FREEZE_H + 1, 216):
        stock = P[:, t - 1]
        X = np.hstack([_state_block(stock, uh, dh, cum, hrs, W["gust"][:, t]),
                       np.column_stack([W[c][:, t] for c in WCOLS]), S])
        p = m.clf.predict_proba(X)[:, 1]
        mag = np.clip(m.reg.predict(X), 0, None) ** 2 * m.smear
        up = c_dmg * p * mag
        rf = np.clip(m.res.predict(X), 0, 1)
        dn = np.minimum(c_res * stock * rf, stock + up)
        P[:, t] = np.clip(stock + up - dn, 0, 1)
        UP[:, t], DN[:, t] = up, dn
        cum = cum + up
        hrs = np.where(up > 1e-9, 0.0, hrs + 1)
        uh = np.roll(uh, -1, 1); uh[:, -1] = up
        dh = np.roll(dh, -1, 1); dh[:, -1] = dn
    return fips, P, UP, DN


def main():
    tr = load_train()
    truth = osi_trajectory(tr)
    sel = np.isin(np.arange(216), PRED_HOURS)
    base_parts, corr_parts, coefs = [], [], []
    for k, (tr_f, va_f) in enumerate(make_folds(tr), 1):
        train = tr[tr.fipsCode.isin(tr_f)]
        val = mask_after_freeze(tr[tr.fipsCode.isin(va_f)])
        m = StockFlowModel(monotone=False).fit(train, dagger_rounds=1)

        # --- fit the two scalars on the OUTER-TRAINING counties only
        _, Ptr, UPtr, DNtr = rollout(m, mask_after_freeze(train))
        Ttr = np.nan_to_num(train.pivot_table(index="fipsCode", columns="hour",
                                              values="P_t").sort_index()
                            .reindex(columns=np.arange(216)).to_numpy())
        dT = np.diff(Ttr, axis=1, prepend=Ttr[:, :1])
        c_d = np.clip(np.clip(dT, 0, None)[:, 72:].sum() / max(UPtr[:, 72:].sum(), 1e-9), 0.5, 2.0)
        c_r = np.clip(np.clip(-dT, 0, None)[:, 72:].sum() / max(DNtr[:, 72:].sum(), 1e-9), 0.5, 2.0)
        coefs.append((c_d, c_r))
        print(f"  fold {k}: c_damage={c_d:.3f}  c_restore={c_r:.3f}", flush=True)

        for cd, cr, store in ((1.0, 1.0, base_parts), (c_d, c_r, corr_parts)):
            fips, P, UP, DN = rollout(m, val, cd, cr)
            osi = reconstruct_osi(P, UP, DN)
            store.append(pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                                       "hour": np.tile(PRED_HOURS, len(fips)),
                                       "osi_pred": osi[:, sel].ravel()}))

    for lab, parts, slug in (("P1 uncorrected", base_parts, "r6base"),
                             ("P1 + R6 flow correction", corr_parts, "r6corr")):
        d = pd.concat(parts, ignore_index=True)
        d.to_csv(f"oof/{slug}.csv", index=False)
        t = score_trajectory(d, truth)
        print(f"{lab:26s} RMSE {t.loc['mean','rmse']:.5f}  MAE {t.loc['mean','mae']:.5f}")
    cs = np.array(coefs)
    print(f"\ncoefficient stability: c_damage {cs[:,0].mean():.3f} +/- {cs[:,0].std():.3f}"
          f" | c_restore {cs[:,1].mean():.3f} +/- {cs[:,1].std():.3f}")


if __name__ == "__main__":
    main()
