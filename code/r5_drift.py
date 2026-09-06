"""R5 — mandatory drift decomposition for the P1 stock-flow model.

Determines the SIGN and SOURCE of rollout drift before anything is fitted to correct
it: chronic damage over/under-prediction, restoration over/under-prediction, or
state-feedback interaction, resolved by rollout horizon and by storm phase.
"""
import numpy as np
import pandas as pd

from src.data import FREEZE_H, load_train, mask_after_freeze
from src.validate import make_folds
from src.stockflow import (StockFlowModel, _weather_matrices, _static_matrix,
                           _state_block, WCOLS)

BANDS = [("h73-95 wave-1 decay", 73, 96), ("h96-119 lull", 96, 120),
         ("h120-167 wave 2", 120, 168), ("h168-215 tail", 168, 216)]


def main():
    tr = load_train()
    rows = []
    for k, (tr_f, va_f) in enumerate(make_folds(tr), 1):
        train = tr[tr.fipsCode.isin(tr_f)]
        val = tr[tr.fipsCode.isin(va_f)]
        m = StockFlowModel(monotone=False).fit(train, dagger_rounds=1)
        fips, Pc, UPc, DNc = m.predict(mask_after_freeze(val))

        Pt = (val.pivot_table(index="fipsCode", columns="hour", values="P_t")
              .sort_index().reindex(columns=np.arange(216)).to_numpy())
        Pt = np.nan_to_num(Pt)
        dP = np.diff(Pt, axis=1, prepend=Pt[:, :1])
        upT, dnT = np.clip(dP, 0, None), np.clip(-dP, 0, None)

        # teacher-forced pass: true state in, predicted flows out
        W = _weather_matrices(val)
        S, _, _ = _static_matrix(mask_after_freeze(val))
        n = len(Pt)
        UPf = np.zeros_like(Pt); DNf = np.zeros_like(Pt); Pf = Pt.copy()
        cum = upT[:, :FREEZE_H + 1].sum(1)
        hrs = np.full(n, 99.0)
        uh = upT[:, FREEZE_H - 5:FREEZE_H + 1].copy()
        dh = dnT[:, FREEZE_H - 5:FREEZE_H + 1].copy()
        for t in range(FREEZE_H + 1, 216):
            stock = Pt[:, t - 1]
            X = np.hstack([_state_block(stock, uh, dh, cum, hrs, W["gust"][:, t]),
                           np.column_stack([W[c][:, t] for c in WCOLS]), S])
            p = m.clf.predict_proba(X)[:, 1]
            mag = np.clip(m.reg.predict(X), 0, None) ** 2 * m.smear
            up = p * mag
            rf = np.clip(m.res.predict(X), 0, 1)
            dn = np.minimum(stock * rf, stock + up)
            UPf[:, t], DNf[:, t] = up, dn
            Pf[:, t] = np.clip(stock + up - dn, 0, 1)
            cum = cum + upT[:, t]; hrs = np.where(upT[:, t] > 0, 0.0, hrs + 1)
            uh = np.roll(uh, -1, 1); uh[:, -1] = upT[:, t]
            dh = np.roll(dh, -1, 1); dh[:, -1] = dnT[:, t]
        rows.append((Pc, Pt, UPc, DNc, upT, dnT, UPf, DNf, Pf))

    Pc, Pt, UPc, DNc, upT, dnT, UPf, DNf, Pf = [np.vstack([r[i] for r in rows])
                                                for i in range(9)]

    print("=== 1. STATE ERROR by rollout horizon (closed loop) ===")
    print(f"  {'horizon':>10s} {'RMSE(Q)':>10s} {'mean bias':>11s}")
    for h in (1, 6, 12, 24, 48, 144):
        t = min(FREEZE_H + h, 215)
        e = Pc[:, t] - Pt[:, t]
        print(f"  {h:>8d}h {np.sqrt((e**2).mean()):>10.5f} {e.mean():>+11.6f}")

    print("\n=== 2. CUMULATIVE FLOW BIAS, closed loop vs teacher forced ===")
    print(f"  {'band':22s} {'dmg CL':>8s} {'dmg TF':>8s} {'res CL':>8s} {'res TF':>8s}"
          f" {'stock bias':>11s}")
    for lab, lo, hi in BANDS:
        dc = UPc[:, lo:hi].sum(); df = UPf[:, lo:hi].sum(); dt = upT[:, lo:hi].sum()
        rc = DNc[:, lo:hi].sum(); rf_ = DNf[:, lo:hi].sum(); rt = dnT[:, lo:hi].sum()
        sb = (Pc[:, lo:hi] - Pt[:, lo:hi]).mean()
        print(f"  {lab:22s} {dc/max(dt,1e-9):>8.2f} {df/max(dt,1e-9):>8.2f}"
              f" {rc/max(rt,1e-9):>8.2f} {rf_/max(rt,1e-9):>8.2f} {sb:>+11.6f}")
    print("  (ratios are predicted/true; 1.00 = calibrated)")

    print("\n=== 3. VERDICT ===")
    d_cl = UPc[:, 72:].sum() / max(upT[:, 72:].sum(), 1e-9)
    d_tf = UPf[:, 72:].sum() / max(upT[:, 72:].sum(), 1e-9)
    r_cl = DNc[:, 72:].sum() / max(dnT[:, 72:].sum(), 1e-9)
    r_tf = DNf[:, 72:].sum() / max(dnT[:, 72:].sum(), 1e-9)
    print(f"  damage      : closed-loop {d_cl:.2f}x   teacher-forced {d_tf:.2f}x")
    print(f"  restoration : closed-loop {r_cl:.2f}x   teacher-forced {r_tf:.2f}x")
    print(f"  net stock bias over the whole window: {(Pc[:,72:]-Pt[:,72:]).mean():+.6f}")
    if d_tf < 1.05 and d_cl > 1.3:
        print("  -> (E) STATE-FEEDBACK: flows are calibrated on true states and inflate")
        print("        only in closed loop. The fault is the feedback path, not the")
        print("        flow models -- so rollout-aware training (R2/R3/R4) is indicated,")
        print("        and a static flow correction (R6) would be treating a symptom.")
    elif d_cl > 1.3:
        print("  -> (A) chronic damage over-prediction, present even teacher-forced")
    else:
        print("  -> see the per-band table; no single dominant mode")


if __name__ == "__main__":
    main()
