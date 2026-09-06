"""Strictly nested cross-fitted distillation.

The quick version used cached out-of-fold teacher predictions. Those come from models
that never saw the county being taught, but DID see the outer validation fold — an
indirect path by which validation information can reach the student. It measured a
gain (0.00892 vs 0.00910), so it has to be re-tested with teachers that are blind to
the outer fold entirely.

Here, for each outer fold k:
  * take the counties outside fold k,
  * split them into 3 inner folds,
  * train the teacher on inner-train and predict inner-val,
so every teacher prediction used to train the student comes from a model that never
saw fold k in any capacity. The student is then scored on fold k as usual.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

from src.data import load_train, osi_trajectory, mask_after_freeze, PRED_HOURS
from src.validate import make_folds, score_trajectory
from src.dl import Y_SCALE
from src.ideas import _prep, _dl_out
from src.ideas2 import sqrt_xgb_forecaster
from src.ideas3 import BiEncSeq2Seq, _fit

ALPHA = 0.3          # teacher weight, exactly as specified; not tuned
SEED = 42


def teacher_predictions(sub_train, targets):
    """Teacher = mean of a bi-encoder and a sqrt-XGBoost, trained on `sub_train`,
    predicting the counties in `targets` (both are full frames)."""
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(sub_train, mask_after_freeze(targets))
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    m = _fit(BiEncSeq2Seq, Xe, Xd, Xs, Y, seed=0)
    with torch.no_grad():
        seq = np.clip(m(*tv).numpy() / Y_SCALE, 0, None)
    seq = pd.DataFrame(seq, index=vf, columns=PRED_HOURS).stack().rename("a").reset_index()
    seq.columns = ["fipsCode", "hour", "a"]
    tab = sqrt_xgb_forecaster(sub_train, mask_after_freeze(targets)).rename(
        columns={"osi_pred": "b"})
    both = seq.merge(tab, on=["fipsCode", "hour"], how="inner")
    both["teacher"] = 0.5 * both["a"] + 0.5 * both["b"]
    return both[["fipsCode", "hour", "teacher"]]


def main():
    train = load_train()
    truth = osi_trajectory(train)
    outer = list(make_folds(train))
    preds = []
    for k, (tr_f, va_f) in enumerate(outer):
        outer_tr = train[train.fipsCode.isin(tr_f)]
        counties = np.array(sorted(outer_tr.fipsCode.unique()))
        # ---- teacher predictions for the outer-training counties, blind to fold k
        parts = []
        for itr, ite in KFold(3, shuffle=True, random_state=SEED).split(counties):
            parts.append(teacher_predictions(
                outer_tr[outer_tr.fipsCode.isin(counties[itr])],
                outer_tr[outer_tr.fipsCode.isin(counties[ite])]))
        T = pd.concat(parts)
        print(f"  fold {k+1}: teacher built for {T.fipsCode.nunique()} counties", flush=True)

        # ---- student trains on the outer-training counties with the distillation loss
        val = mask_after_freeze(train[train.fipsCode.isin(va_f)])
        (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(outer_tr, val)
        fips = np.sort(outer_tr.fipsCode.unique())
        Tm = (T.pivot_table(index="fipsCode", columns="hour", values="teacher")
              .reindex(index=fips, columns=PRED_HOURS).to_numpy() * Y_SCALE)
        t = [torch.tensor(v, dtype=torch.float32)
             for v in (Xe, Xd, Xs, np.nan_to_num(Y), np.nan_to_num(Tm))]
        tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
        fold_pred = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed)
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(Xe))
            nv = max(12, len(idx) // 8)
            va_i, tr_i = idx[:nv], idx[nv:]
            model = BiEncSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
            opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
            best, state, bad = np.inf, None, 0
            for _ in range(300):
                model.train()
                for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                    opt.zero_grad()
                    o = model(t[0][b], t[1][b], t[2][b])
                    ((1 - ALPHA) * nn.functional.mse_loss(o, t[3][b])
                     + ALPHA * nn.functional.mse_loss(o, t[4][b])).backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    v = nn.functional.mse_loss(
                        model(t[0][va_i], t[1][va_i], t[2][va_i]), t[3][va_i]).item()
                if v < best - 1e-6:
                    best, state, bad = v, {q: p.clone() for q, p in model.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 30:
                        break
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                fold_pred.append(model(*tv).numpy())
        preds.append(_dl_out(vf, np.mean(fold_pred, 0)))

    out = pd.concat(preds, ignore_index=True)
    out.to_csv("oof/s64strict.csv", index=False)
    t = score_trajectory(out, truth)
    print(f"\nSTRICT nested distillation : RMSE {t.loc['mean','rmse']:.5f}  "
          f"MAE {t.loc['mean','mae']:.5f}")
    print(f"quick (cached-teacher) version: RMSE 0.00892")
    print(f"plain bi-encoder student      : RMSE 0.00910")


if __name__ == "__main__":
    main()
