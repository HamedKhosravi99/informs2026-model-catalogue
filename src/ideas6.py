"""Sweep 6: run under the empirical law established by sweeps 1-5 —
raise effective sample size or reduce variance; do not add parameters,
specialise, or widen sequence inputs.

  S60  factorial: does static + sub-county information stack on the best tabular model?
  S61  spatially resolved storm-exposure summaries -> trees only
  S62  combining the three sequence mechanisms that independently worked
  S63  more legitimate training origins (3 -> 5 -> 8)
  S64  cross-fitted ensemble distillation
  S65  nested additive tail-residual model
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .dl import Y_SCALE
from .features import build_rows, feature_cols
from .ideas import _out, _rows_and_y, _val_rows, add_static, add_physics
from .models import SEED
from .subcounty import load_subcounty

MPH = 1.609344          # the fetched archive reports km/h


# ------------------------------------------------------------------ S60 factorial
def _rows(df, statics, sub, val=False, spatial=False):
    if val:
        rows = _val_rows(df, static=statics, physics=statics)
        y = None
    else:
        rows, y = _rows_and_y(df, static=statics, physics=statics)
    if sub:
        rows = _add_sub(rows, spatial=spatial)
    return rows if val else (rows, y)


def _add_sub(rows, spatial=False):
    """Within-county gust spread; with spatial=True also the area-fraction features."""
    sc = load_subcounty()
    if sc is False:
        return rows
    idx = pd.MultiIndex.from_frame(rows[["fipsCode", "hour"]])
    gmax_ratio = sc["gmax_ratio"].reindex(idx).to_numpy()
    gstd_ratio = sc["gstd_ratio"].reindex(idx).to_numpy()
    gmax_abs = sc["gmax_abs"].reindex(idx).to_numpy() / MPH        # -> mph
    rows["gmax_ratio"] = gmax_ratio
    rows["gp90_ratio"] = sc["gp90_ratio"].reindex(idx).to_numpy()
    rows["gstd_ratio"] = gstd_ratio
    rows["local_peak_gust"] = gmax_abs
    if "gust" in rows.columns:
        rows["local_minus_mean"] = gmax_abs - rows["gust"]
    if not spatial:
        return rows

    # reconstruct the within-county gust distribution from the stored summaries
    gmean = gmax_abs / np.maximum(gmax_ratio, 1e-6)
    gstd = np.maximum(gstd_ratio * gmean, 1e-6)
    from scipy.stats import norm
    for thr in (40, 50, 60, 70):
        rows[f"area_frac_gt{thr}"] = 1.0 - norm.cdf((thr - gmean) / gstd)
    rows["gust_mean_sub"] = gmean
    rows["gust_std_sub"] = gstd
    rows["excess40_local"] = np.clip(gmax_abs - 40, 0, None)
    rows["excess50_local"] = np.clip(gmax_abs - 50, 0, None)

    # trajectory summaries over the local-peak series (weather only, any timestamp)
    d = rows[["fipsCode", "hour"]].copy()
    d["lp"] = gmax_abs
    d["a50"] = rows["area_frac_gt50"]
    d = d.sort_values(["fipsCode", "hour"])
    g = d.groupby("fipsCode")
    for w in (6, 24, 48):
        d[f"lp_max{w}"] = g["lp"].transform(lambda s: s.rolling(w, min_periods=1).max())
        d[f"a50_sum{w}"] = g["a50"].transform(lambda s: s.rolling(w, min_periods=1).sum())
    d["lp_cum_ex40"] = g["lp"].transform(lambda s: np.clip(s - 40, 0, None).cumsum())
    d["a50_cum"] = g["a50"].transform("cumsum")
    d = d.set_index(["fipsCode", "hour"])
    for c in [c for c in d.columns if c not in ("lp", "a50")]:
        rows[c] = d[c].reindex(idx).to_numpy()
    return rows


def _xgb_sqrt(train_full, val_masked, statics, sub, spatial=False, model="xgb"):
    tr, y = _rows(train_full, statics, sub, spatial=spatial)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    if model == "xgb":
        import xgboost as xgb
        mdl = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                               subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                               reg_lambda=1.0, random_state=SEED, n_jobs=4,
                               tree_method="hist")
    else:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        mdl = make_pipeline(SimpleImputer(strategy="median"),
                            ExtraTreesRegressor(n_estimators=400, min_samples_leaf=5,
                                                max_features=0.5, n_jobs=4,
                                                random_state=SEED))
    mdl.fit(tr.loc[keep, cols], np.sqrt(y[keep]))
    va = _rows(val_masked, statics, sub, val=True, spatial=spatial)
    return _out(va, np.clip(mdl.predict(va[cols]), 0, None) ** 2)


def f_base(tr, va):      return _xgb_sqrt(tr, va, False, False)
def f_stat(tr, va):      return _xgb_sqrt(tr, va, True, False)
def f_sub(tr, va):       return _xgb_sqrt(tr, va, False, True)
def f_both(tr, va):      return _xgb_sqrt(tr, va, True, True)
def f_spatial(tr, va):   return _xgb_sqrt(tr, va, True, True, spatial=True)
def f_spatial_et(tr, va): return _xgb_sqrt(tr, va, True, True, spatial=True, model="et")


# ============================ S62 combining the sequence mechanisms that worked
def _fit_scoring_dual(cls, Xe, Xd, Xs, Y, seed, w, max_epochs=300, patience=30):
    """Scoring-aligned per-hour weights AND the dual raw+sqrt objective."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = cls(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))

    def loss_fn(out, i):
        raw = (((torch.clamp(out, min=0) ** 2) - t[3][i]) ** 2 * tw * msk[i]).sum() / (msk[i] * tw).sum()
        root = ((out - sq[i]) ** 2 * tw * msk[i]).sum() / (msk[i] * tw).sum()
        return 0.5 * raw + 0.5 * root

    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            loss_fn(model(t[0][b], t[1][b], t[2][b]), b).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.clamp(model(t[0][va_i], t[1][va_i], t[2][va_i]), min=0) ** 2
            v = float((((p - t[3][va_i]) ** 2) * tw * msk[va_i]).sum() / (msk[va_i] * tw).sum())
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def bienc_scoring_dual_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    from .ideas import _prep, _dl_out
    from .ideas3 import BiEncSeq2Seq
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    w = scoring_weights()
    preds = []
    for s in seeds:
        m = _fit_scoring_dual(BiEncSeq2Seq, Xe, Xd, Xs, Y, s, w)
        with torch.no_grad():
            preds.append((torch.clamp(m(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ==================================================== S63 more training origins
def multi_origin_5(train_full, val_masked):
    from .ideas4 import multi_origin_forecaster
    return multi_origin_forecaster(train_full, val_masked, origins=(39, 47, 55, 63, 71))


def multi_origin_8(train_full, val_masked):
    from .ideas4 import multi_origin_forecaster
    return multi_origin_forecaster(train_full, val_masked,
                                   origins=(23, 31, 39, 47, 55, 63, 71, 79))


IDEAS6 = {
    "S60 sqrt-XGB base (no statics, no sub)": f_base,
    "S60b sqrt-XGB + statics": f_stat,
    "S60c sqrt-XGB + sub-county": f_sub,
    "S60d sqrt-XGB + statics + sub-county": f_both,
    "S61 + spatial storm-exposure summaries": f_spatial,
    "S61b ExtraTrees + all tabular info": f_spatial_et,
    "S62 BiGRU + scoring + dual loss": bienc_scoring_dual_forecaster,
    "S63 multi-origin, 5 origins": multi_origin_5,
    "S63b multi-origin, 8 origins": multi_origin_8,
}


def multi_origin_storm5(train_full, val_masked):
    """Five origins, all inside the storm (GPT's instruction: freeze points should
    resemble the deployment state, which sits at the wave-1 peak). The earlier
    5-origin set included pre-event hours 39/47 and scored 0.01016."""
    from .ideas4 import multi_origin_forecaster
    return multi_origin_forecaster(train_full, val_masked, origins=(55, 59, 63, 67, 71))


IDEAS6["S63c multi-origin, 5 storm-phase origins"] = multi_origin_storm5


# ==================================== S64 cross-fitted ensemble distillation
def distill_forecaster(train_full, val_masked, seeds=(0, 1, 2), alpha=0.3):
    """Student = plain bi-encoder; teacher = the cached out-of-fold ensemble.

    L = 0.7*MSE(student, truth) + 0.3*MSE(student, teacher).

    The teacher is cross-fitted in the sense that each county's teacher trajectory
    comes from members that never trained on that county. Caveat recorded in the
    report: those members did see the OUTER validation fold, so a weak indirect path
    exists; if this shows a gain it must be re-run with teachers regenerated inside
    each outer fold before being believed.
    """
    from .ideas import _prep, _dl_out
    from .ideas3 import BiEncSeq2Seq
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    fips = np.sort(train_full["fipsCode"].unique())
    T = (pd.read_csv("oof/zC.csv").pivot_table(index="fipsCode", columns="hour",
                                               values="osi_pred")
         .reindex(index=fips, columns=PRED_HOURS).to_numpy() * Y_SCALE)
    t = [torch.tensor(v, dtype=torch.float32)
         for v in (Xe, Xd, Xs, np.nan_to_num(Y), np.nan_to_num(T))]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = BiEncSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                o = model(t[0][b], t[1][b], t[2][b])
                loss = ((1 - alpha) * nn.functional.mse_loss(o, t[3][b])
                        + alpha * nn.functional.mse_loss(o, t[4][b]))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():                       # early stop on TRUTH only
                v = nn.functional.mse_loss(
                    model(t[0][va_i], t[1][va_i], t[2][va_i]), t[3][va_i]).item()
            if v < best - 1e-6:
                best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


IDEAS6["S64 cross-fitted ensemble distillation"] = distill_forecaster
