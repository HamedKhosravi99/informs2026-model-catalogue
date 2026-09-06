"""The 20 candidate ideas, each as a forecaster evaluated under the identical
5-fold county-holdout protocol (see src/validate.py).

Every forecaster has signature f(train_full, val_masked) -> DataFrame
(fipsCode, hour, osi_pred) over hours 73-215. `val_masked` never contains
outage data after FREEZE_H, so causality holds structurally.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import ROOT, FREEZE_H, PRED_HOURS, mask_after_freeze, osi_trajectory
from .features import build_rows, feature_cols, county_static, weather_by_hour
from .models import make_gbm, decay_forecaster, HURDLE_THETA, SEED
from .dl import (Y_SCALE, ENC_COLS, DEC_COLS, STATIC_COLS, Seq2Seq, _fit_one,
                 _tensors, _pivot, _seq_tensor, gru_forecaster)

# ---------------------------------------------------------------- shared helpers
def _rows_and_y(df, static=False, physics=False):
    rows = build_rows(mask_after_freeze(df))
    if static:
        rows = add_static(rows)
    if physics:
        rows = add_physics(rows, df)
    y = df.set_index(["fipsCode", "hour"])["osi"].reindex(
        pd.MultiIndex.from_frame(rows[["fipsCode", "hour"]])).to_numpy()
    return rows, y


def _val_rows(va, static=False, physics=False):
    rows = build_rows(va)
    if static:
        rows = add_static(rows)
    if physics:
        rows = add_physics(rows, va)
    return rows


def _out(rows, pred):
    o = rows[["fipsCode", "hour"]].copy()
    o["osi_pred"] = np.clip(pred, 0.0, None)
    return o


def _traj(df):
    """County x hour matrix of OSI over the scored window."""
    return (df.pivot_table(index="fipsCode", columns="hour", values="osi")
            .sort_index().reindex(columns=PRED_HOURS))


def _grid(val_masked):
    fips = np.sort(val_masked["fipsCode"].unique())
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips))}), fips


# ============================================================ I16 static enrichment
_STATIC_CACHE = None


def load_static():
    """Public county attributes: USDA rural-urban code, Census population /
    land area / centroid. Legal (not outage data; known for test counties)."""
    global _STATIC_CACHE
    if _STATIC_CACHE is None:
        s = pd.read_csv(ROOT / "data_static" / "county_static.csv").set_index("fipsCode")
        s["log_pop"] = np.log1p(s["population"])
        s["log_density"] = np.log1p(s["pop_density"])
        s["log_area"] = np.log(s["land_sqmi"])
        _STATIC_CACHE = s[["log_pop", "log_density", "log_area", "rucc", "lat", "lon"]]
    return _STATIC_CACHE


def add_static(rows):
    s = load_static()
    rows = rows.join(s, on="fipsCode")
    # customers per square mile ~ service-density / line-length proxy (EIA-861 stand-in)
    rows["cust_per_sqmi"] = np.expm1(rows["log_customers"]) / np.exp(rows["log_area"])
    rows["cust_per_capita"] = np.expm1(rows["log_customers"]) / np.expm1(rows["log_pop"])
    return rows


def static_enrichment_forecaster(train_full, val_masked):
    tr, y = _rows_and_y(train_full, static=True)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_gbm().fit(tr.loc[keep, cols], y[keep])
    va = _val_rows(val_masked, static=True)
    return _out(va, mdl.predict(va[cols]))


# ============================================================ I17 physics indices
def add_physics(rows, src_df):
    """Domain indices: icing, turbulence, wind run, frontal passage, wind power."""
    w = weather_by_hour(src_df).set_index(["fipsCode", "hour"])
    idx = pd.MultiIndex.from_frame(rows[["fipsCode", "hour"]])
    t2m, r2 = w["t2m"].reindex(idx).to_numpy(), w["r2"].reindex(idx).to_numpy()
    tp, gust = w["tp"].reindex(idx).to_numpy(), w["gust"].reindex(idx).to_numpy()
    wind = w["wind_speed_10m"].reindex(idx).to_numpy()
    tc = t2m - 273.15
    rows["icing_idx"] = np.where((tc > -3) & (tc < 3) & (r2 > 85), tp * (1 + gust / 20.0), 0.0)
    rows["gust_factor"] = gust / np.maximum(wind, 1.0)           # turbulence proxy
    rows["wind_power"] = wind ** 3 / 1e3                          # ~ kinetic flux
    rows["gust_power"] = gust ** 3 / 1e3
    rows["soil_x_gust"] = w["soil_moist"].reindex(idx).to_numpy() * gust  # wind-throw risk
    src = src_df.sort_values(["fipsCode", "hour"])
    g = src.groupby("fipsCode")
    dirchg = (g["wind_dir_10m"].diff().abs() % 360).clip(upper=180)
    prs = g["mslma"].diff()
    aux = pd.DataFrame({"fipsCode": src["fipsCode"].to_numpy(), "hour": src["hour"].to_numpy(),
                        "dir_change": dirchg.to_numpy(), "pres_drop": (-prs).clip(lower=0).to_numpy(),
                        "wind_run": g["wind_speed_10m"].cumsum().to_numpy()}
                       ).set_index(["fipsCode", "hour"])
    for c in aux.columns:
        rows[c] = aux[c].reindex(idx).to_numpy()
    return rows


def physics_indices_forecaster(train_full, val_masked):
    tr, y = _rows_and_y(train_full, physics=True)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_gbm().fit(tr.loc[keep, cols], y[keep])
    va = _val_rows(val_masked, physics=True)
    return _out(va, mdl.predict(va[cols]))


def static_physics_forecaster(train_full, val_masked):
    """Both enrichments together (the 'all information' tabular model)."""
    tr, y = _rows_and_y(train_full, static=True, physics=True)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_gbm().fit(tr.loc[keep, cols], y[keep])
    va = _val_rows(val_masked, static=True, physics=True)
    return _out(va, mdl.predict(va[cols]))


# ================================================= I01 amplitude x shape decomposition
def amplitude_shape_forecaster(train_full, val_masked):
    T = _traj(train_full)
    amp = np.log1p(T.sum(1) * 10.0)                     # county event "size"
    st_tr = add_static(county_static(mask_after_freeze(train_full)).reset_index()).set_index("fipsCode")
    feats = [c for c in st_tr.columns if c != "fipsCode"]
    mdl = make_gbm(max_iter=300).fit(st_tr.loc[amp.index, feats], amp.loc[amp.index])

    shape = T.div(T.sum(1) + 1e-9, axis=0)              # normalized trajectory
    states = train_full.groupby("fipsCode")["stateAbbr"].first()
    shape_by_state = shape.groupby(states).mean()
    shape_global = shape.mean(0)

    st_va = add_static(county_static(val_masked).reset_index()).set_index("fipsCode")
    amp_hat = np.expm1(mdl.predict(st_va[feats])) / 10.0
    va_states = val_masked.groupby("fipsCode")["stateAbbr"].first()
    g, fips = _grid(val_masked)
    S = np.stack([shape_by_state.loc[va_states[f]].to_numpy()
                  if va_states[f] in shape_by_state.index else shape_global.to_numpy()
                  for f in fips])
    A = pd.Series(amp_hat, index=st_va.index).reindex(fips).to_numpy()[:, None]
    g["osi_pred"] = np.clip(A * S, 0, None).ravel()
    return g


# ================================================== I02 impulse-response (system ID)
def impulse_response_forecaster(train_full, val_masked, taus=(4, 8, 12, 18, 24, 36, 48)):
    """Stage 1: weather -> damage impulses N_t. Stage 2: convolve with a
    recovery kernel and add decaying initial state; kernel/weights fit on train."""
    tr, _ = _rows_and_y(train_full)
    cols = feature_cols(tr)
    yN = train_full.set_index(["fipsCode", "hour"])["N_t"].reindex(
        pd.MultiIndex.from_frame(tr[["fipsCode", "hour"]])).to_numpy()
    keep = np.isfinite(yN)
    imp_mdl = make_gbm().fit(tr.loc[keep, cols], yN[keep])

    def _basis(rows, masked_src):
        N = _out(rows, imp_mdl.predict(rows[cols])).pivot_table(
            index="fipsCode", columns="hour", values="osi_pred").reindex(columns=PRED_HOURS)
        o71 = masked_src[masked_src["hour"] == FREEZE_H].set_index("fipsCode")["osi"].reindex(N.index)
        dt = (PRED_HOURS - FREEZE_H).astype(float)
        out = {}
        for tau in taus:
            k = np.exp(-np.arange(len(PRED_HOURS)) / tau)
            conv = np.stack([np.convolve(row, k)[:len(PRED_HOURS)] for row in N.to_numpy()])
            out[f"conv{tau}"] = conv
            out[f"decay{tau}"] = o71.to_numpy()[:, None] * np.exp(-dt / tau)[None, :]
        return N.index, out

    fips_tr, B_tr = _basis(tr, mask_after_freeze(train_full))
    Ytr = _traj(train_full).reindex(fips_tr).to_numpy()
    X = np.stack([v.ravel() for v in B_tr.values()], 1)
    m = np.isfinite(Ytr.ravel())
    coef = np.linalg.lstsq(X[m], Ytr.ravel()[m], rcond=None)[0]

    va = _val_rows(val_masked)
    fips_va, B_va = _basis(va, val_masked)
    Xv = np.stack([v.ravel() for v in B_va.values()], 1)
    P = np.clip(Xv @ coef, 0, None).reshape(len(fips_va), len(PRED_HOURS))
    return pd.DataFrame({"fipsCode": np.repeat(fips_va.to_numpy(), len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips_va)),
                         "osi_pred": P.ravel()})


# ================================================================ I03 FPCA two-stage
def fpca_forecaster(train_full, val_masked, k=5):
    T = _traj(train_full).fillna(0.0)
    mu = T.mean(0).to_numpy()
    U, S, Vt = np.linalg.svd(T.to_numpy() - mu, full_matrices=False)
    scores = (T.to_numpy() - mu) @ Vt[:k].T                       # counties x k

    st_tr = add_static(county_static(mask_after_freeze(train_full)).reset_index()).set_index("fipsCode")
    feats = [c for c in st_tr.columns if c != "fipsCode"]
    mdls = [make_gbm(max_iter=300).fit(st_tr.loc[T.index, feats], scores[:, j]) for j in range(k)]

    st_va = add_static(county_static(val_masked).reset_index()).set_index("fipsCode")
    Sv = np.stack([m.predict(st_va[feats]) for m in mdls], 1)
    P = np.clip(mu + Sv @ Vt[:k], 0, None)
    fips = st_va.index.to_numpy()
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips)),
                         "osi_pred": P.ravel()})


# ========================================================= I04 wave-conditioned cascade
WAVE_SPLIT = 110


def wave_cascade_forecaster(train_full, val_masked):
    """Stage 1 forecasts the wave-1 decay window; its aggregate becomes a
    depletion feature for the stage-2 wave-2 model."""
    tr, y = _rows_and_y(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    m1 = keep & (tr["hour"] <= WAVE_SPLIT).to_numpy()
    mdl1 = make_gbm().fit(tr.loc[m1, cols], y[m1])

    def _stage2_feats(rows):
        p1 = np.clip(mdl1.predict(rows[cols]), 0, None)
        d = rows[["fipsCode", "hour"]].copy()
        d["p"] = p1
        agg = d[d["hour"] <= WAVE_SPLIT].groupby("fipsCode")["p"].agg(["mean", "max", "sum"])
        agg.columns = ["w1_mean", "w1_max", "w1_sum"]
        return rows.join(agg, on="fipsCode")

    tr2 = _stage2_feats(tr)
    cols2 = feature_cols(tr2)
    m2 = keep & (tr["hour"] > WAVE_SPLIT).to_numpy()
    mdl2 = make_gbm().fit(tr2.loc[m2, cols2], y[m2])

    va = _stage2_feats(_val_rows(val_masked))
    pred = np.where(va["hour"] <= WAVE_SPLIT,
                    mdl1.predict(va[cols]), mdl2.predict(va[cols2]))
    return _out(va, pred)


# ======================================================== I05 trajectory-archetype gate
def _archetype_labels(train_full, k=4):
    T = _traj(train_full).fillna(0.0)
    norm = T.div(T.max(1).replace(0, np.nan), axis=0).fillna(0.0)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(norm.to_numpy())
    return pd.Series(km.labels_, index=T.index)


def _moe(train_full, val_masked, labels, k, oracle_labels=None):
    tr, y = _rows_and_y(train_full, static=True)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    row_lab = tr["fipsCode"].map(labels)
    experts = {}
    for c in range(k):
        m = keep & (row_lab == c).to_numpy()
        experts[c] = make_gbm().fit(tr.loc[m, cols], y[m]) if m.sum() > 500 else None
    st_tr = add_static(county_static(mask_after_freeze(train_full)).reset_index()).set_index("fipsCode")
    feats = [c for c in st_tr.columns if c != "fipsCode"]
    gate = HistGradientBoostingClassifier(
        learning_rate=0.08, max_iter=300, max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=1.0, random_state=SEED).fit(
            st_tr.loc[labels.index, feats], labels.to_numpy())

    va = _val_rows(val_masked, static=True)
    st_va = add_static(county_static(val_masked).reset_index()).set_index("fipsCode")
    valid = [c for c in range(k) if experts[c] is not None]
    P = np.stack([np.clip(experts[c].predict(va[cols]), 0, None) for c in valid], 1)
    if oracle_labels is not None:
        W = np.zeros((len(va), len(valid)))
        lab = va["fipsCode"].map(oracle_labels).to_numpy()
        for j, c in enumerate(valid):
            W[:, j] = (lab == c).astype(float)
        W = W / np.maximum(W.sum(1, keepdims=True), 1e-9)
    else:
        proba = gate.predict_proba(st_va[feats])
        classes = list(gate.classes_)
        keepc = [classes.index(c) for c in valid]
        Pr = proba[:, keepc]
        Pr = Pr / Pr.sum(1, keepdims=True)
        W = Pr[pd.Index(st_va.index).get_indexer(va["fipsCode"])]
    return _out(va, (P * W).sum(1))


def archetype_gate_forecaster(train_full, val_masked, k=4):
    labels = _archetype_labels(train_full, k)
    return _moe(train_full, val_masked, labels, k)


def archetype_oracle_forecaster(train_full, val_masked, k=4):
    """Ceiling diagnostic: same experts, true archetype routing."""
    allc = pd.concat([train_full, val_masked.assign(osi=np.nan)]) if False else train_full
    labels = _archetype_labels(train_full, k)
    # oracle labels for val counties come from their true trajectories
    T = _traj(val_masked) if val_masked["osi"].notna().any() else None
    from .data import load_train
    full = load_train()
    Tv = _traj(full[full["fipsCode"].isin(val_masked["fipsCode"].unique())]).fillna(0.0)
    normv = Tv.div(Tv.max(1).replace(0, np.nan), axis=0).fillna(0.0)
    Tt = _traj(train_full).fillna(0.0)
    normt = Tt.div(Tt.max(1).replace(0, np.nan), axis=0).fillna(0.0)
    cent = normt.groupby(labels).mean()
    d = ((normv.to_numpy()[:, None, :] - cent.to_numpy()[None, :, :]) ** 2).sum(-1)
    oracle = pd.Series(cent.index.to_numpy()[d.argmin(1)], index=normv.index)
    return _moe(train_full, val_masked, labels, k, oracle_labels=oracle)


# ============================================================== I06 quantile two-stage
def quantile_forecaster(train_full, val_masked):
    tr, y = _rows_and_y(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    qs = {}
    for q in (0.1, 0.5, 0.9):
        qs[q] = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, learning_rate=0.06, max_iter=400,
            max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
            random_state=SEED).fit(tr.loc[keep, cols], y[keep])
    Qtr = np.stack([qs[q].predict(tr.loc[keep, cols]) for q in (0.1, 0.5, 0.9)], 1)
    comb = Ridge(alpha=1.0, positive=True).fit(Qtr, y[keep])
    va = _val_rows(val_masked)
    Qva = np.stack([qs[q].predict(va[cols]) for q in (0.1, 0.5, 0.9)], 1)
    return _out(va, comb.predict(Qva))


# =========================================================== I13 Tweedie-family GBM
def poisson_gbm_forecaster(train_full, val_masked):
    """LightGBM/Tweedie unavailable (libomp missing); sklearn's Poisson loss is
    the Tweedie-family (p=1) stand-in for the zero-inflated target."""
    tr, y = _rows_and_y(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    mdl = make_gbm(loss="poisson").fit(tr.loc[keep, cols], y[keep] * Y_SCALE)
    va = _val_rows(val_masked)
    return _out(va, mdl.predict(va[cols]) / Y_SCALE)


# ======================================================== I14 hierarchical pooling
def hierarchical_forecaster(train_full, val_masked, kappa=8000.0):
    """Partial pooling: per-state model shrunk toward the pooled model by
    n_s/(n_s+kappa) (rows per state)."""
    tr, y = _rows_and_y(train_full)
    cols = feature_cols(tr)
    keep = np.isfinite(y)
    pooled = make_gbm().fit(tr.loc[keep, cols], y[keep])
    states = train_full.groupby("fipsCode")["stateAbbr"].first()
    tr["st"] = tr["fipsCode"].map(states)
    per = {}
    n_s = {}
    for s in ["IN", "OH", "PA", "WV"]:
        m = keep & (tr["st"] == s).to_numpy()
        n_s[s] = int(m.sum())
        per[s] = make_gbm().fit(tr.loc[m, cols], y[m])
    va = _val_rows(val_masked)
    va["st"] = va["fipsCode"].map(val_masked.groupby("fipsCode")["stateAbbr"].first())
    p_pool = pooled.predict(va[cols])
    pred = p_pool.copy()
    for s in ["IN", "OH", "PA", "WV"]:
        m = (va["st"] == s).to_numpy()
        if m.sum():
            w = n_s[s] / (n_s[s] + kappa)
            pred[m] = w * per[s].predict(va.loc[m, cols]) + (1 - w) * p_pool[m]
    return _out(va, pred)


# ==================================================== I15 restoration survival model
def survival_restoration_forecaster(train_full, val_masked, taus=np.arange(3, 61, 3)):
    """Predict each county's restoration time-constant from its observed window,
    then decay the last observed state at that county-specific rate."""
    T = _traj(train_full)
    o71 = train_full[train_full["hour"] == FREEZE_H].set_index("fipsCode")["osi"].reindex(T.index)
    dt = (PRED_HOURS - FREEZE_H).astype(float)
    best = []
    for i in range(len(T)):
        curve = T.iloc[i].to_numpy()
        err = [np.nanmean((o71.iloc[i] * np.exp(-dt / tau) - curve) ** 2) for tau in taus]
        best.append(taus[int(np.argmin(err))])
    tgt = pd.Series(np.log(best), index=T.index)

    st_tr = add_static(county_static(mask_after_freeze(train_full)).reset_index()).set_index("fipsCode")
    feats = [c for c in st_tr.columns if c != "fipsCode"]
    mdl = make_gbm(max_iter=300).fit(st_tr.loc[tgt.index, feats], tgt)
    st_va = add_static(county_static(val_masked).reset_index()).set_index("fipsCode")
    tau_hat = np.clip(np.exp(mdl.predict(st_va[feats])), 3, 90)
    o71_va = val_masked[val_masked["hour"] == FREEZE_H].set_index("fipsCode")["osi"].reindex(st_va.index)
    P = o71_va.to_numpy()[:, None] * np.exp(-dt[None, :] / tau_hat[:, None])
    fips = st_va.index.to_numpy()
    return pd.DataFrame({"fipsCode": np.repeat(fips, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(fips)),
                         "osi_pred": np.clip(P, 0, None).ravel()})


# =============================================================== DL shared prep
def _prep(train_full, val_masked, dec_hours=PRED_HOURS):
    _, Xe, Xd, Xs = _tensors(mask_after_freeze(train_full), dec_hours)
    Y = (train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
         .sort_index().reindex(columns=dec_hours).to_numpy() * Y_SCALE)
    stats = []
    for a in (Xe, Xd, Xs):
        m = a.reshape(-1, a.shape[-1]).mean(0)
        s = a.reshape(-1, a.shape[-1]).std(0) + 1e-6
        stats.append((m, s))
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]
    vf, Ve, Vd, Vs = _tensors(val_masked, dec_hours)
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    return (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs)


def _dl_out(vf, P):
    return pd.DataFrame({"fipsCode": np.repeat(vf, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(vf)),
                         "osi_pred": np.clip(P / Y_SCALE, 0, None).ravel()})


# ==================================================== I07 cross-county attention
class CrossCountyNet(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, hidden=48, heads=4, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.dec = nn.GRU(n_dec + n_static + hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def encode(self, enc):
        _, h = self.enc(enc)
        return h[-1]

    def forward(self, enc, dec, static, mem_enc):
        e = self.encode(enc)                       # (B, H)
        mem = self.encode(mem_enc)                 # (M, H) — observed-window only
        ctx, _ = self.attn(e[:, None, :], mem[None].expand(len(e), -1, -1),
                           mem[None].expand(len(e), -1, -1))
        ctx = ctx.squeeze(1)
        s = torch.cat([static, ctx], -1)[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), e[None].contiguous())
        return self.head(self.drop(out)).squeeze(-1)


def cross_county_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    t = {k: torch.tensor(v, dtype=torch.float32)
         for k, v in dict(Xe=Xe, Xd=Xd, Xs=Xs, Y=Y, Ve=Ve, Vd=Vd, Vs=Vs).items()}
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = CrossCountyNet(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                mem = t["Xe"][rng.choice(tr_i, min(64, len(tr_i)), replace=False)]
                loss = nn.functional.mse_loss(
                    model(t["Xe"][b], t["Xd"][b], t["Xs"][b], mem), t["Y"][b])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(
                    model(t["Xe"][va_i], t["Xd"][va_i], t["Xs"][va_i], t["Xe"][tr_i]),
                    t["Y"][va_i]).item()
            if v < best - 1e-6:
                best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds.append(model(t["Ve"], t["Vd"], t["Vs"], t["Xe"]).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ================================================= I08 transductive weather pretraining
def transductive_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    """Self-supervised pretraining on ALL counties' weather (train + val — legal,
    weather is given for every county), then supervised fine-tuning on train."""
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    # pretext target: reconstruct gust channel of the decoder window from masked input
    gi = DEC_COLS.index("gust")
    allXd = np.concatenate([Xd, Vd], 0)
    allXe = np.concatenate([Xe, Ve], 0)
    allXs = np.concatenate([Xs, Vs], 0)
    pre_target = allXd[:, :, gi].copy()
    masked_dec = allXd.copy()
    rng0 = np.random.RandomState(SEED)
    mask = rng0.rand(*pre_target.shape) < 0.25
    masked_dec[:, :, gi][mask] = 0.0

    t = {k: torch.tensor(v, dtype=torch.float32) for k, v in dict(
        Xe=Xe, Xd=Xd, Xs=Xs, Y=Y, Ve=Ve, Vd=Vd, Vs=Vs,
        aXe=allXe, aXd=masked_dec, aXs=allXs, aY=pre_target).items()}
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        model = Seq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(40):                                    # pretrain
            model.train()
            for b in np.array_split(rng.permutation(len(allXe)), max(1, len(allXe) // 32)):
                opt.zero_grad()
                nn.functional.mse_loss(model(t["aXe"][b], t["aXd"][b], t["aXs"][b]),
                                       t["aY"][b]).backward()
                opt.step()
        # fine-tune on the real target
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                nn.functional.mse_loss(model(t["Xe"][b], t["Xd"][b], t["Xs"][b]),
                                       t["Y"][b]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(model(t["Xe"][va_i], t["Xd"][va_i], t["Xs"][va_i]),
                                           t["Y"][va_i]).item()
            if v < best - 1e-6:
                best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds.append(model(t["Ve"], t["Vd"], t["Vs"]).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ================================================== I09 probabilistic (Tweedie) head
def _tweedie_loss(mu, y, p=1.5, eps=1e-6):
    mu = mu.clamp(min=eps)
    return (-y * mu.pow(1 - p) / (1 - p) + mu.pow(2 - p) / (2 - p)).mean()


class SoftplusSeq2Seq(Seq2Seq):
    def forward(self, enc, dec, static):
        return nn.functional.softplus(super().forward(enc, dec, static))


def tweedie_head_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    t = {k: torch.tensor(v, dtype=torch.float32) for k, v in
         dict(Xe=Xe, Xd=Xd, Xs=Xs, Y=np.nan_to_num(Y), Ve=Ve, Vd=Vd, Vs=Vs).items()}
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = SoftplusSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                _tweedie_loss(model(t["Xe"][b], t["Xd"][b], t["Xs"][b]), t["Y"][b]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(model(t["Xe"][va_i], t["Xd"][va_i], t["Xs"][va_i]),
                                           t["Y"][va_i]).item()
            if v < best - 1e-6:
                best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds.append(model(t["Ve"], t["Vd"], t["Vs"]).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ==================================================== I10 MC-dropout trajectory sampler
def mc_dropout_forecaster(train_full, val_masked, n_samples=30, seeds=(0, 1)):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for seed in seeds:
        model = _fit_one(Xe, Xd, Xs, Y, seed)
        model.train()                                  # keep dropout active
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ============================================================= I11 storm augmentation
def augmented_gru_forecaster(train_full, val_masked, seeds=(0, 1, 2), n_aug=2,
                             cyclic=None, w=None):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    rng0 = np.random.RandomState(SEED)
    Xe_a, Xd_a, Xs_a, Y_a = [Xe], [Xd], [Xs], [Y]
    for _ in range(n_aug):
        shift = rng0.randint(-2, 3)
        amp = rng0.uniform(0.85, 1.15, size=(len(Xe), 1, 1))
        Xe_a.append(np.roll(Xe, shift, axis=1) * amp)
        Xd_a.append(np.roll(Xd, shift, axis=1) * rng0.uniform(0.9, 1.1, size=(len(Xd), 1, 1)))
        Xs_a.append(Xs)
        Y_a.append(np.roll(Y, shift, axis=1) * amp[:, :, 0])
    Xe2, Xd2, Xs2, Y2 = [np.concatenate(a, 0) for a in (Xe_a, Xd_a, Xs_a, Y_a)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for seed in seeds:
        got = _fit_one(Xe2, Xd2, Xs2, Y2, seed, cyclic=cyclic, w=w)
        for model in (got if cyclic is not None else [got]):
            with torch.no_grad():
                preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ==================================================================== I12 TFT-lite
class TFTLite(nn.Module):
    """Variable-selection network + GRU encoder/decoder + attention (TFT in spirit)."""

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.vsn = nn.Sequential(nn.Linear(n_dec + n_static, 64), nn.ReLU(),
                                 nn.Linear(64, n_dec), nn.Softmax(-1))
        self.enc = nn.GRU(n_enc, hidden, batch_first=True)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.grn = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, 1)
        self.last_weights = None

    def forward(self, enc, dec, static):
        eo, h = self.enc(enc)
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        w = self.vsn(torch.cat([dec, s], -1))
        self.last_weights = w.detach().mean((0, 1))
        do, _ = self.dec(torch.cat([dec * w * dec.shape[-1] ** 0.5, s], -1), h)
        ctx, _ = self.attn(do, eo, eo)
        return self.head(self.grn(torch.cat([do, ctx], -1))).squeeze(-1)


def _fit_generic(cls, Xe, Xd, Xs, Y, seed, max_epochs=300, patience=30):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = cls(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, Y)]
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            nn.functional.mse_loss(model(t[0][b], t[1][b], t[2][b]), t[3][b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(t[0][va_i], t[1][va_i], t[2][va_i]),
                                       t[3][va_i]).item()
        if v < best - 1e-6:
            best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def tft_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for seed in seeds:
        model = _fit_generic(TFTLite, Xe, Xd, Xs, Y, seed)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ============================================ I18 in-dataset pretraining (EAGLE-I proxy)
def pretrain_indataset_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    """External historical pretraining (EAGLE-I) was not run — data access and
    external-data rules are unresolved. This is the in-dataset analogue:
    pretrain on the observed window (predict OSI h1-71 from weather), then
    fine-tune on the scored window."""
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    pre_hours = np.arange(1, FREEZE_H + 1)
    _, PXe, PXd, PXs = _tensors(mask_after_freeze(train_full), pre_hours)
    PY = (train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
          .sort_index().reindex(columns=pre_hours).to_numpy() * Y_SCALE)
    for a in (PXe, PXd, PXs):
        a[:] = (a - a.reshape(-1, a.shape[-1]).mean(0)) / (a.reshape(-1, a.shape[-1]).std(0) + 1e-6)
    t = {k: torch.tensor(v, dtype=torch.float32) for k, v in dict(
        Xe=Xe, Xd=Xd, Xs=Xs, Y=Y, Ve=Ve, Vd=Vd, Vs=Vs,
        PXe=PXe, PXd=PXd, PXs=PXs, PY=np.nan_to_num(PY)).items()}
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        model = Seq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        for _ in range(60):                                     # pretrain on observed window
            model.train()
            for b in np.array_split(rng.permutation(len(PXe)), max(1, len(PXe) // 32)):
                opt.zero_grad()
                nn.functional.mse_loss(model(t["PXe"][b], t["PXd"][b], t["PXs"][b]),
                                       t["PY"][b]).backward()
                opt.step()
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                nn.functional.mse_loss(model(t["Xe"][b], t["Xd"][b], t["Xs"][b]),
                                       t["Y"][b]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(model(t["Xe"][va_i], t["Xd"][va_i], t["Xs"][va_i]),
                                           t["Y"][va_i]).item()
            if v < best - 1e-6:
                best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds.append(model(t["Ve"], t["Vd"], t["Vs"]).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ============================================================= I20 more seeds (8)
def gru_8seed_forecaster(train_full, val_masked):
    return gru_forecaster(train_full, val_masked, seeds=tuple(range(8)))


ENSEMBLE_V6_W = {"i07": 0.25, "i11": 0.20, "gru_comp": 0.15, "transformer": 0.15,
                 "gru": 0.05, "perstate_hurdle": 0.10, "i16": 0.10}


def ensemble_v6_forecaster(train_full, val_masked):
    """Production ensemble after the 20-idea sweep: cross-county attention and
    storm augmentation lead, with the component-GRU, Transformer, plain GRU,
    per-state hurdle blend, and static-enriched GBM for diversity.

    Round weights on the flat optimum (NNLS-exact 0.00878 concentrates on two
    members and is treated as weight-overfitting; these round weights score
    0.00901 OOF and win 3/5 folds against the previous 5-way at 0.00930).
    """
    from .dl import gru_forecaster, gru_component_forecaster, transformer_forecaster
    from .models import blend_perstate_hurdle_forecaster
    fns = {"i07": cross_county_forecaster,
           "i11": augmented_gru_forecaster,
           "gru_comp": gru_component_forecaster,
           "transformer": transformer_forecaster,
           "gru": gru_forecaster,
           "perstate_hurdle": blend_perstate_hurdle_forecaster,
           "i16": static_enrichment_forecaster}
    out = None
    for name, fn in fns.items():
        p = fn(train_full, val_masked).rename(columns={"osi_pred": name})
        out = p if out is None else out.merge(p, on=["fipsCode", "hour"])
    out["osi_pred"] = sum(w * out[n] for n, w in ENSEMBLE_V6_W.items())
    return out[["fipsCode", "hour", "osi_pred"]]


IDEAS = {
    "I01 amplitude x shape": amplitude_shape_forecaster,
    "I02 impulse-response (system ID)": impulse_response_forecaster,
    "I03 FPCA two-stage": fpca_forecaster,
    "I04 wave-conditioned cascade": wave_cascade_forecaster,
    "I05 archetype gate": archetype_gate_forecaster,
    "I05b archetype ORACLE (ceiling)": archetype_oracle_forecaster,
    "I06 quantile two-stage": quantile_forecaster,
    "I07 cross-county attention": cross_county_forecaster,
    "I08 transductive pretraining": transductive_forecaster,
    "I09 Tweedie emission head": tweedie_head_forecaster,
    "I10 MC-dropout sampler": mc_dropout_forecaster,
    "I11 storm augmentation": augmented_gru_forecaster,
    "I12 TFT-lite": tft_forecaster,
    "I13 Poisson/Tweedie GBM": poisson_gbm_forecaster,
    "I14 hierarchical pooling": hierarchical_forecaster,
    "I15 survival restoration": survival_restoration_forecaster,
    "I16 static enrichment": static_enrichment_forecaster,
    "I17 physics indices": physics_indices_forecaster,
    "I17b static + physics": static_physics_forecaster,
    "I18 in-dataset pretraining": pretrain_indataset_forecaster,
    "I20 GRU 8 seeds": gru_8seed_forecaster,
}
