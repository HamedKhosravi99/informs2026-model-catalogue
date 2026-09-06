"""Deep-learning forecasters: MLP on engineered features, GRU seq2seq on raw series.

Same protocol as every other forecaster: f(train_full, val_masked) -> trajectory
DataFrame (fipsCode, hour, osi_pred) for hours 73-215. Outage inputs come only
from the masked frame (<= h71); weather conditions the decoder at all hours.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .features import build_rows, feature_cols, county_static, weather_by_hour

SEED = 42
Y_SCALE = 50.0  # OSI is O(0.01); scale targets so MSE gradients are healthy

ENC_COLS = ["osi", "P_t", "N_t", "R_t", "gust", "wind_speed_10m", "tp", "t2m"]
DEC_COLS = ["gust", "wind_speed_10m", "wind_sin", "wind_cos", "t2m", "dew_spread",
            "tp", "csnow", "mslma", "blh", "soil_moist", "sdwe", "hod_sin",
            "hod_cos", "gust_max6", "gust_max24", "new_exposure"]
STATIC_COLS = ["state_IN", "state_OH", "state_PA", "state_WV", "log_customers",
               "obs_osi_max", "obs_osi_h71", "restored_frac", "fragility",
               "obs_gust_max", "pre_mean_pct"]


# ------------------------------------------------------------------ MLP (rung A)
def mlp_forecaster(train_full, val_masked):
    tr_rows = build_rows(mask_after_freeze(train_full))
    y = train_full.set_index(["fipsCode", "hour"])["osi"].reindex(
        pd.MultiIndex.from_frame(tr_rows[["fipsCode", "hour"]])).to_numpy()
    keep = np.isfinite(y)
    cols = feature_cols(tr_rows)
    model = make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-4, batch_size=512,
                     learning_rate_init=1e-3, max_iter=400, early_stopping=True,
                     n_iter_no_change=20, random_state=SEED))
    model.fit(tr_rows.loc[keep, cols], y[keep] * Y_SCALE)
    va_rows = build_rows(val_masked)
    out = va_rows[["fipsCode", "hour"]].copy()
    out["osi_pred"] = np.clip(model.predict(va_rows[cols]) / Y_SCALE, 0.0, None)
    return out


# ------------------------------------------------------------- GRU seq2seq (rung B)
DEC_HOURS = np.arange(FREEZE_H + 1, 216)  # 72..215; component model decodes h72 too


def _pivot(df, col, hours):
    return (df.pivot_table(index="fipsCode", columns="hour", values=col)
            .reindex(columns=hours).sort_index())


def _tensors(masked: pd.DataFrame, dec_hours=PRED_HOURS):
    """Per county: encoder seq (72 x E), decoder seq (len(dec_hours) x D), static (S)."""
    enc_hours = np.arange(0, FREEZE_H + 1)
    st = county_static(masked)
    w = weather_by_hour(masked)
    fips = _pivot(masked, "gust", enc_hours).index.to_numpy()

    enc = np.stack([_pivot(masked, c, enc_hours).to_numpy() for c in ENC_COLS], -1)
    dec = np.stack([_pivot(w, c, dec_hours).to_numpy() for c in DEC_COLS], -1)
    # per-step extras: staleness clock + decay basis of the last observed state
    dt = (dec_hours - FREEZE_H).astype(float)
    osi71 = st.loc[fips, "obs_osi_h71"].to_numpy()[:, None]
    extras = np.stack([np.repeat(dt[None, :] / 100.0, len(fips), 0),
                       osi71 * np.exp(-dt / 16.0) * Y_SCALE,
                       osi71 * np.exp(-dt / 32.0) * Y_SCALE], -1)
    dec = np.concatenate([dec, extras], -1)
    static = st.loc[fips, STATIC_COLS].to_numpy()
    return fips, np.nan_to_num(enc), np.nan_to_num(dec), np.nan_to_num(static)


class Seq2Seq(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), h)
        return self.head(self.drop(out)).squeeze(-1)


def _wmse(a, b, w):
    """MSE with per-target-hour weights, normalised to mean 1 so the loss scale
    -- and therefore the learning rate -- is unchanged when w is supplied.
    w=None returns the unweighted loss itself, so existing callers are exact."""
    return (((a - b) ** 2) * w).mean() if w is not None else nn.functional.mse_loss(a, b)


def _fit_one(Xe, Xd, Xs, Y, seed, max_epochs=400, patience=40, cyclic=None, w=None):
    """cyclic=(period, n_snapshots) enables cosine warm restarts and returns a
    LIST of models, one per cycle end (snapshot ensembling, law 13). Default
    None reproduces the original single-model behaviour byte-for-byte.

    w=None keeps the flat MSE. Supplying `scoring_weights()` aligns the loss
    with the metric, which scores each submission column over its own target
    hours -- late hours appear in all four columns, hours 73-77 in only one."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va, tr = idx[:n_val], idx[n_val:]
    model = Seq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t = {k: torch.tensor(v, dtype=torch.float32) for k, v in
         dict(Xe=Xe, Xd=Xd, Xs=Xs, Y=Y).items()}
    tw = None if w is None else torch.tensor(w, dtype=torch.float32)
    best, best_state, bad = np.inf, None, 0
    snaps = []
    n_ep = max_epochs if cyclic is None else cyclic[0] * cyclic[1]
    for ep in range(n_ep):
        if cyclic is not None:                      # cosine warm restarts
            pos = (ep % cyclic[0]) / cyclic[0]
            for gp in opt.param_groups:
                gp["lr"] = 2e-3 * 0.5 * (1 + np.cos(np.pi * pos))
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
            opt.zero_grad()
            loss = _wmse(model(t["Xe"][b], t["Xd"][b], t["Xs"][b]), t["Y"][b], tw)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = _wmse(model(t["Xe"][va], t["Xd"][va], t["Xs"][va]), t["Y"][va], tw).item()
        if cyclic is not None:
            if (ep + 1) % cyclic[0] == 0:           # end of a cycle -> snapshot
                m2 = Seq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
                m2.load_state_dict({k: p.clone() for k, p in model.state_dict().items()})
                m2.eval()
                snaps.append(m2)
            continue
        if v < best - 1e-6:
            best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if cyclic is not None:
        return snaps
    model.load_state_dict(best_state)
    model.eval()
    return model


def gru_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    _, Xe, Xd, Xs = _tensors(mask_after_freeze(train_full))
    truth = train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
    Y = truth.sort_index().reindex(columns=PRED_HOURS).to_numpy() * Y_SCALE

    def norm_stats(a):
        m = a.reshape(-1, a.shape[-1]).mean(0)
        s = a.reshape(-1, a.shape[-1]).std(0) + 1e-6
        return m, s

    stats = [norm_stats(x) for x in (Xe, Xd, Xs)]
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]

    vf, Ve, Vd, Vs = _tensors(val_masked)
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]

    preds = []
    for seed in seeds:
        model = _fit_one(Xe, Xd, Xs, Y, seed)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    P = np.clip(np.mean(preds, 0) / Y_SCALE, 0.0, None)
    out = pd.DataFrame({"fipsCode": np.repeat(vf, len(PRED_HOURS)),
                        "hour": np.tile(PRED_HOURS, len(vf)),
                        "osi_pred": P.ravel()})
    return out


# ------------------------------------------------------- Transformer (full attention)
class TransformerTraj(nn.Module):
    """Single encoder over the full 216h sequence; outage channels are zeroed
    after the freeze (with an observed-flag channel), weather runs the whole
    window, every target hour attends to the entire storm history and future."""

    def __init__(self, n_in, d=64, heads=4, layers=2, ff=128, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_in, d)
        self.pos = nn.Parameter(torch.zeros(1, 216, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        h = self.enc(self.proj(x) + self.pos)
        return self.head(h).squeeze(-1)[:, PRED_HOURS[0]:]     # hours 73..215


def _seq_tensor(masked: pd.DataFrame):
    """(B, 216, F) input: masked outage block + flag, weather, static, decay, time."""
    hours = np.arange(216)
    st = county_static(masked)
    w = weather_by_hour(masked)
    fips = _pivot(masked, "gust", hours).index.to_numpy()

    outage = np.stack([np.nan_to_num(_pivot(masked, c, hours).to_numpy()) * Y_SCALE
                       for c in ["osi", "P_t", "N_t", "R_t"]], -1)
    flag = np.repeat((hours <= FREEZE_H)[None, :, None].astype(float), len(fips), 0)
    weather = np.stack([np.nan_to_num(_pivot(w, c, hours).to_numpy()) for c in DEC_COLS], -1)
    static = np.repeat(np.nan_to_num(st.loc[fips, STATIC_COLS].to_numpy())[:, None, :], 216, 1)
    dt = np.clip(hours - FREEZE_H, 0, None).astype(float)
    osi71 = st.loc[fips, "obs_osi_h71"].to_numpy()[:, None]
    decay = np.stack([np.where(hours > FREEZE_H, osi71 * np.exp(-dt / 16.0), 0) * Y_SCALE,
                      np.where(hours > FREEZE_H, osi71 * np.exp(-dt / 32.0), 0) * Y_SCALE], -1)
    tpos = np.repeat((hours[None, :, None] / 216.0), len(fips), 0)
    X = np.concatenate([outage, flag, weather, static, decay, tpos], -1)
    return fips, X


def _fit_transformer(X, Y, seed, max_epochs=400, patience=40):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_val = max(12, len(idx) // 8)
    va, tr = idx[:n_val], idx[n_val:]
    model = TransformerTraj(X.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tX = torch.tensor(X, dtype=torch.float32)
    tY = torch.tensor(Y, dtype=torch.float32)
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(tX[b]), tY[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = nn.functional.mse_loss(model(tX[va]), tY[va]).item()
        if v < best - 1e-6:
            best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def transformer_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    _, X = _seq_tensor(mask_after_freeze(train_full))
    Y = (train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
         .sort_index().reindex(columns=PRED_HOURS).to_numpy() * Y_SCALE)
    m = X.reshape(-1, X.shape[-1]).mean(0)
    s = X.reshape(-1, X.shape[-1]).std(0) + 1e-6
    X = (X - m) / s
    vf, V = _seq_tensor(val_masked)
    V = torch.tensor((V - m) / s, dtype=torch.float32)
    preds = []
    for seed in seeds:
        model = _fit_transformer(X, Y, seed)
        with torch.no_grad():
            preds.append(model(V).numpy())
    P = np.clip(np.mean(preds, 0) / Y_SCALE, 0.0, None)
    return pd.DataFrame({"fipsCode": np.repeat(vf, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(vf)),
                         "osi_pred": P.ravel()})


# ------------------------------------------- component-structured GRU (physics head)
class ComponentSeq2Seq(nn.Module):
    """Predict P_t, N_t, R_t per hour; OSI is composed via its exact definition.

    D_t is the 6h rolling mean of P_t (an identity verified on the data), seeded
    with the observed P at hours 66-71 so early-window persistence is exact.
    """

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(3)])  # P, N, R

    def forward(self, enc, dec, static, p_obs):
        _, h = self.enc(enc)
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), h)
        out = self.drop(out)
        P, N, R = [nn.functional.softplus(hd(out)).squeeze(-1) for hd in self.heads]
        p_full = torch.cat([p_obs, P], 1)                       # hours 66..215
        D = p_full.unfold(1, 6, 1).mean(-1)                     # D at hours 71..215
        osi = (0.40 * P + 0.35 * N + 0.25 * D[:, 1:] - 0.10 * R).clamp(min=0)
        return osi, P, N, R                                     # hours 72..215


def _fit_component(Xe, Xd, Xs, Pobs, YP, YN, YR, Yosi, seed,
                   max_epochs=400, patience=40, aux_w=0.3):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va, tr = idx[:n_val], idx[n_val:]
    model = ComponentSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    t = {k: torch.tensor(v, dtype=torch.float32) for k, v in
         dict(Xe=Xe, Xd=Xd, Xs=Xs, Pobs=Pobs, YP=YP, YN=YN, YR=YR, Yosi=Yosi).items()}
    mse = nn.functional.mse_loss
    best, best_state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        model.train()
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 32)):
            opt.zero_grad()
            osi, P, N, R = model(t["Xe"][b], t["Xd"][b], t["Xs"][b], t["Pobs"][b])
            loss = mse(osi, t["Yosi"][b]) + aux_w * (
                mse(P, t["YP"][b]) + mse(N, t["YN"][b]) + mse(R, t["YR"][b]))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            osi, *_ = model(t["Xe"][va], t["Xd"][va], t["Xs"][va], t["Pobs"][va])
            v = mse(osi, t["Yosi"][va]).item()   # early-stop on the scored quantity
        if v < best - 1e-6:
            best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def gru_component_forecaster(train_full, val_masked, seeds=(0, 1, 2)):
    masked_tr = mask_after_freeze(train_full)
    _, Xe, Xd, Xs = _tensors(masked_tr, DEC_HOURS)
    obs_hours = np.arange(66, FREEZE_H + 1)
    Pobs = np.nan_to_num(_pivot(masked_tr, "P_t", obs_hours).to_numpy()) * Y_SCALE
    YP = _pivot(train_full, "P_t", DEC_HOURS).to_numpy() * Y_SCALE
    YN = _pivot(train_full, "N_t", DEC_HOURS).to_numpy() * Y_SCALE
    YR = _pivot(train_full, "R_t", DEC_HOURS).to_numpy() * Y_SCALE
    Yosi = _pivot(train_full, "osi", DEC_HOURS).to_numpy() * Y_SCALE

    def norm_stats(a):
        m = a.reshape(-1, a.shape[-1]).mean(0)
        s = a.reshape(-1, a.shape[-1]).std(0) + 1e-6
        return m, s

    stats = [norm_stats(x) for x in (Xe, Xd, Xs)]
    Xe, Xd, Xs = [(x - m) / s for x, (m, s) in zip((Xe, Xd, Xs), stats)]

    vf, Ve, Vd, Vs = _tensors(val_masked, DEC_HOURS)
    VPobs = np.nan_to_num(_pivot(val_masked, "P_t", obs_hours).to_numpy()) * Y_SCALE
    Ve, Vd, Vs = [(x - m) / s for x, (m, s) in zip((Ve, Vd, Vs), stats)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs, VPobs)]

    preds = []
    for seed in seeds:
        model = _fit_component(Xe, Xd, Xs, Pobs, YP, YN, YR, Yosi, seed)
        with torch.no_grad():
            osi, *_ = model(*tv)
            preds.append(osi.numpy())
    P = np.clip(np.mean(preds, 0) / Y_SCALE, 0.0, None)[:, 1:]   # keep hours 73..215
    out = pd.DataFrame({"fipsCode": np.repeat(vf, len(PRED_HOURS)),
                        "hour": np.tile(PRED_HOURS, len(vf)),
                        "osi_pred": P.ravel()})
    return out


def _decay_blend(fc):
    from .models import decay_forecaster
    def blended(train_full, val_masked):
        g = fc(train_full, val_masked).rename(columns={"osi_pred": "g"})
        d = decay_forecaster(train_full, val_masked).rename(columns={"osi_pred": "d"})
        m = g.merge(d, on=["fipsCode", "hour"])
        w = 1.0 / (1.0 + np.exp(-(m["hour"] - 100) / 10.0))
        m["osi_pred"] = (1 - w) * m["d"] + w * m["g"]
        return m[["fipsCode", "hour", "osi_pred"]]
    return blended


mlp_blend_forecaster = _decay_blend(mlp_forecaster)
gru_blend_forecaster = _decay_blend(gru_forecaster)

ENSEMBLE_W = {"gru": 0.25, "gru_comp": 0.25, "transformer": 0.2,
              "perstate_hurdle": 0.15, "mlp": 0.15}


def ensemble_forecaster(train_full, val_masked):
    """Primary model: three sequence models (GRU, component-GRU, Transformer)
    plus two tabular models (hurdle blend + MLP), simple weighted average.

    Weights are round values on the flat optimum of the OOF sweep (NNLS exact
    0.00931 vs round 0.00934); the 5-way wins 4/5 folds vs the 4-way and has
    the best MAE (0.00282). Point RMSE gains between ensemble variants are
    within fold noise and are claimed as robustness, not improvement.
    """
    from .models import blend_perstate_hurdle_forecaster
    parts = {"gru": gru_forecaster(train_full, val_masked),
             "gru_comp": gru_component_forecaster(train_full, val_masked),
             "transformer": transformer_forecaster(train_full, val_masked),
             "perstate_hurdle": blend_perstate_hurdle_forecaster(train_full, val_masked),
             "mlp": mlp_forecaster(train_full, val_masked)}
    out = None
    for name, p in parts.items():
        p = p.rename(columns={"osi_pred": name})
        out = p if out is None else out.merge(p, on=["fipsCode", "hour"])
    out["osi_pred"] = sum(w * out[n] for n, w in ENSEMBLE_W.items())
    return out[["fipsCode", "hour", "osi_pred"]]
