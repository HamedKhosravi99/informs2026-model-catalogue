"""Sweep 15 (B-series continued) - the two 2025-finalist model families not yet
in the champion-calibrated table, reproduced under OUR deployable protocol.

  B2  ARIMA-VAR with county clustering (2025 2nd place, Kazemi Kheiri &
      Hajifar). Transferable form: KMeans county clusters on legal descriptors
      (statics + observed-prefix summaries + full-window weather summaries),
      then per-cluster pooled linear AR(3) with a cross-county cluster-mean
      lag (the "V" of VAR) + exogenous weather, ridge-fitted on training
      counties, rolled out CLOSED-LOOP from h71 for held-out counties - own
      predictions feed own lags, cluster-mates' predictions feed the
      cluster-mean lag (self-consistent; no post-h71 truth anywhere).

  B3  Geo-temporal deep learning (2025 3rd place, Devaraj/Nammi/Ranjan).
      Literal adjacency version of what our cross-county attention (S39) does
      softly: GRU-encode every county's observed 72 h, one message-passing
      round over each county's k=5 nearest TRAINING counties (lat/lon
      geography - static and legal; self excluded, per the S39 self-leak
      lesson), decode with the fused state. The 2025 4th-place arXiv reports
      county-GNNs overfit badly on their task; this measures it on ours.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .dl import Y_SCALE, _pivot
from .features import county_static, weather_by_hour
from .models import SEED

IDEAS13 = {}


# ------------------------------------------------------------------ B2 helpers
def _cluster_feats(masked):
    """Per-county descriptors, all legal at deployment (statics, observed
    prefix, weather any hour)."""
    from .ideas import load_static
    st = county_static(masked)
    ext = load_static().reindex(st.index)
    g = _pivot(masked, "gust", np.arange(216))
    tp = _pivot(masked, "tp", np.arange(216))
    X = np.column_stack([
        st[["obs_osi_max", "obs_osi_h71", "restored_frac", "fragility",
            "obs_gust_max", "pre_mean_pct", "log_customers"]].to_numpy(),
        ext[["log_pop", "log_density", "rucc", "lat", "lon"]].to_numpy(),
        np.nan_to_num(g.to_numpy()).max(1)[:, None],
        np.nan_to_num(g.to_numpy()).mean(1)[:, None],
        np.nan_to_num(tp.to_numpy()).sum(1)[:, None],
    ])
    return st.index.to_numpy(), np.nan_to_num(X)


def cluster_var_forecaster(train_full, val_masked, k=6, p_lags=3, alpha=1.0):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    masked_tr = mask_after_freeze(train_full)
    f_tr, X_tr = _cluster_feats(masked_tr)
    f_va, X_va = _cluster_feats(val_masked)
    sc = StandardScaler().fit(X_tr)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(sc.transform(X_tr))
    c_tr = km.labels_
    c_va = km.predict(sc.transform(X_va))

    O_tr = np.nan_to_num(_pivot(train_full, "osi", np.arange(216)).to_numpy())
    W_tr = {c: np.nan_to_num(_pivot(train_full, c, np.arange(216)).to_numpy())
            for c in ("gust", "tp", "t2m")}
    gm6_tr = np.stack([np.pad(W_tr["gust"], ((0, 0), (j, 0)), mode="edge")[:, :216]
                       for j in range(6)]).max(0)

    hod = np.arange(216) % 24
    hs, hc = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)

    coefs = {}
    for cl in range(k):
        m = c_tr == cl
        if m.sum() < 3:
            m = np.ones(len(c_tr), bool)          # tiny cluster: pool all
        O = O_tr[m]
        clm = O.mean(0)                            # cluster-mean series
        rows, ys = [], []
        for t in range(p_lags, 216):
            feats = [O[:, t - j] for j in range(1, p_lags + 1)]
            feats.append(np.full(m.sum(), clm[t - 1]))
            feats += [W_tr["gust"][m, t], gm6_tr[m, t], W_tr["tp"][m, t],
                      W_tr["t2m"][m, t] - 273.15,
                      np.full(m.sum(), hs[t]), np.full(m.sum(), hc[t]),
                      np.ones(m.sum())]
            rows.append(np.column_stack(feats))
            ys.append(O[:, t])
        A = np.vstack(rows)
        y = np.concatenate(ys)
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)
        coefs[cl] = w

    # ---- closed-loop rollout for held-out counties, clusters rolled jointly
    O_va = np.nan_to_num(_pivot(val_masked, "osi", np.arange(216)).to_numpy())
    W_va = {c: np.nan_to_num(_pivot(val_masked, c, np.arange(216)).to_numpy())
            for c in ("gust", "tp", "t2m")}
    gm6_va = np.stack([np.pad(W_va["gust"], ((0, 0), (j, 0)), mode="edge")[:, :216]
                       for j in range(6)]).max(0)
    P = O_va.copy()                                # h<=71 observed, rest predicted
    for t in range(FREEZE_H + 1, 216):
        clm_now = {cl: P[c_va == cl, t - 1].mean() if (c_va == cl).any() else 0.0
                   for cl in range(k)}
        for cl in range(k):
            m = c_va == cl
            if not m.any():
                continue
            feats = [P[m, t - j] for j in range(1, p_lags + 1)]
            feats.append(np.full(m.sum(), clm_now[cl]))
            feats += [W_va["gust"][m, t], gm6_va[m, t], W_va["tp"][m, t],
                      W_va["t2m"][m, t] - 273.15,
                      np.full(m.sum(), hs[t]), np.full(m.sum(), hc[t]),
                      np.ones(m.sum())]
            P[m, t] = np.clip(np.column_stack(feats) @ coefs[cl], 0, None)

    sel = np.isin(np.arange(216), PRED_HOURS)
    return pd.DataFrame({"fipsCode": np.repeat(f_va, len(PRED_HOURS)),
                         "hour": np.tile(PRED_HOURS, len(f_va)),
                         "osi_pred": P[:, sel].ravel()})


# ------------------------------------------------------------------ B3 GNN
class GeoGNN(nn.Module):
    """Bi-encoder + one adjacency message-passing round before decoding."""

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)
        self.msg = nn.Linear(hidden, hidden)
        self.fuse = nn.Linear(hidden * 2, hidden)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def encode(self, enc):
        _, h = self.enc(enc)
        return self.proj(torch.cat([h[0], h[1]], -1))

    def forward(self, enc, dec, static, mem_enc, nbr_idx):
        e = self.encode(enc)                          # (B, H)
        mem = self.encode(mem_enc)                    # (M, H) training pool
        nbr = torch.relu(self.msg(mem[nbr_idx]))      # (B, k, H)
        h0 = self.fuse(torch.cat([e, nbr.mean(1)], -1))[None]
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), h0.contiguous())
        return self.head(self.drop(out)).squeeze(-1)


def geo_gnn_forecaster(train_full, val_masked, seeds=(0, 1, 2), k_nbr=5):
    from .ideas import _prep, _dl_out, load_static
    from .ideas4 import scoring_weights

    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tf = np.sort(train_full["fipsCode"].unique())
    ll = load_static()
    ll_tr = ll.reindex(tf)[["lat", "lon"]].to_numpy()
    ll_va = ll.reindex(vf)[["lat", "lon"]].to_numpy()

    def knn(pts, pool, exclude_self):
        d = ((pts[:, None, :] - pool[None, :, :]) ** 2).sum(-1)
        if exclude_self:
            np.fill_diagonal(d, np.inf)
        return np.argsort(d, 1)[:, :k_nbr]

    nbr_tr = knn(ll_tr, ll_tr, exclude_self=True)     # S39 lesson: no self
    nbr_va = knn(ll_va, ll_tr, exclude_self=False)

    w = scoring_weights()
    tw = torch.tensor(w, dtype=torch.float32)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    nb_tr = torch.tensor(nbr_tr, dtype=torch.long)
    nb_va = torch.tensor(nbr_va, dtype=torch.long)

    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = GeoGNN(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

        def loss_fn(out, i):
            raw = torch.clamp(out, min=0) ** 2
            den = (msk[i] * tw).sum()
            return (0.5 * ((raw - t[3][i]) ** 2 * tw * msk[i]).sum() / den
                    + 0.5 * ((out - sq[i]) ** 2 * tw * msk[i]).sum() / den)

        best, state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                out = model(t[0][b], t[1][b], t[2][b], t[0], nb_tr[b])
                loss_fn(out, b).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                out = model(t[0][va_i], t[1][va_i], t[2][va_i], t[0], nb_tr[va_i])
                p = torch.clamp(out, min=0) ** 2
                v = float((((p - t[3][va_i]) ** 2) * tw * msk[va_i]).sum()
                          / (msk[va_i] * tw).sum())
            if v < best - 1e-6:
                best, state, bad = v, {k2: q.clone() for k2, q in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            preds.append((torch.clamp(model(tv[0], tv[1], tv[2], t[0], nb_va),
                                      min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def cluster_varx_direct(train_full, val_masked, k=6, alpha=3.0):
    """B2b - the ARIMA-VAR/SARIMAX families ADAPTED to the 2026 regime.

    2025's task was a temporal split over the same 83 Michigan counties, so
    per-county recursive AR was sensible. Under county transfer with a frozen
    prefix the fair version is the DIRECT form: frozen-boundary AR features
    (last observed levels + their exponential decay transforms), seasonal
    harmonics, exogenous future weather at the target hour, cluster one-hots
    and cluster interactions - one pooled ridge, no rollout, no drift.
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    masked_tr = mask_after_freeze(train_full)
    f_tr, X_tr = _cluster_feats(masked_tr)
    f_va, X_va = _cluster_feats(val_masked)
    sc = StandardScaler().fit(X_tr)
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(sc.transform(X_tr))
    c_tr = km.labels_
    c_va = km.predict(sc.transform(X_va))

    def design(df_for_weather, masked_frame, cl):
        O = np.nan_to_num(_pivot(masked_frame, "osi", np.arange(216)).to_numpy())
        W = {c: np.nan_to_num(_pivot(df_for_weather, c, np.arange(216)).to_numpy())
             for c in ("gust", "tp", "t2m")}
        gm6 = np.stack([np.pad(W["gust"], ((0, 0), (j, 0)), mode="edge")[:, :216]
                        for j in range(6)]).max(0)
        o71, o70, o69 = O[:, 71], O[:, 70], O[:, 69]
        om = O[:, 66:72].mean(1)
        hod = np.arange(216) % 24
        hs, hc = np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)
        rows = []
        for t in range(FREEZE_H + 2, 216):
            lead = float(t - FREEZE_H)
            n = len(O)
            dec = [o71 * np.exp(-lead / T) for T in (6, 12, 24, 48)]
            cl1 = np.eye(k)[cl]
            feats = [o71, o70, o69, om, *dec,
                     np.full(n, lead / 100), W["gust"][:, t], gm6[:, t],
                     W["tp"][:, t], W["t2m"][:, t] - 273.15,
                     np.full(n, hs[t]), np.full(n, hc[t])]
            base = np.column_stack(feats)
            inter = np.column_stack([cl1 * o71[:, None], cl1 * W["gust"][:, t][:, None]])
            rows.append(np.hstack([base, cl1, inter, np.ones((n, 1))]))
        return np.stack(rows)                       # (T, n, F)

    D_tr = design(train_full, masked_tr, c_tr)
    O_full = np.nan_to_num(_pivot(train_full, "osi", np.arange(216)).to_numpy())
    ys = np.stack([O_full[:, t] for t in range(FREEZE_H + 2, 216)])
    A = D_tr.reshape(-1, D_tr.shape[-1])
    y = ys.ravel()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ y)

    D_va = design(val_masked, val_masked, c_va)
    P = np.clip(D_va @ w, 0, None)                  # (T, n_va)
    hours = np.arange(FREEZE_H + 2, 216)
    out = pd.DataFrame({
        "fipsCode": np.repeat(f_va, len(hours)),
        "hour": np.tile(hours, len(f_va)),
        "osi_pred": P.T.ravel()})
    return out


def b2_cluster_var(tr, va):  return cluster_var_forecaster(tr, va)
def b2b_varx_direct(tr, va): return cluster_varx_direct(tr, va)
def b3_geo_gnn(tr, va):      return geo_gnn_forecaster(tr, va)


IDEAS13 = {
    "B2 cluster-VAR (2025 2nd-place analog)": b2_cluster_var,
    "B2b cluster-VARX direct form (adapted to 2026 regime)": b2b_varx_direct,
    "B3 geo-temporal adjacency GNN (2025 3rd-place analog)": b3_geo_gnn,
}
