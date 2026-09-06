"""Sweep 14 (N-series) - negative-correlation learning: train FOR a new error basis.

Ten saturation confirmations say ensemble headroom is governed by error-basis
diversity, not member quality (law 12) - and every member to date was trained
independently, with decorrelation left to chance. NCL trains K copies JOINTLY:

    L = (1/K) sum_k L_k  -  lambda * (1/K) sum_k w.(f_k - f_bar)^2

By the Krogh-Vedelsby ambiguity decomposition, ensemble MSE =
avg individual MSE - avg ambiguity, so this loss interpolates between
independent training (lambda=0) and directly optimizing the deployed
ensemble-mean MSE (lambda=1). Citations: Liu & Yao, Neural Networks 1999;
Brown, Wyatt & Tino, JMLR 2005; Buschjaeger et al., arXiv:2011.02952.

Implementation: K=4 BiEncSeq2Seq copies sharing S62's exact objective (scoring
hour weights + dual raw/sqrt loss), ambiguity computed on the RAW OSI scale
(where the metric lives) with the same hour weights. Early stopping on the
GROUP MEAN's weighted OSI loss - the deployed quantity. Per project law 13,
seeds (0,1,2) each train an independent K=4 group; the member output is the
mean over groups of the group mean.

N1t adds a transductive arm: ambiguity additionally computed on the held-out
counties' INPUTS (their observed prefix h<=71 + weather - both legal; no
labels touched), encouraging disagreement exactly where the metric will be
computed (D-BAT flavour, ICLR 2023, adapted to regression).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS
from .dl import Y_SCALE

IDEAS12 = {}


def _fit_ncl(Xe, Xd, Xs, Y, seed, w_hours, K=4, lam=0.3, trans=None,
             max_epochs=180, patience=25, lr=2e-3):
    from .ideas3 import BiEncSeq2Seq
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    models = nn.ModuleList([BiEncSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
                            for _ in range(K)])
    # break symmetry: distinct init per copy
    for k, m in enumerate(models):
        torch.manual_seed(seed * 100 + k)
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    opt = torch.optim.Adam(models.parameters(), lr=lr, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tw = torch.tensor(w_hours, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))
    tt = None
    if trans is not None:                        # transductive input block
        tt = [torch.tensor(v, dtype=torch.float32) for v in trans]

    def dual(out_sqrt, raw, i):
        den = (msk[i] * tw).sum()
        raw_t = ((raw - t[3][i]) ** 2 * tw * msk[i]).sum() / den
        root_t = ((out_sqrt - sq[i]) ** 2 * tw * msk[i]).sum() / den
        return 0.5 * raw_t + 0.5 * root_t

    def forward_all(e, d, s):
        outs = [m(e, d, s) for m in models]
        raws = torch.stack([torch.clamp(o, min=0) ** 2 for o in outs], 0)
        return outs, raws

    best, state, bad = np.inf, None, 0
    for _ in range(max_epochs):
        models.train()
        for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
            opt.zero_grad()
            outs, raws = forward_all(t[0][b], t[1][b], t[2][b])
            mean_raw = raws.mean(0)
            l_ind = torch.stack([dual(outs[k], raws[k], b) for k in range(K)]).mean()
            amb = (((raws - mean_raw[None]) ** 2) * tw * msk[b][None]).sum() \
                / (K * (msk[b] * tw).sum())
            loss = l_ind - lam * amb
            if tt is not None:                   # disagreement on held-out inputs
                m_i = rng.randint(0, len(tt[0]), size=min(32, len(tt[0])))
                _, raws_v = forward_all(tt[0][m_i], tt[1][m_i], tt[2][m_i])
                amb_v = ((raws_v - raws_v.mean(0, keepdim=True)) ** 2).mean()
                loss = loss - trans_lam(lam) * amb_v
            loss.backward()
            opt.step()
        models.eval()
        with torch.no_grad():
            _, raws = forward_all(t[0][va_i], t[1][va_i], t[2][va_i])
            mr = raws.mean(0)
            v = float((((mr - t[3][va_i]) ** 2) * tw * msk[va_i]).sum()
                      / (msk[va_i] * tw).sum())
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in models.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    models.load_state_dict(state)
    models.eval()
    return models


def trans_lam(lam):
    return 0.5 * lam                             # half weight on the unlabeled block


def _run_ncl(train_full, val_masked, seeds=(0, 1, 2), K=4, lam=0.3, trans=False):
    from .ideas import _prep, _dl_out
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    w = scoring_weights()
    tr_block = (Ve, Vd, Vs) if trans else None
    preds = []
    for s in seeds:
        models = _fit_ncl(Xe, Xd, Xs, Y, s, w, K=K, lam=lam, trans=tr_block)
        tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
        with torch.no_grad():
            raws = torch.stack([torch.clamp(m(*tv), min=0) ** 2 for m in models], 0)
            preds.append(raws.mean(0).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ---------------- N2: anti-correlation vs the SHIPPED ensemble's error vector
# N1's lesson: within-group ambiguity diversifies members around their own
# mean, which averaging then removes (group-mean corr vs s62 = 0.966). The
# seat pays for decorrelation against the EXISTING ensemble, so penalize that
# directly: train one member with S62's loss + lam * corr(f - y, ens - y)^2,
# where ens is the cached shipped-ensemble OOF (outer-training counties only -
# their OOF values come from models that never saw them, and the member never
# sees held-out counties' references at all).
def _fit_anticorr(Xe, Xd, Xs, Y, R, seed, w_hours, lam=0.5,
                  max_epochs=300, patience=30, lr=2e-3):
    from .ideas3 import BiEncSeq2Seq
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = BiEncSeq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    tR = torch.tensor(np.nan_to_num(R), dtype=torch.float32)
    tw = torch.tensor(w_hours, dtype=torch.float32)
    msk = torch.tensor(np.isfinite(Y).astype(float), dtype=torch.float32)
    mskR = torch.tensor((np.isfinite(Y) & np.isfinite(R)).astype(float),
                        dtype=torch.float32)
    sq = torch.sqrt(torch.clamp(t[3], min=0))

    def loss_fn(out, i):
        raw = torch.clamp(out, min=0) ** 2
        den = (msk[i] * tw).sum()
        base = 0.5 * ((raw - t[3][i]) ** 2 * tw * msk[i]).sum() / den             + 0.5 * ((out - sq[i]) ** 2 * tw * msk[i]).sum() / den
        m = mskR[i] * tw
        e_new = (raw - t[3][i]) * m
        e_ens = (tR[i] - t[3][i]) * m
        num = (e_new * e_ens).sum()
        den2 = torch.sqrt((e_new ** 2).sum() * (e_ens ** 2).sum() + 1e-12)
        return base + lam * (num / den2) ** 2

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
            v = float((((p - t[3][va_i]) ** 2) * tw * msk[va_i]).sum()
                      / (msk[va_i] * tw).sum())
        if v < best - 1e-6:
            best, state, bad = v, {k: q.clone() for k, q in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    return model


def _run_anticorr(train_full, val_masked, seeds=(0, 1, 2), lam=0.5):
    from .ideas import _prep, _dl_out
    from .ideas4 import scoring_weights
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    fips = np.sort(train_full["fipsCode"].unique())
    R = (pd.read_csv("oof/ens8p1b.csv")
         .pivot_table(index="fipsCode", columns="hour", values="osi_pred")
         .reindex(index=fips, columns=PRED_HOURS).to_numpy() * Y_SCALE)
    w = scoring_weights()
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        m = _fit_anticorr(Xe, Xd, Xs, Y, R, s, w, lam=lam)
        with torch.no_grad():
            preds.append((torch.clamp(m(*tv), min=0) ** 2).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def n2_ac05(tr, va):    return _run_anticorr(tr, va, lam=0.5)
def n2_ac10(tr, va):    return _run_anticorr(tr, va, lam=1.0)


def n1_lam0(tr, va):    return _run_ncl(tr, va, lam=0.0)
def n1_lam03(tr, va):   return _run_ncl(tr, va, lam=0.3)
def n1_lam06(tr, va):   return _run_ncl(tr, va, lam=0.6)
def n1t_trans(tr, va):  return _run_ncl(tr, va, lam=0.3, trans=True)


IDEAS12 = {
    "N1 NCL K=4 lam=0 (joint control)": n1_lam0,
    "N1b NCL K=4 lam=0.3": n1_lam03,
    "N1c NCL K=4 lam=0.6": n1_lam06,
    "N1t NCL lam=0.3 + transductive disagreement": n1t_trans,
    "N2 anti-corr vs shipped ensemble, lam=0.5": n2_ac05,
    "N2b anti-corr lam=1.0": n2_ac10,
}
