"""Sweep 3: alternative sequence architectures, physics-informed models,
augmentation variants, and variance-reduction schemes.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import FREEZE_H, PRED_HOURS, mask_after_freeze
from .dl import Y_SCALE, DEC_COLS, Seq2Seq, _fit_one, _tensors, _pivot
from .ideas import _prep, _dl_out, cross_county_forecaster
from .models import SEED

# ------------------------------------------------------------- generic fit helper
def _fit(cls, Xe, Xd, Xs, Y, seed, max_epochs=300, patience=30, lr=2e-3, **kw):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xe))
    n_val = max(12, len(idx) // 8)
    va_i, tr_i = idx[:n_val], idx[n_val:]
    model = cls(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1], **kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
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


def _run(cls, train_full, val_masked, seeds=(0, 1, 2), **kw):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        model = _fit(cls, Xe, Xd, Xs, Y, s, **kw)
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ==================================================================== S30 LSTM
class LSTMSeq2Seq(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.enc = nn.LSTM(n_enc, hidden, batch_first=True)
        self.dec = nn.LSTM(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, enc, dec, static):
        _, (h, c) = self.enc(enc)
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), (h, c))
        return self.head(self.drop(out)).squeeze(-1)


def lstm_forecaster(train_full, val_masked):
    return _run(LSTMSeq2Seq, train_full, val_masked)


# ======================================================= S31 bidirectional encoder
class BiEncSeq2Seq(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, hidden)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        h0 = self.proj(torch.cat([h[0], h[1]], -1))[None]
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), h0.contiguous())
        return self.head(self.drop(out)).squeeze(-1)


def bienc_forecaster(train_full, val_masked):
    return _run(BiEncSeq2Seq, train_full, val_masked)


# ============================================================= S32 temporal CNN (TCN)
class TCN(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, hidden=64, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True)
        layers, ch = [], n_dec + n_static + hidden
        for d in (1, 2, 4, 8, 16):
            layers += [nn.Conv1d(ch, hidden, 3, padding=d, dilation=d), nn.ELU(),
                       nn.Dropout(dropout)]
            ch = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Conv1d(hidden, 1, 1)

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        s = torch.cat([static, h[-1]], -1)[:, None, :].expand(-1, dec.shape[1], -1)
        x = torch.cat([dec, s], -1).transpose(1, 2)
        return self.head(self.tcn(x)).squeeze(1)


def tcn_forecaster(train_full, val_masked):
    return _run(TCN, train_full, val_masked)


# ==================================================== S33 N-BEATS style basis expansion
class BasisNet(nn.Module):
    """Predict coefficients of a fixed temporal basis (polynomial + Fourier +
    exponential-decay), so the trajectory is smooth by construction."""

    def __init__(self, n_enc, n_dec, n_static, hidden=64, dropout=0.15):
        super().__init__()
        H = len(PRED_HOURS)
        t = np.linspace(0, 1, H)
        basis = [np.ones(H), t, t ** 2, t ** 3]
        for k in (1, 2, 3, 4):
            basis += [np.sin(2 * np.pi * k * t), np.cos(2 * np.pi * k * t)]
        for tau in (0.05, 0.1, 0.2, 0.4):
            basis.append(np.exp(-t / tau))
        self.register_buffer("B", torch.tensor(np.stack(basis), dtype=torch.float32))
        self.enc = nn.GRU(n_enc, hidden, batch_first=True)
        n_in = hidden + n_static + n_dec * 4
        self.net = nn.Sequential(nn.Linear(n_in, 128), nn.ELU(), nn.Dropout(dropout),
                                 nn.Linear(128, self.B.shape[0]))

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        d = torch.cat([dec.mean(1), dec.max(1).values, dec[:, :24].mean(1),
                       dec[:, -24:].mean(1)], -1)
        coef = self.net(torch.cat([h[-1], static, d], -1))
        return coef @ self.B


def basis_forecaster(train_full, val_masked):
    return _run(BasisNet, train_full, val_masked)


# ============================================== S34 physics-informed net (fragility+decay)
class PhysicsNet(nn.Module):
    """Mechanistic core: damage rate = fragility(gust) x exposure, restoration =
    exponential with learned county rate; a bounded neural residual corrects it.
    OSI is composed from the simulated state, so the dynamics are enforced."""

    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.gi = DEC_COLS.index("gust")
        self.enc = nn.GRU(n_enc, hidden, batch_first=True)
        self.params = nn.Sequential(nn.Linear(hidden + n_static, 64), nn.ELU(),
                                    nn.Linear(64, 4))     # k, x0, log_tau, gain
        self.res = nn.GRU(n_dec + n_static + 1, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.res_head = nn.Linear(hidden, 1)

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        p = self.params(torch.cat([h[-1], static], -1))
        k = nn.functional.softplus(p[:, 0:1]) + 0.1
        x0 = p[:, 1:2]
        tau = nn.functional.softplus(p[:, 2:3]) + 1.0
        gain = nn.functional.softplus(p[:, 3:4])
        gust = dec[:, :, self.gi]
        frag = torch.sigmoid(k * (gust - x0))                  # fragility curve
        state, out = enc[:, -1, 0:1].squeeze(-1), []            # last observed OSI level
        decay = torch.exp(-1.0 / tau).squeeze(-1)
        for t in range(dec.shape[1]):
            state = state * decay + (gain.squeeze(-1) * frag[:, t])
            out.append(state)
        phys = torch.stack(out, 1)
        r, _ = self.res(torch.cat([dec, static[:, None, :].expand(-1, dec.shape[1], -1),
                                   phys[:, :, None]], -1))
        return phys + self.res_head(self.drop(r)).squeeze(-1)


def physics_net_forecaster(train_full, val_masked):
    return _run(PhysicsNet, train_full, val_masked)


# ================================================================ S35 mixup augmentation
def mixup_forecaster(train_full, val_masked, seeds=(0, 1, 2), alpha=0.3, n_mix=2,
                     cyclic=None, w=None):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    rng = np.random.RandomState(SEED)
    Xe_a, Xd_a, Xs_a, Y_a = [Xe], [Xd], [Xs], [np.nan_to_num(Y)]
    n = len(Xe)
    for _ in range(n_mix):
        i, j = rng.permutation(n), rng.permutation(n)
        lam = rng.beta(alpha, alpha, size=(n, 1, 1))
        Xe_a.append(lam * Xe[i] + (1 - lam) * Xe[j])
        Xd_a.append(lam * Xd[i] + (1 - lam) * Xd[j])
        Xs_a.append(lam[:, :, 0] * Xs[i] + (1 - lam[:, :, 0]) * Xs[j])
        Y_a.append(lam[:, :, 0] * np.nan_to_num(Y)[i] + (1 - lam[:, :, 0]) * np.nan_to_num(Y)[j])
    A = [np.concatenate(a, 0) for a in (Xe_a, Xd_a, Xs_a, Y_a)]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for s in seeds:
        got = _fit_one(A[0], A[1], A[2], A[3], s, cyclic=cyclic, w=w)
        for model in (got if cyclic is not None else [got]):
            with torch.no_grad():
                preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ==================================================== S36 stochastic weight averaging
def swa_forecaster(train_full, val_masked, seeds=(0, 1, 2), n_snap=5):
    """Average weights from the last epochs (SWA) instead of just the best epoch."""
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    t = [torch.tensor(v, dtype=torch.float32) for v in (Xe, Xd, Xs, np.nan_to_num(Y))]
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = Seq2Seq(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, best_state, bad, snaps = np.inf, None, 0, []
        for ep in range(300):
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
                best, bad = v, 0
                snaps.append({k: p.clone() for k, p in model.state_dict().items()})
                snaps = snaps[-n_snap:]
            else:
                bad += 1
                if bad >= 30:
                    break
        avg = {k: torch.stack([s[k].float() for s in snaps]).mean(0) for k in snaps[0]}
        model.load_state_dict(avg)
        model.eval()
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ============================================ S37 multi-task (OSI + weather response)
class MultiTaskNet(nn.Module):
    def __init__(self, n_enc, n_dec, n_static, hidden=48, dropout=0.15):
        super().__init__()
        self.enc = nn.GRU(n_enc, hidden, batch_first=True)
        self.dec = nn.GRU(n_dec + n_static, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 3)      # OSI, P_t, N_t

    def forward(self, enc, dec, static):
        _, h = self.enc(enc)
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), h)
        return self.head(self.drop(out))[:, :, 0]

    def all_heads(self, enc, dec, static):
        _, h = self.enc(enc)
        s = static[:, None, :].expand(-1, dec.shape[1], -1)
        out, _ = self.dec(torch.cat([dec, s], -1), h)
        return self.head(self.drop(out))


def multitask_forecaster(train_full, val_masked, seeds=(0, 1, 2), aux_w=0.3):
    (Xe, Xd, Xs, Y), (vf, Ve, Vd, Vs) = _prep(train_full, val_masked)
    aux = np.stack([_pivot(train_full, c, PRED_HOURS).to_numpy() * Y_SCALE
                    for c in ("P_t", "N_t")], -1)
    T = [torch.tensor(v, dtype=torch.float32) for v in
         (Xe, Xd, Xs, np.nan_to_num(Y), np.nan_to_num(aux))]
    tv = [torch.tensor(v, dtype=torch.float32) for v in (Ve, Vd, Vs)]
    preds = []
    for seed in seeds:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(Xe))
        n_val = max(12, len(idx) // 8)
        va_i, tr_i = idx[:n_val], idx[n_val:]
        model = MultiTaskNet(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                opt.zero_grad()
                o = model.all_heads(T[0][b], T[1][b], T[2][b])
                loss = (nn.functional.mse_loss(o[:, :, 0], T[3][b])
                        + aux_w * nn.functional.mse_loss(o[:, :, 1:], T[4][b]))
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(model(T[0][va_i], T[1][va_i], T[2][va_i]),
                                           T[3][va_i]).item()
            if v < best - 1e-6:
                best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            preds.append(model(*tv).numpy())
    return _dl_out(vf, np.mean(preds, 0))


# ================================== S38 cross-county attention with more seeds/capacity
def cross_county_5seed_forecaster(train_full, val_masked):
    return cross_county_forecaster(train_full, val_masked, seeds=(0, 1, 2, 3, 4))


# ================================ S39 cross-county attention: self-leak fix (+transductive)
def _wmse_s39(pred, target, w):
    """Per-hour weighted MSE; w=None returns nn.functional.mse_loss itself, so the
    default path is byte-identical to the original loop."""
    if w is None:
        return nn.functional.mse_loss(pred, target)
    tw = torch.tensor(w, dtype=torch.float32)
    return (((pred - target) ** 2) * tw).mean()


def cross_county_fixed_forecaster(train_full, val_masked, seeds=(0, 1, 2),
                                  transductive=False, w=None):
    """Fixes a train/inference mismatch in the original I07: the attention memory
    pool was sampled from the training index without excluding the current batch,
    so a county could retrieve its own encoder output — a shortcut unavailable at
    inference. Here the pool always excludes the batch. With transductive=True the
    inference pool also includes the val counties' observed-window encodings
    (legal: those are hours <= FREEZE_H, which the test file provides).
    """
    from .ideas import CrossCountyNet
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
        model = CrossCountyNet(Xe.shape[-1], Xd.shape[-1], Xs.shape[-1])
        opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
        best, best_state, bad = np.inf, None, 0
        for _ in range(300):
            model.train()
            for b in np.array_split(rng.permutation(tr_i), max(1, len(tr_i) // 32)):
                pool = np.setdiff1d(tr_i, b)                    # never attend to self
                sel = rng.choice(pool, max(8, int(0.7 * len(pool))), replace=False)
                opt.zero_grad()
                _wmse_s39(
                    model(t["Xe"][b], t["Xd"][b], t["Xs"][b], t["Xe"][sel]),
                    t["Y"][b], w).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                v = _wmse_s39(
                    model(t["Xe"][va_i], t["Xd"][va_i], t["Xs"][va_i], t["Xe"][tr_i]),
                    t["Y"][va_i], w).item()
            if v < best - 1e-6:
                best, best_state, bad = v, {k: p.clone() for k, p in model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 30:
                    break
        model.load_state_dict(best_state)
        model.eval()
        mem = torch.cat([t["Xe"], t["Ve"]]) if transductive else t["Xe"]
        with torch.no_grad():
            preds.append(model(t["Ve"], t["Vd"], t["Vs"], mem).numpy())
    return _dl_out(vf, np.mean(preds, 0))


def cross_county_transductive_forecaster(train_full, val_masked):
    return cross_county_fixed_forecaster(train_full, val_masked, transductive=True)


IDEAS3 = {
    "S39 cross-county self-leak fixed": cross_county_fixed_forecaster,
    "S39b cross-county + transductive pool": cross_county_transductive_forecaster,
    "S30 LSTM seq2seq": lstm_forecaster,
    "S31 bidirectional encoder": bienc_forecaster,
    "S32 temporal CNN (TCN)": tcn_forecaster,
    "S33 basis expansion (N-BEATS style)": basis_forecaster,
    "S34 physics-informed net": physics_net_forecaster,
    "S35 mixup augmentation": mixup_forecaster,
    "S36 stochastic weight averaging": swa_forecaster,
    "S37 multi-task (OSI+P+N)": multitask_forecaster,
    "S38 cross-county 5 seeds": cross_county_5seed_forecaster,
}
