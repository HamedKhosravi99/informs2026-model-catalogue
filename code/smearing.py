"""Duan's smearing estimator on the sqrt-target tabular member.

f2 fits squared error on sqrt(OSI) and squares the prediction back. By Jensen,
E[y|x] > (E[sqrt(y)|x])^2, so squaring back estimates something below the
conditional mean -- which is the functional RMSE actually rewards. Measured
bias: E[pred]/E[truth] = 0.841.

The classical fix (Duan 1983) is nonparametric: back-transform the fitted value
*plus each training residual* and average.  For g(y)=sqrt(y):

    yhat = mean_i ( shat + r_i )^2  =  shat^2 + 2*shat*mean(r) + mean(r^2)

shat is recovered exactly from the cached predictions, since f2 stores
clip(shat,0)^2.  Every correction below is fitted on four folds and applied to
the fifth, and each is tested twice: on f2 alone, and with corrected-f2 swapped
into the 11-member ensemble.
"""
import numpy as np
import pandas as pd

from canonical import (MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, load, truth,
                       TAIL_SPLIT_HOUR, TAIL_SCALE)
from src.combine import ampshape
from src.data import load_train
from src.validate import make_folds

HZ = (1, 6, 24, 48)


def main():
    t = truth(); train = load_train()
    fold_of = {c: k for k, (_, va) in enumerate(make_folds(train, 5)) for c in va}

    f2 = load("f2").set_index(["fipsCode", "hour"])["osi_pred"]
    y = np.array([t.loc[a, b] if b in t.columns else np.nan for a, b in f2.index])
    d = pd.DataFrame({"f": f2.index.get_level_values(0), "h": f2.index.get_level_values(1),
                      "pred": f2.to_numpy(), "y": y}).dropna().reset_index(drop=True)
    d["fold"] = d.f.map(fold_of)
    d["shat"] = np.sqrt(d.pred)                       # exact: f2 stores clip(shat,0)^2
    d["sy"] = np.sqrt(d.y)
    d["r"] = d.sy - d.shat                            # residual on the sqrt scale

    def solo(p, m=None):
        m = np.ones(len(d), bool) if m is None else m
        return float(np.mean([np.sqrt(((p - d.y) ** 2)[m & (d.h >= 72 + k)].mean()) for k in HZ]))

    base = solo(d.pred.to_numpy())
    print(f"f2 as shipped: RMSE {base:.6f}   E[pred]/E[truth] {d.pred.mean() / d.y.mean():.3f}")
    print(f"  residual on the sqrt scale: mean {d.r.mean():+.4f}, var {d.r.var():.4f}\n")

    # ---- four corrections, each nested (fit on 4 folds, apply to the 5th)
    def duan(tr, te):
        r = d.r[tr]
        return d.shat[te] ** 2 + 2 * d.shat[te] * r.mean() + (r ** 2).mean()

    def duan_binned(tr, te, nb=10):
        q = np.quantile(d.shat[tr], np.linspace(0, 1, nb + 1))[1:-1]
        b_tr, b_te = np.digitize(d.shat[tr], q), np.digitize(d.shat[te], q)
        out = d.shat[te].to_numpy() ** 2 + 0.0
        for b in range(nb):
            mt, me = b_tr == b, b_te == b
            if mt.sum() < 20 or me.sum() == 0:
                continue
            r = d.r[tr].to_numpy()[mt]
            out[me] = d.shat[te].to_numpy()[me] ** 2 + 2 * d.shat[te].to_numpy()[me] * r.mean() + (r ** 2).mean()
        return pd.Series(out, index=d.index[te])

    def mult(tr, te):
        c = (d.pred[tr] * d.y[tr]).sum() / max((d.pred[tr] ** 2).sum(), 1e-12)
        return d.pred[te] * c

    def addvar(tr, te):
        return d.shat[te] ** 2 + d.r[tr].var()

    CORR = {"Duan smearing (global)": duan, "Duan smearing (10 shat bins)": duan_binned,
            "multiplicative c (least squares)": mult, "additive residual variance": addvar}
    corrected = {}
    print(f"{'correction':34} {'f2 RMSE':>10} {'vs base':>10} {'E[p]/E[y]':>10}")
    for name, fn in CORR.items():
        out = np.zeros(len(d))
        for k in range(5):
            tr, te = (d.fold != k).to_numpy(), (d.fold == k).to_numpy()
            out[te] = np.clip(np.asarray(fn(tr, te)), 0, None)
        corrected[name] = out
        print(f"  {name:32} {solo(out):10.6f} {solo(out) - base:+10.6f} {out.mean() / d.y.mean():10.3f}")

    # ---- does any of it help the ensemble?
    fz = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
    P = {m: load(m).set_index(["fipsCode", "hour"])["osi_pred"] for m in fz + ["a3x25"]}
    idx = P["a3x25"].index
    for m in fz:
        idx = idx.intersection(P[m].index)
    key = pd.MultiIndex.from_arrays([d.f, d.h])
    ff = idx.get_level_values(0).to_numpy(); hh = idx.get_level_values(1).to_numpy()
    Y = np.array([t.loc[a, b] if b in t.columns else np.nan for a, b in idx]); ok = np.isfinite(Y)
    fold = np.array([fold_of[c] for c in ff])
    tail = np.where(hh >= TAIL_SPLIT_HOUR, TAIL_SCALE, 1.0)
    V = {m: P[m].reindex(idx) for m in fz}; A = P["a3x25"].reindex(idx).to_numpy()

    def blend(f2_series):
        cols = [f2_series if m == "f2" else V[m] for m in fz]
        return (0.5 * ampshape(cols, idx).to_numpy() + 0.5 * A) * tail

    def emr(pred, m): return float(np.mean([np.sqrt(((pred - Y) ** 2)[m & ok & (hh >= 72 + k)].mean()) for k in HZ]))
    B = blend(V["f2"]); allm = np.ones(len(Y), bool); r0 = emr(B, allm)
    fips = np.array(sorted(set(ff))); rng = np.random.default_rng(42)
    bidx = rng.integers(0, len(fips), size=(3000, len(fips))); fi = pd.Series(ff)
    print(f"\nIN THE 11-MEMBER ENSEMBLE (base {r0:.6f})")
    print(f"{'correction':34} {'naive':>10} {'nested':>10} {'adopt':>6}  P(better) per horizon")
    for name, out in corrected.items():
        s = pd.Series(out, index=key).reindex(idx)
        pred = blend(s.fillna(V["f2"]))
        num = {k: [0.0, 0] for k in HZ}; adopts = 0
        for k in range(5):
            tr, te = fold != k, fold == k
            ad = emr(pred, tr) < emr(B, tr); adopts += ad; D = pred if ad else B
            for kk in HZ:
                mm = te & ok & (hh >= 72 + kk)
                num[kk][0] += ((D - Y) ** 2)[mm].sum(); num[kk][1] += mm.sum()
        nested = float(np.mean([np.sqrt(v[0] / v[1]) for v in num.values()]))
        ps = []
        for k in HZ:
            mm = ok & (hh >= 72 + k)
            cnt = pd.Series(mm.astype(int)).groupby(fi.to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            ga = pd.Series(((pred - Y) ** 2)[mm]).groupby(fi[mm].to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            gb = pd.Series(((B - Y) ** 2)[mm]).groupby(fi[mm].to_numpy()).sum().reindex(fips).fillna(0).to_numpy()
            ps.append((np.sqrt(ga[bidx].sum(1) / cnt[bidx].sum(1)) < np.sqrt(gb[bidx].sum(1) / cnt[bidx].sum(1))).mean())
        print(f"  {name:32} {emr(pred, allm):10.6f} {nested:10.6f} {adopts:>3}/5  " + "  ".join(f"{p:.2f}" for p in ps))


if __name__ == "__main__":
    main()
