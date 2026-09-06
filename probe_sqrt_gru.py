"""Probe: sqrt-target GRU (S24b's winning transform applied to the sequence model)."""
import numpy as np, pandas as pd, torch, time
from src.data import load_train, mask_after_freeze, PRED_HOURS
from src.validate import run_cv
from src.dl import _tensors, _fit_one, Y_SCALE

YS = 10.0  # sqrt(osi) is O(0.04); scale for healthy gradients

def sqrt_gru_forecaster(train_full, val_masked, seeds=(0,1,2)):
    _, Xe, Xd, Xs = _tensors(mask_after_freeze(train_full))
    truth = train_full.pivot_table(index="fipsCode", columns="hour", values="osi")
    Y = np.sqrt(np.clip(truth.sort_index().reindex(columns=PRED_HOURS).to_numpy(),0,None)) * YS
    def ns(a):
        f=a.reshape(-1,a.shape[-1]); return f.mean(0), f.std(0)+1e-6
    stats=[ns(x) for x in (Xe,Xd,Xs)]
    Xe,Xd,Xs=[(x-m)/s for x,(m,s) in zip((Xe,Xd,Xs),stats)]
    vf,Ve,Vd,Vs=_tensors(val_masked)
    Ve,Vd,Vs=[(x-m)/s for x,(m,s) in zip((Ve,Vd,Vs),stats)]
    tv=[torch.tensor(v,dtype=torch.float32) for v in (Ve,Vd,Vs)]
    preds=[]
    for sd in seeds:
        m=_fit_one(Xe,Xd,Xs,Y,sd)
        with torch.no_grad(): preds.append(m(*tv).numpy())
    P=np.clip(np.mean(preds,0)/YS,0,None)**2
    return pd.DataFrame({"fipsCode":np.repeat(vf,len(PRED_HOURS)),
                         "hour":np.tile(PRED_HOURS,len(vf)),"osi_pred":P.ravel()})

if __name__=="__main__":
    tr=load_train(); t0=time.time()
    tab,pooled,folds=run_cv(sqrt_gru_forecaster,tr,per_fold=True)
    print(tab.round(5).to_string())
    print("folds:"," ".join(f"{f.loc['mean','rmse']:.5f}" for f in folds), f"({time.time()-t0:.0f}s)")
    pooled.to_csv("/tmp/oof_sqrtgru.csv",index=False)
