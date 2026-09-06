"""Chronos-2 (amazon/chronos-2, 120M) zero-shot with known-future weather covariates.

Pass --uni for the univariate ablation (no covariates; outputs suffixed _uni).

Measured on the 239 training counties. Zero-shot means the model has never seen
any of them, so scoring on training counties is legitimate and, if anything,
conservative relative to a fine-tuned model. Output goes to oof/chronos2.csv in
the project's standard (fipsCode, hour, osi_pred) layout so the main
environment's official scorer can grade it.

NOT a submission candidate: a pretrained checkpoint is external data of a kind
the rules' "self-contained code on the provided files" clause does not admit,
and it needs torch >= 2.x against the project's pinned 1.12. This is a
measurement for the report.
"""
import sys, time
import numpy as np, pandas as pd, torch
torch.set_num_threads(4)
from chronos import Chronos2Pipeline

ROOT = "/Users/hkhosravi7/Documents/GitHub/DM_compettion"
WEATHER = ["gust", "wind_speed_10m", "wind_dir_10m", "t2m", "d2m", "sp", "mslma",
           "blh", "tp", "rain", "csnow", "sdwe", "tcc", "lcc", "mcc", "hcc",
           "sdswrf", "direct_rad", "diffuse_rad", "r2", "vpd", "et0", "soil_moist"]
FREEZE_H, LAST = 71, 215

df = pd.read_csv(f"{ROOT}/DM_Train.csv", parse_dates=["timestamp_et"])
t0 = pd.Timestamp("2026-03-11 00:00:00")
df["hour"] = ((df["timestamp_et"] - t0).dt.total_seconds() // 3600).astype(int)
df = df.sort_values(["fipsCode", "hour"])
cov = [] if "--uni" in sys.argv else [c for c in WEATHER if c in df.columns]
print("counties", df.fipsCode.nunique(), "covariates", len(cov))

ctx = df[df.hour <= FREEZE_H][["fipsCode", "timestamp_et", "osi"] + cov].rename(
    columns={"fipsCode": "id", "timestamp_et": "timestamp", "osi": "target"})
fut = None if not cov else df[(df.hour > FREEZE_H) & (df.hour <= LAST)][["fipsCode", "timestamp_et"] + cov].rename(
    columns={"fipsCode": "id", "timestamp_et": "timestamp"})
# Chronos wants a clean regular index with no NaN covariates
if cov: ctx[cov] = ctx[cov].ffill().fillna(0.0); fut[cov] = fut[cov].ffill().fillna(0.0)
plen = LAST - FREEZE_H          # 144 steps: hours 72..215

pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
t1 = time.time()
Q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pred = pipe.predict_df(ctx, future_df=fut, prediction_length=plen, quantile_levels=Q,
                       id_column="id", timestamp_column="timestamp", target="target")
print(f"predicted in {time.time()-t1:.0f}s; columns: {list(pred.columns)[:8]}...")
pred["hour"] = ((pd.to_datetime(pred["timestamp"]) - t0).dt.total_seconds() // 3600).astype(int)
qcols = [c for c in pred.columns if c in [str(q) for q in Q] or c in Q]
pred["mean_est"] = pred[qcols].astype(float).mean(axis=1)     # quantile-average ~ E[y]
pred["median"] = pred[[c for c in qcols if str(c) == "0.5"][0]].astype(float)
out = pred[pred.hour >= FREEZE_H + 2].rename(columns={"id": "fipsCode"})
for name, col in (("chronos2" + ("_uni" if not cov else ""), "mean_est"), ("chronos2_med" + ("_uni" if not cov else ""), "median")):
    o = out[["fipsCode", "hour", col]].rename(columns={col: "osi_pred"}).copy()
    o["osi_pred"] = o["osi_pred"].clip(lower=0.0)
    o.to_csv(f"{ROOT}/oof/{name}.csv", index=False)
    print("wrote", name, len(o), "rows")
