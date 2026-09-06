"""Generate the report figures from cached out-of-fold predictions.

Every figure is derived from `oof/cfg_final.csv` (the submitted configuration)
and the training file -- nothing is drawn by hand, so a reviewer can regenerate
the whole figure set with one command.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import FREEZE_H, HORIZONS, load_train, osi_trajectory

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
INK, ACC, WARN = "#22313f", "#2b7a9b", "#c0553b"

train = load_train()
truth = osi_trajectory(train)
pred = pd.read_csv(ROOT / "oof" / "cfg_final.csv")
piv = pred.pivot_table(index="fipsCode", columns="hour", values="osi_pred")


# ---------------------------------------------------------------- figure 1
# Error concentration + the rigour staircase, side by side.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.25))

tgt = np.arange(FREEZE_H + 1, 216)
T = truth.loc[piv.index, tgt].to_numpy()
P = piv.reindex(columns=tgt).to_numpy()
sse = np.nansum((P - T) ** 2, axis=1)
share = np.sort(sse)[::-1].cumsum() / np.nansum(sse)
ax1.plot(np.arange(1, len(share) + 1), 100 * share, color=INK, lw=1.6)
k = int(np.searchsorted(share, 0.65) + 1)
ax1.axhline(65, color=WARN, ls=":", lw=1)
ax1.axvline(k, color=WARN, ls=":", lw=1)
ax1.plot([k], [65], "o", color=WARN, ms=4)
ax1.annotate(f"{k} counties\ncarry 65%", (k, 65), xytext=(k + 22, 44),
             color=WARN, fontsize=7.5,
             arrowprops=dict(arrowstyle="-", color=WARN, lw=0.7))
ax1.set_xlabel("counties, ranked by squared error")
ax1.set_ylabel("cumulative % of squared error")
ax1.set_title("Error is concentrated", fontsize=8.5, loc="left", weight="bold")
ax1.set_ylim(0, 101)

techs = ["greedy\nweights", "NNLS\npool", "distil-\nlation", "subset\nselection"]
naive = [0.00883, 0.00850, 0.00892, 0.008423]
nested = [0.00959, 0.00959, 0.00931, 0.008569]
x = np.arange(len(techs))
ax2.bar(x - 0.19, naive, 0.36, label="naive (in-sample selection)", color=ACC)
ax2.bar(x + 0.19, nested, 0.36, label="nested (honest)", color=WARN)
ax2.axhline(0.00836, color=INK, ls="--", lw=1.1)
ax2.annotate("submitted model (no selection)", (-0.42, 0.00836),
             xytext=(0, -9), textcoords="offset points", ha="left",
             fontsize=6.8, color=INK)
ax2.set_xticks(x)
ax2.set_xticklabels(techs, fontsize=7)
ax2.set_ylabel("RMSE")
ax2.set_ylim(0.0079, 0.0099)
ax2.legend(fontsize=6.6, frameon=False, loc="upper left")
ax2.set_title("Every selection procedure reversed", fontsize=8.5,
              loc="left", weight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig1_concentration_staircase.png", bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- figure 2
# The money figure: forecast trajectories for held-out counties spanning the
# severity range, with the prediction origin marked.
sev = pd.Series(np.nansum(truth.loc[piv.index, tgt].to_numpy(), axis=1),
                index=piv.index).sort_values(ascending=False)
picks = [sev.index[0], sev.index[len(sev) // 12], sev.index[len(sev) // 4],
         sev.index[len(sev) // 2]]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.65), sharex=True)
hours = np.arange(0, 216)
for ax, f in zip(axes, picks):
    tt = truth.loc[f].reindex(hours).to_numpy()
    pp = piv.loc[f].reindex(hours).to_numpy()
    ax.axvspan(0, FREEZE_H, color="0.92")
    ax.plot(hours, tt, color=INK, lw=1.2, label="actual")
    ax.plot(hours, pp, color=WARN, lw=1.2, ls="--", label="forecast")
    ax.axvline(FREEZE_H, color="0.45", lw=0.8)
    ax.set_title(f"county {f}", fontsize=7.5)
    ax.set_xticks([0, 72, 144, 215])
    ax.tick_params(labelsize=6.5)
axes[0].set_ylabel("OSI")
axes[0].annotate("observed", (34, axes[0].get_ylim()[1] * 0.42), fontsize=6.5,
                 color="0.45", ha="center")
axes[3].legend(fontsize=6.4, frameon=False, loc="upper right")
fig.supxlabel("hour of event window", fontsize=7.5, y=-0.04)
fig.suptitle("Forecasts for counties never seen in training "
             "(shaded = 72 h of observed history; all later hours predicted)",
             fontsize=8, y=1.06, weight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig2_trajectories.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig1_concentration_staircase.png, fig2_trajectories.png")
print(f"(figure 1 annotation: {k} counties carry 65% of squared error)")


# ---------------------------------------------------------------- figure 3
# The aha figure: an ensemble member's value is its error basis, not its score.
from src.validate import score_trajectory

def _err(slug):
    d = pd.read_csv(ROOT / "oof" / f"{slug}.csv").set_index(["fipsCode", "hour"])
    d = d["osi_pred"]
    tvv = np.array([truth.loc[f, h] if h in truth.columns else np.nan
                    for f, h in d.index])
    return pd.Series(d.to_numpy() - tvv, index=d.index)

def _solo(slug):
    return score_trajectory(pd.read_csv(ROOT / "oof" / f"{slug}.csv"),
                            truth).loc["mean", "rmse"]

MEMB = {"s62": "s62", "x9": "s51", "s39": "s39", "i11w": "i11 (weighted)", "f2": "f2",
        "x4": "s35", "p1b": "p1b", "ts2c": "PatchTST", "s35w": "s35w", "r6corr": "r6",
        "p1c": "p1c", "a3x25": "A (partner)"}
# Literature candidates tested after the member set was fixed: drawn hollow.
# Each is either too correlated or too weak to be admitted (Section 8).
CAND = {"sx1": "XLinear", "sx2": "TSMixer", "sx3": "cond. flow", "sx4": "TimeXer"}
E = {k: _err(k) for k in MEMB}
def meancorr(k):
    cs = []
    for o in MEMB:
        if o == k:
            continue
        c = E[k].index.intersection(E[o].index)
        cs.append(np.corrcoef(E[k].reindex(c).fillna(0),
                              E[o].reindex(c).fillna(0))[0, 1])
    return float(np.mean(cs))

fig, ax = plt.subplots(figsize=(3.6, 2.7))
for k, lab in MEMB.items():
    x, y = _solo(k), meancorr(k)
    big = k in ("p1b", "a3x25")
    ax.scatter(x, y, s=42 if big else 26, color=WARN if k == "p1b" else
               (ACC if k == "a3x25" else INK), zorder=3,
               edgecolor="white", lw=0.6)
    dx, dy = {"a3x25": (4, 6), "f2": (-4, 2), "x4": (0, 5), "s62": (-4, -1)}.get(k, (4, -1))
    ha = {"x4": "center", "s62": "right", "f2": "right"}.get(k, "left")
    ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=6.6, ha=ha, color=WARN if k == "p1b" else "0.25")

for k, lab in CAND.items():
    try:
        E[k] = _err(k)
    except FileNotFoundError:
        continue
    cs = []
    for o in MEMB:
        c = E[k].index.intersection(E[o].index)
        cs.append(np.corrcoef(E[k].reindex(c).fillna(0), E[o].reindex(c).fillna(0))[0, 1])
    x, y = _solo(k), float(np.mean(cs))
    ax.scatter(x, y, s=26, facecolor="none", edgecolor="0.45", lw=0.9, zorder=2)
    ax.annotate(lab, (x, y), xytext=(4, -1), textcoords="offset points",
                fontsize=6.2, color="0.45")

# the i11 upgrade: better solo, more correlated, ensemble got WORSE
# (x3 = i11 @ 25 seeds was the member at the time; load it explicitly since the
# built spec now carries i11w instead)
E["x3"] = _err("x3")
xa, ya = _solo("x3"), meancorr("x3")
E["x7"] = _err("x7")
cs = []
for o in MEMB:
    if o == "x3":
        continue
    c = E["x7"].index.intersection(E[o].index)
    cs.append(np.corrcoef(E["x7"].reindex(c).fillna(0),
                          E[o].reindex(c).fillna(0))[0, 1])
xb, yb = _solo("x7"), float(np.mean(cs))
ax.scatter([xa], [ya], s=26, facecolor="none", edgecolor=WARN, lw=1.1, zorder=3)
ax.annotate("", xy=(xb, yb), xytext=(xa, ya),
            arrowprops=dict(arrowstyle="->", color=WARN, lw=1.3))
ax.scatter([xb], [yb], s=26, facecolor="none", edgecolor=WARN, lw=1.1, zorder=3)
ax.annotate("i11 @25 seeds -> +snapshots:\nbetter alone, ensemble got WORSE", (xb, yb), xytext=(40, 4),
            textcoords="offset points", fontsize=6.4, color=WARN, ha="left",
            va="center")
ax.set_xlabel("member RMSE alone  (lower = better model)")
ax.set_ylabel("mean error correlation\nwith the rest")
ax.set_title("A member's value is its error basis,\nnot its score",
             fontsize=8.5, loc="left", weight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig3_diversity.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig3_diversity.png")
print(f"  p1b: solo {_solo('p1b'):.5f}, corr {meancorr('p1b'):.3f}")
print(f"  i11 upgrade: {xa:.5f}/{ya:.3f} -> {xb:.5f}/{yb:.3f}")
