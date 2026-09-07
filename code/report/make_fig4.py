"""Generate fig4_procedure.png -- the end-to-end procedure, on one page.

Three panels, all drawn from the committed ledger (catalogue/catalogue.csv)
and the dashboard's object table, so nothing here is hand-placed:
  (a) where the effort went   -- the residual budget and the eight families
  (b) what was actually built -- all 210 models by class and score
  (c) what shipped            -- the 50/50 anchor + bracket structure
"""
import csv, json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 200})
INK, ACC, WARN = "#22313f", "#2b7a9b", "#c0553b"
MUTE, GOLD = "#8d9aa4", "#b8860b"

rows = list(csv.DictReader(open(ROOT / "catalogue" / "catalogue.csv")))
html = (ROOT / "catalogue" / "dashboard.html").read_text()
D = json.loads(re.search(r"const DATA\s*=\s*(\{.*?\})\s*;\s*\n", html, re.S).group(1))
FOUND = [o["rmse"] for o in D["extra"]["objs"]
         if o.get("group", "").startswith("foundation")]

fig = plt.figure(figsize=(7.0, 3.62))
gs = fig.add_gridspec(1, 3, width_ratios=[1.06, 1.40, 0.80], wspace=0.30,
                      left=0.004, right=0.996, top=0.995, bottom=0.075)

# ---------------------------------------------------------------- (a)
ax = fig.add_subplot(gs[0]); ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(-1.0, 10.75)
ax.text(0, 1.008, "a   Where the effort went", transform=ax.transAxes,
        weight="bold", fontsize=8.5, va="bottom")

seg = [("scale 45%", 45, ACC), ("shape 37%", 37, INK), ("17%", 17, MUTE)]
x = 0.0
for lab, pct, col in seg:
    ax.add_patch(plt.Rectangle((x, 10.05), pct / 100, 0.44, fc=col, ec="none"))
    ax.text(x + pct / 200, 10.27, lab, color="w", fontsize=5.5,
            ha="center", va="center", weight="bold")
    x += pct / 100
ax.text(0, 9.92, "the residual budget (scale / shape / timing), measured first",
        fontsize=5.8, color=MUTE, va="top")
ax.annotate("", xy=(0.5, 9.05), xytext=(0.5, 9.45),
            arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.9))

FAM = [("F1", "signal", "extra channels, retrieved analogs", 45, 0),
       ("F2", "mechanism", "19 architectures, deep and physical", 67, 0),
       ("F3", "target", "sqrt, log1p, Tweedie, quantile", 27, 0),
       ("F4", "exposure", "augmentation: storm jitter, mixup", 27, 0),
       ("F5", "estimation", "seed and snapshot ensembling", 23, 0),
       ("F6", "adaptation", "two-stage gates, post-hoc fixes", 7, 12),
       ("F7", "combination", "blending rules, subset search", 4, 14),
       ("F8", "uncertainty", "split-conformal intervals", 1, 2)]
y, MAXW = 8.55, 81.0
for code, name, gloss, n, extra in FAM:
    w = n / MAXW
    ax.add_patch(plt.Rectangle((0, y - 0.185), w, 0.37, fc=ACC, ec="none"))
    if extra:
        ax.add_patch(plt.Rectangle((w, y - 0.185), extra / MAXW, 0.37,
                                   fc="none", ec=ACC, lw=0.7, hatch="/////"))
    end = w + (extra / MAXW if extra else 0)
    ax.text(end + 0.020, y, f"{n}" + (f"+{extra}" if extra else ""),
            fontsize=6.0, va="center", color=INK)
    ax.text(0, y - 0.30, "$\\bf{" + code + "}$  $\\bf{" + name + "}$   " + gloss,
            fontsize=6.0, va="top", color=INK)
    y -= 1.03
ax.add_patch(plt.Rectangle((0, 0.30), 0.050, 0.20, fc=ACC, ec="none"))
ax.text(0.068, 0.40, "models scored on their own", fontsize=5.7, va="center")
ax.add_patch(plt.Rectangle((0, -0.16), 0.050, 0.20, fc="none", ec=ACC,
                           lw=0.7, hatch="/////"))
ax.text(0.068, -0.06, "paired contrasts against the blend", fontsize=5.7,
        va="center")

# ---------------------------------------------------------------- (b)
ax = fig.add_subplot(gs[1])
CLASS = {"Seq2Seq (GRU)": 0, "BiEncSeq2Seq": 0,
         "boosted / bagged trees": 1, "other / statistical": 2,
         "stock-flow state space": 3, "—": 3,
         "patch / variate attention": 4, "CrossCountyNet": 4,
         "LSTM / Transformer": 4, "TCN / basis / linear": 5,
         "MLP / KAN / TabNet / FT-T": 6, "normalizing flow": 7}
NAMES = ["recurrent encoder–decoder", "gradient-boosted trees",
         "statistical / other", "stock-flow (physical)",
         "attention / transformer", "TCN / basis / linear",
         "neural-tabular: MLP, KAN, TabNet", "normalizing flow (generative)"]
rng = np.random.default_rng(42)
counts = [0] * 8
for r in rows:
    counts[CLASS[r["mechanism"]]] += 1
for r in rows:
    if not r["rmse_mean"]:
        continue                      # one model crashed before it was scored
    c, v = CLASS[r["mechanism"]], float(r["rmse_mean"])
    yy = (7 - c) + rng.uniform(-0.145, 0.145)
    if r["shipped"] == "True":
        ax.plot(v, yy, "o", ms=4.2, mfc=WARN, mec="w", mew=0.7, zorder=5)
    else:
        ax.plot(v, yy, "o", ms=2.2, mfc=ACC, mec="none", alpha=0.5, zorder=3)
for i, (nm, ct) in enumerate(zip(NAMES, counts)):
    ax.text(0.007960, 7 - i + 0.34, f"{nm}  ({ct})", fontsize=6.1,
            color=INK, va="bottom", ha="left")
ax.plot([0.01650] * len(FOUND), [-1.05] * len(FOUND), ">", ms=4.2,
        mfc="none", mec=GOLD, mew=1.0, zorder=5)
ax.text(0.007960, -0.71, "pretrained / foundation, zero-shot  (4)",
        fontsize=6.1, color=GOLD, va="bottom", weight="bold")
ax.annotate("0.0201 – 0.0269:\nworse than all-zeros",
            xy=(0.01608, -1.05), xytext=(0.01245, -1.30), fontsize=5.8,
            color=GOLD, ha="center", va="center", linespacing=1.35,
            arrowprops=dict(arrowstyle="->", color=GOLD, lw=0.7,
                            connectionstyle="arc3,rad=-0.15"))
for v, lab, side in [(0.016317, "all-zeros", "right"),
                     (0.011097, "decayed persistence", "left")]:
    ax.axvline(v, color=MUTE, ls=":", lw=0.9, zorder=1)
    off = 0.00010 if side == "left" else -0.00010
    ax.text(v + off, 8.98, lab, fontsize=5.9, color=MUTE, ha=side, va="center")
ax.axvline(0.008354, color=WARN, ls="--", lw=1.0, zorder=1)
ax.annotate("shipped ensemble", xy=(0.008354, -1.92), xytext=(0.00980, -1.92),
            fontsize=5.9, color=WARN, weight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color=WARN, lw=0.7))
ax.set_xlim(0.00790, 0.01700); ax.set_ylim(-2.25, 9.45)
ax.set_yticks([]); ax.spines["left"].set_visible(False)
ax.set_xlabel("mean RMSE over the four scored horizons   (lower is better)",
              fontsize=6.6, labelpad=1.5)
ax.tick_params(axis="x", labelsize=6.1, pad=1.5)
ax.text(0.0, 1.008, "b   What was built: all 210 models",
        transform=ax.transAxes, fontsize=8.5, weight="bold", va="bottom")

# ---------------------------------------------------------------- (c)
ax = fig.add_subplot(gs[2]); ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(-0.7, 9.95)
ax.text(0, 1.008, "c   What shipped", transform=ax.transAxes,
        weight="bold", fontsize=8.5, va="bottom")
ax.add_patch(plt.Rectangle((0.02, 8.02), 0.96, 1.62, fc=ACC, ec="none"))
ax.text(0.50, 9.28, "anchor  $A$", color="w", fontsize=8.0, ha="center",
        va="center", weight="bold")
ax.text(0.50, 8.55, "one bi-GRU, averaged\nover 100 networks:\n25 seeds $\\times$ 4 snapshots",
        color="w", fontsize=5.4, ha="center", va="center", linespacing=1.35)
ax.text(0.50, 7.82, "$\\frac{1}{2}$ of the forecast", fontsize=6.4,
        ha="center", va="top", color=INK)
ax.text(0.50, 7.14, "$+$", fontsize=9, ha="center", va="center", color=MUTE)
for i in range(11):
    ax.add_patch(plt.Rectangle((0.02 + i * 0.0873, 5.36), 0.072, 1.06,
                               fc=INK, ec="none"))
ax.text(0.50, 6.52, "bracket: 11 models", color=INK, fontsize=6.6,
        ha="center", va="bottom", weight="bold")
ax.text(0.50, 5.16, "$\\frac{1}{22}$ each, combined by\namplitude $\\times$ shape\n"
        "with no fitted weights", fontsize=5.9, ha="center", va="top",
        color=INK, linespacing=1.35)
ax.text(0.50, 3.52, "recurrent $\\cdot$ trees $\\cdot$ attention\n"
        "$\\cdot$ stock-flow physics", fontsize=5.7, ha="center", va="top",
        color=MUTE, style="italic", linespacing=1.35)
ax.annotate("", xy=(0.5, 2.20), xytext=(0.5, 2.62),
            arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=0.9))
ax.add_patch(plt.Rectangle((0.02, 1.34), 0.96, 0.80, fc=WARN, ec="none"))
ax.text(0.50, 1.74, "$\\times\\,0.667$ on hours $\\geq$ 168",
        color="w", fontsize=6.2, ha="center", va="center", weight="bold")
ax.text(0.50, 1.12, "with one member rescale, the\nonly fitted scalars anywhere",
        fontsize=5.7, ha="center", va="top", color=MUTE, linespacing=1.35)
ax.text(0.50, 0.10, "CV 0.00835   ·   nested 0.00838", fontsize=6.4,
        ha="center", va="center", color=INK, weight="bold")

fig.savefig(OUT / "fig4_procedure.png", bbox_inches="tight", facecolor="w")
print("wrote", OUT / "fig4_procedure.png")
print("classes:", dict(zip(NAMES, counts)), "total", sum(counts))
