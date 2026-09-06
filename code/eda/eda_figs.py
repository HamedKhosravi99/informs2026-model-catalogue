# EDA figures for the INFORMS 2026 DM Data Challenge (light-mode, validated palette)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = Path(__file__).resolve().parent / "figs"
FIGS.mkdir(exist_ok=True)

# palette roles (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"   # series 1
ORANGE = "#eb6834" # series 2

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

tr = pd.read_csv(ROOT / "DM_Train.csv", parse_dates=["timestamp_et"])
t0 = tr.timestamp_et.min()
tr["hour"] = ((tr.timestamp_et - t0).dt.total_seconds() // 3600).astype(int)

# ---------------------------------------------------------------- fig 1: timeline
hourly = tr.groupby("timestamp_et").agg(osi=("osi", "mean"), gust=("gust", "mean"))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12})
cut = t0 + pd.Timedelta(hours=72)
for ax in (ax1, ax2):
    ax.axvspan(t0, cut, color="#f0efec", zorder=0)
    ax.grid(axis="x", visible=False)
ax1.plot(hourly.index, hourly.osi, color=BLUE, lw=1.8)
ax1.set_ylabel("mean OSI (239 train counties)")
ax1.set_ylim(0, None)
ax1.annotate("observed in test\n(Mar 11–13)", xy=(t0 + pd.Timedelta(hours=8), 0.034),
             fontsize=9, color=INK2)
ax1.annotate("prediction window (Mar 14–19)", xy=(cut + pd.Timedelta(hours=6), 0.034),
             fontsize=9, color=INK2)
ax1.annotate("wave 1 peak\nh67, OSI 0.037", xy=(t0 + pd.Timedelta(hours=67), 0.0369),
             xytext=(t0 + pd.Timedelta(hours=30), 0.026), fontsize=8.5, color=INK2,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax1.annotate("wave 2 hump\nh112, OSI 0.009", xy=(t0 + pd.Timedelta(hours=112), 0.0089),
             xytext=(t0 + pd.Timedelta(hours=128), 0.017), fontsize=8.5, color=INK2,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax2.plot(hourly.index, hourly.gust, color=ORANGE, lw=1.8)
ax2.set_ylabel("mean gust (mph)")
ax2.set_ylim(0, None)
ax2.annotate("wave 2 gusts rival wave 1,\nbut cause far fewer outages", xy=(t0 + pd.Timedelta(hours=111), 35.7),
             xytext=(t0 + pd.Timedelta(hours=140), 36), fontsize=8.5, color=INK2,
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax2.xaxis.set_major_locator(mdates.DayLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter("Mar %d"))
fig.suptitle("Two-storm event: outage severity vs wind, March 11–19, 2026", x=0.02, ha="left", fontsize=12)
fig.savefig(FIGS / "01_timeline.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------- fig 2: wave response asymmetry (dot plot)
bins = [0, 20, 30, 40, 50, 100]
labels = ["0–20", "20–30", "30–40", "40–50", "50+"]
rows = []
for name, lo, hi in [("wave 1", 48, 96), ("wave 2", 120, 168)]:
    w = tr[(tr.hour >= lo) & (tr.hour < hi)]
    m = w.groupby(pd.cut(w.gust, bins, right=False, labels=labels), observed=True).N_t.mean()
    rows.append(m.rename(name))
asym = pd.concat(rows, axis=1)

fig, ax = plt.subplots(figsize=(7, 3.6))
y = np.arange(len(asym))
ax.hlines(y, asym["wave 2"], asym["wave 1"], color=GRID, lw=1.5, zorder=1)
ax.scatter(asym["wave 1"], y, s=70, color=BLUE, zorder=2, label="wave 1 (Mar 13–14)")
ax.scatter(asym["wave 2"], y, s=70, color=ORANGE, zorder=2, label="wave 2 (Mar 16–17)")
ax.set_yticks(y, [f"{l} mph" for l in asym.index])
ax.set_xlabel("mean new-outage rate N_t (per hour)")
ax.set_xlim(0, None)
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.annotate("same gusts, ~6× less damage\nin the second wave", xy=(asym.loc["40–50", "wave 2"], 3),
            xytext=(0.011, 2.1), fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.set_title("Damage response to wind collapses after the first wave", loc="left", fontsize=12, color=INK)
fig.savefig(FIGS / "02_wave_asymmetry.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------- fig 3: county x hour heatmap
piv = tr.pivot_table(index="fipsCode", columns="hour", values="osi")
order = piv.max(axis=1).sort_values(ascending=False).index
piv = piv.loc[order]
seq = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(9, 4.6))
im = ax.imshow(piv.values, aspect="auto", cmap=seq, norm=PowerNorm(0.35, vmin=0, vmax=0.65),
               interpolation="nearest")
ax.axvline(71.5, color=INK, lw=1.0, ls=(0, (4, 3)))
ax.text(74, 232, "last hour observed in test → forecast from here", fontsize=8.5, color=INK)
ax.set_xlabel("hour (Mar 11 00:00 = 0, ET)")
ax.set_ylabel("239 train counties, sorted by peak OSI")
ax.set_yticks([])
ax.grid(visible=False)
day_ticks = np.arange(0, 216, 24)
ax.set_xticks(day_ticks, [f"Mar {11 + d // 24}" for d in day_ticks], fontsize=8.5)
cb = fig.colorbar(im, ax=ax, pad=0.01, ticks=[0, 0.01, 0.05, 0.15, 0.35, 0.65])
cb.set_label("OSI (power-scaled color)", color=INK2, fontsize=9)
cb.ax.tick_params(labelsize=8, color=MUTED, labelcolor=MUTED)
cb.outline.set_edgecolor(BASELINE)
ax.set_title("County-level severity: two waves, heavy tail, most counties near zero",
             loc="left", fontsize=12, color=INK)
fig.savefig(FIGS / "03_county_heatmap.png", bbox_inches="tight")
plt.close(fig)

print("wrote:", *[p.name for p in sorted(FIGS.glob("*.png"))])
