"""Generate a self-contained HTML dashboard for the experiment catalogue.

Reads the run log and the cached out-of-fold predictions, embeds everything as
JSON in a single file with no external dependencies, and writes it where the
catalogue repository can serve it. Run from the main repository, which is the
only place the out-of-fold predictions live.

Usage: python3 build_dashboard.py [outdir]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT


def diversity_points():
    """Solo RMSE against mean error correlation with the shipped bracket."""
    from canonical import MEMBERS, FREEZE_SUB, FREEZE_ADD, FREEZE_DROP, load, truth
    from src.validate import score_trajectory
    t = truth()
    fz = [FREEZE_SUB.get(m, m) for m in MEMBERS if m not in FREEZE_DROP] + FREEZE_ADD
    label = {"s62": "s62", "x9": "s51+snap", "s39": "s39", "i11w": "i11w", "f2": "f2",
             "x4": "s35@25", "p1b": "p1b", "ts2c": "PatchTST", "p1c": "p1c",
             "r6corr": "r6corr", "s35w": "s35w", "a3x25": "A — the anchor"}
    cand = {"sx1": "XLinear", "sx2": "TSMixer", "sx3": "cond. flow", "sx4": "TimeXer",
            "ts1b": "DLinear", "ts3b": "iTransformer", "x11": "i11w@25", "x3": "i11@25",
            "s40": "s40 (dropped)", "f2_hurdle": "hurdle f2", "decay": "persistence"}

    def err(slug):
        p = load(slug).set_index(["fipsCode", "hour"])["osi_pred"]
        y = np.array([t.loc[a, b] if b in t.columns else np.nan for a, b in p.index])
        return pd.Series(p.to_numpy() - y, index=p.index).dropna()

    E = {}
    for s in list(label) + list(cand):
        try:
            E[s] = err(s)
        except Exception:
            pass
    pts = []
    for s, nm in {**label, **cand}.items():
        if s not in E:
            continue
        cs = []
        for o in fz + ["a3x25"]:
            if o == s or o not in E:
                continue
            j = E[s].index.intersection(E[o].index)
            cs.append(np.corrcoef(E[s].reindex(j), E[o].reindex(j))[0, 1])
        try:
            r = score_trajectory(load(s), t).loc["mean", "rmse"]
        except Exception:
            continue
        grp = "anchor" if s == "a3x25" else ("bracket" if s in fz else "tested")
        pts.append({"id": s, "label": nm, "rmse": round(float(r), 6),
                    "corr": round(float(np.mean(cs)), 4), "group": grp})
    return sorted(pts, key=lambda d: d["rmse"])


ANCHOR_CODE = "A3x25"


def main():
    cat = pd.read_csv(OUT / "catalogue" / "catalogue.csv") if (OUT / "catalogue" / "catalogue.csv").exists() \
        else pd.read_csv(ROOT / "catalogue" / "catalogue.csv")
    cat["rmse_num"] = pd.to_numeric(cat.rmse_mean, errors="coerce")
    fam = (cat.groupby("family")
              .agg(runs=("code", "size"), best=("rmse_num", "min"), median=("rmse_num", "median"),
                   shipped=("shipped", "sum"))
              .reset_index().sort_values("runs", ascending=False))
    data = {
        "runs": [{"code": r["code"], "name": str(r["name"])[:70], "family": r["family"],
                  "mech": r["mechanism"], "obj": r["objective"],
                  "rmse": None if pd.isna(r["rmse_num"]) else round(float(r["rmse_num"]), 6),
                  "h": [None if pd.isna(pd.to_numeric(r.get(f"rmse_{h}"), errors="coerce"))
                        else round(float(r[f"rmse_{h}"]), 6) for h in ("t01h", "t06h", "t24h", "t48h")],
                  "mae": None if pd.isna(pd.to_numeric(r.get("mae_mean"), errors="coerce"))
                         else round(float(r["mae_mean"]), 6),
                  "secs": None if pd.isna(r.get("secs")) else int(r["secs"]),
                  "shipped": bool(r["shipped"]),
                  # the anchor holds half the forecast by itself; the other eleven
                  # shipped models share the other half, 1/22 each
                  "role": ("anchor" if r["code"] == ANCHOR_CODE
                           else "bracket" if bool(r["shipped"]) else None)}
                 for _, r in cat.iterrows()],
        "families": [{"family": r["family"], "runs": int(r["runs"]),
                      "best": None if pd.isna(r["best"]) else round(float(r["best"]), 6),
                      "median": None if pd.isna(r["median"]) else round(float(r["median"]), 6),
                      "shipped": int(r["shipped"])} for _, r in fam.iterrows()],
        "diversity": diversity_points(),
        "baselines": {"zeros": 0.016317, "persistence": 0.011097, "anchor": 0.008621, "final": 0.008354},
        "meta": {"n": len(cat), "scored": int(cat.rmse_num.notna().sum())},
    }
    (OUT / "catalogue").mkdir(parents=True, exist_ok=True)
    extra = ROOT / "catalogue" / "_extra.json"
    if extra.exists():
        data["extra"] = json.loads(extra.read_text())
    blob = json.dumps(data, separators=(",", ":"))
    (OUT / "catalogue" / "dashboard_data.json").write_text(blob)
    # Inline the data so the page works when opened from disk: a fetch() of a sibling
    # file is blocked by the file:// origin policy in every current browser.
    tpl = (ROOT / "catalogue" / "_dashboard_template.html").read_text()
    html = tpl.replace("fetch('dashboard_data.json').then(r=>r.json()).then(D=>{",
                       "((D)=>{").replace("});\n</script>", "})(DATA);\n</script>")
    html = html.replace("<script>\nconst tip=", "<script>\nconst DATA=" + blob + ";\nconst tip=")
    assert "DATA=" in html and "fetch(" not in html, "inlining failed"
    (OUT / "catalogue" / "dashboard.html").write_text(html)
    print(f"wrote dashboard.html ({len(html)/1024:.0f} KB, self-contained) and dashboard_data.json"
          f"  — {data['meta']['n']} runs, {len(data['diversity'])} diversity points")
    return data


if __name__ == "__main__":
    main()
