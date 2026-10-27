#!/usr/bin/env python3
"""Two heatmaps for the Results chapter, both on the ideal set (Set B, 24 photographs).

    python make_heatmaps.py          (from python/)

heatmap_leads_ideal   share of each lead's photographs with PCC >= 0.9, drawn in the
                      positions of the 3x4 printout (rows I/II/III, columns 0-2.5 ...
                      7.5-10 s), learned and classical tracer side by side. Shows where
                      on the page fidelity is lost, e.g. the last column, which has a
                      time marker on one side only.
heatmap_clf_ideal     the classifier's decisions on the 24 sheets: rows the record's
                      PTB-XL superclass, columns the superclasses flagged, cells the
                      number of sheets; on the record's true PTB-XL signal, the learned
                      and the classical tracer. Valid as a matrix only because each test
                      record carries a single superclass (Section 3.5.4); a record can be
                      flagged with several, so rows may sum to more than their sheets.

Presentation of committed results only: the per-lead scores of
results/device/sheet_scores_all_sets.csv with python/test_set_all.csv, loaded and
summarised with make_report_figures' own `load` and `summary` (an unrecovered lead counts
as below 0.9), and the classifier decisions of results/ml/clf_phone_eval.csv. Writes the
.pdf, the numbers behind it as .csv, and a JSON sidecar per figure to results/report/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import make_report_figures as R

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SCORES = REPO / "results" / "device" / "sheet_scores_all_sets.csv"
MANIFEST = HERE / "test_set_all.csv"
CLF = REPO / "results" / "ml" / "clf_phone_eval.csv"
OUT = REPO / "results" / "report"

#: The 3x4 printout: row r, column c holds LAYOUT[r][c].
LAYOUT = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6"))
COLUMNS = ("0–2.5 s", "2.5–5 s", "5–7.5 s", "7.5–10 s")
CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
TRACER_TITLE = {"learned": "Learned tracer", "classical": "Classical tracer"}


def cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("eacg", ["#FFFFFF", R.COLOUR["learned"]])


def ink(frac: float) -> str:
    return "white" if frac >= 0.55 else "black"


def lead_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind in R.TRACERS:
        d = df[df["tracer_kind"] == kind]
        for r, names in enumerate(LAYOUT):
            for c, lead in enumerate(names):
                s = R.summary(d[d["lead"] == lead]["pcc_asis"])
                rows.append({"tracer": kind, "lead": lead, "row": r + 1, "col": c + 1,
                             "n_leads": s["n"], "n_unavailable": s["n_unavailable"],
                             "pct_ge_09": s["pct_ge_09"], "median_pcc": s["median"]})
    return pd.DataFrame(rows)


def fig_leads(plt, t: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(R.FULL_W, 1.85), constrained_layout=True)
    for ax, kind in zip(axes, R.TRACERS):
        d = t[t["tracer"] == kind]
        grid = np.full((3, 4), np.nan)
        for _, x in d.iterrows():
            grid[x["row"] - 1, x["col"] - 1] = x["pct_ge_09"]
        im = ax.imshow(grid, cmap=cmap(), vmin=0, vmax=100, aspect="auto")
        for _, x in d.iterrows():
            frac = x["pct_ge_09"] / 100
            ax.text(x["col"] - 1, x["row"] - 1, f"{x['lead']}\n{x['pct_ge_09']:.0f} %",
                    ha="center", va="center", fontsize=7.5, color=ink(frac), linespacing=1.15)
        ax.set_xticks(range(4), COLUMNS)
        ax.set_yticks([])
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 4), minor=True)
        ax.set_yticks(np.arange(-0.5, 3), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        ax.set_title(TRACER_TITLE[kind], fontsize=9, pad=16)
    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cb.set_label("Leads with PCC ≥ 0.9 (%)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    fig.savefig(path)
    plt.close(fig)


def clf_table(c: pd.DataFrame) -> pd.DataFrame:
    """Counts of sheets per (panel, true superclass, flagged superclass)."""
    b = c[c["set"] == "B"].copy()
    panels = {"truth": b[b["tracer"] == "learned"].assign(says=lambda x: x["truth_says"]),
              "learned": b[b["tracer"] == "learned"].assign(says=lambda x: x["phone_says"]),
              "classical": b[b["tracer"] == "classical"].assign(says=lambda x: x["phone_says"])}
    rows = []
    for panel, d in panels.items():
        for true in CLASSES:
            dt = d[d["labels"] == true]
            flags = [set(str(s).split()) - {"none"} for s in dt["says"]]
            r = {"panel": panel, "true": true, "n_sheets": len(dt)}
            for k in CLASSES:
                r[k] = sum(k in f for f in flags)
            r["none"] = sum(not f for f in flags)
            rows.append(r)
    return pd.DataFrame(rows)


def fig_clf(plt, t: pd.DataFrame, path: Path):
    cols = list(CLASSES) + ["none"]
    titles = {"truth": "True PTB-XL signal", "learned": "Learned tracer", "classical": "Classical tracer"}
    fig, axes = plt.subplots(1, 3, figsize=(R.FULL_W, 2.25), constrained_layout=True, sharey=True)
    for ax, panel in zip(axes, ("truth", "learned", "classical")):
        d = t[t["panel"] == panel].set_index("true").loc[list(CLASSES)]
        counts = d[cols].to_numpy(float)
        frac = counts / d["n_sheets"].to_numpy(float)[:, None]
        ax.imshow(frac, cmap=cmap(), vmin=0, vmax=1, aspect="auto")
        for i in range(len(CLASSES)):
            for j in range(len(cols)):
                if counts[i, j] > 0:
                    ax.text(j, i, f"{counts[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                            color=ink(frac[i, j]))
        ax.set_xticks(range(len(cols)), cols, rotation=90)
        ax.set_yticks(range(len(CLASSES)), [f"{k} ({n})" for k, n in zip(CLASSES, d["n_sheets"])])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(CLASSES)), minor=True)
        ax.grid(which="minor", color="#DDDDDD", linewidth=0.6)
        ax.tick_params(which="minor", length=0)
        for k in range(len(CLASSES)):  # outline the diagonal: the record's own superclass
            ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=0.9))
        ax.set_title(titles[panel], fontsize=9)
    fig.supxlabel("Superclass flagged", fontsize=9)
    axes[0].set_ylabel("PTB-XL superclass (sheets)")
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    plt = R.style()
    df, report = R.load(SCORES, MANIFEST)
    df = R.leads12(df)
    ideal = df[df["set"] == "ideal"]
    out = R.Out(OUT, {"scores": str(SCORES.relative_to(REPO)), "manifest": str(MANIFEST.relative_to(REPO)),
                      "clf": str(CLF.relative_to(REPO)), "set": "ideal (Set B)", "good_pcc": R.GOOD,
                      "leads": "12 (rhythm strip excluded)"})
    out.prov["generated_by"] = "python/make_heatmaps.py"

    t = lead_table(ideal)
    out.csv("heatmap_leads_ideal.csv", t)
    fig_leads(plt, t, OUT / "heatmap_leads_ideal.pdf")
    out.sidecar("heatmap_leads_ideal", ["heatmap_leads_ideal.pdf", "heatmap_leads_ideal.csv"],
                {"n_photos": int(ideal["export"].nunique()), "load_report": report})

    c = clf_table(pd.read_csv(CLF))
    out.csv("heatmap_clf_ideal.csv", c)
    fig_clf(plt, c, OUT / "heatmap_clf_ideal.pdf")
    out.sidecar("heatmap_clf_ideal", ["heatmap_clf_ideal.pdf", "heatmap_clf_ideal.csv"],
                {"decisions": "phone_says (learned, classical); truth_says (true signal)"})

    print(t.pivot_table(index="lead", columns="tracer", values="pct_ge_09").round(0).loc[
        [l for r in LAYOUT for l in r]].to_string())
    print(c.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
