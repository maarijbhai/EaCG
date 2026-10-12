#!/usr/bin/env python3
"""Every figure and table of the Results chapter, from the scored phone exports.

    python make_report_figures.py --draft-manifest      # write python/test_set.csv to edit
    python make_report_figures.py                       # -> results/report/
    python make_report_figures.py --out results/report_dev

INPUTS
  results/device/sheet_scores.csv   score_device_sheets.py: one row per lead per export
  python/test_set.csv               the manifest: export, photo_id, set, condition, photo_path
                                    set in {clean, ideal, robustness}; condition in {clean,
                                    ideal, side_light, shadow, tilt, curl}; photo_id shared by
                                    the learned and classical exports of ONE photograph (the
                                    pairing for the tracer comparison). Exports not in the
                                    manifest are reported and skipped, never guessed.

RULES (fixed; CLAUDE.md and the brief of 2026-10-12)
  * The unit is the lead. The 12 non-rhythm leads are the main statistics; the rhythm
    strip (II_rhythm) is reported separately.
  * A lead with no score (pcc_asis empty) is a FAILURE: counted in n_unavailable and as
    "not >= 0.9". pcc_aligned empty = "not measurable" (the 8.5 s span rule): counted.
  * Latency and memory are per export: de-duplicated by export first. Stages named
    score* (or ending in *) are HARNESS, not pipeline. The legacy key "cut+score"
    (exports before 2026-10-06) mixes cutting with scoring and cannot be split: it is
    left out of pipeline totals and shown separately as legacy.
  * Median and IQR; 95% CI of a median by CLUSTER bootstrap over photographs (leads
    within a photo are not independent), 2000 resamples, seed 20261012. Tracer
    comparison: Wilcoxon signed-rank on pcc_asis paired by (photo_id, lead).
    Robustness: each condition lead minus the median of the same record's ideal-set
    photos for that lead and tracer; median paired difference with cluster-bootstrap CI.

OUTPUTS (results/report/): figures .pdf, tables .tex, the numbers behind each as .csv,
a JSON sidecar per output stem (git commit, dirty flag, parameters incl. the manifest
hash, library versions), and headline.json -- the numbers quoted in the prose.

Analysis only: nothing here changes the app, the pipeline or any scoring function.
The digitised lead and its truth for the example figure come from
score_device_sheets.export_leads, the scorer's own cutting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]
SCORES = REPO / "results" / "device" / "sheet_scores.csv"
MANIFEST = REPO / "python" / "test_set.csv"
EXPORTS = REPO / "data" / "device_exports"

DEV_IDS = (9, 38, 40, 57)
HELDOUT_IDS = (1737, 3908, 4016, 8504, 167, 160, 7916, 5830)
SETS = ("clean", "ideal", "robustness")
CONDITIONS = ("ideal", "side_light", "shadow", "tilt", "curl")
LEADS12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
TRACERS = ("learned", "classical")
COLOUR = {"learned": "#7A1F2B", "classical": "#808080", "truth": "#BBBBBB"}
FULL_W, HALF_W = 6.3, 3.1
N_BOOT = 2000
SEED = 20261012
GOOD = 0.9


# ============================================================ parsing

def parse_timings(s) -> dict:
    """'corners=0 rectify=224 score*=96' -> {stage: ms}; {} for empty/NaN; bad parts skipped."""
    if not isinstance(s, str) or not s.strip():
        return {}
    out = {}
    for part in s.split():
        k, _, v = part.partition("=")
        try:
            out[k] = int(float(v))
        except ValueError:
            continue
    return out


def parse_memory(s) -> tuple:
    """'baseline=59.7/37.2 trace=510.6/65.8' -> (baseline (native, java) or None,
    {stage: (native, java)}); (None, {}) for empty/NaN."""
    if not isinstance(s, str) or not s.strip():
        return None, {}
    base, stages = None, {}
    for part in s.split():
        k, _, v = part.partition("=")
        nj = v.split("/")
        try:
            pair = (float(nj[0]), float(nj[1]))
        except (ValueError, IndexError):
            continue
        if k == "baseline":
            base = pair
        else:
            stages[k] = pair
    return base, stages


def stage_kind(stage: str) -> str:
    """'pipeline', 'harness' (score*, *), or 'legacy' (cut+score, mixed)."""
    if stage.startswith("score") or stage.endswith("*"):
        return "harness"
    if stage == "cut+score":
        return "legacy"
    return "pipeline"


def tracer_kind(s) -> str:
    s = str(s or "")
    return "learned" if s.startswith("model") else "classical" if "classical" in s else "unknown"


def group_of(ecg_id) -> str:
    try:
        e = int(ecg_id)
    except (TypeError, ValueError):
        return "other"
    return "dev" if e in DEV_IDS else "held_out" if e in HELDOUT_IDS else "other"


# ============================================================ data

def load(scores: Path, manifest: Path) -> tuple[pd.DataFrame, dict]:
    """Scores joined with the manifest; plus a report of what was skipped."""
    sc = pd.read_csv(scores)
    man = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    for c in ("export", "photo_id", "set", "condition"):
        if c not in man.columns:
            raise ValueError(f"{manifest}: missing column '{c}'")
    if "photo_path" not in man.columns:
        man["photo_path"] = ""
    bad = man[~man["set"].isin(SETS)]
    if len(bad):
        raise ValueError(f"{manifest}: unknown set(s) {sorted(bad['set'].unique())}")
    in_scores, in_man = set(sc["export"]), set(man["export"])
    report = {"skipped_not_in_manifest": sorted(in_scores - in_man),
              "in_manifest_without_scores": sorted(in_man - in_scores)}
    df = sc.merge(man, on="export", how="inner")
    df["tracer_kind"] = df["tracer"].map(tracer_kind)
    df["group"] = df["ecg_id"].map(group_of)
    df["rhythm"] = df["rhythm"].astype(str).str.lower().eq("true")
    for c in ("pcc_asis", "pcc_aligned", "snr_db_asis", "shifted_snr_db_asis", "rmse_mv_asis",
              "gain_ratio", "r_ratio", "coverage"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["available"] = df["pcc_asis"].notna()
    return df, report


def leads12(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["rhythm"].fillna(False).astype(bool)]


# ============================================================ statistics

def summary(values) -> dict:
    """median / IQR over available values; n counts failures (NaN) too; % >= 0.9 of all."""
    v = np.asarray(values, np.float64)
    ok = v[np.isfinite(v)]
    n = len(v)
    out = {"n": n, "n_unavailable": int(n - len(ok)),
           "pct_ge_09": (100.0 * float((ok >= GOOD).sum()) / n) if n else float("nan")}
    if len(ok):
        out.update(median=float(np.median(ok)), q1=float(np.percentile(ok, 25)), q3=float(np.percentile(ok, 75)))
    else:
        out.update(median=float("nan"), q1=float("nan"), q3=float("nan"))
    return out


def cluster_ci(df: pd.DataFrame, col: str, cluster: str = "photo_id", n_boot: int = N_BOOT,
               seed: int = SEED) -> tuple:
    """95% CI of the median of df[col] by resampling clusters (photos) with replacement."""
    d = df[[cluster, col]].dropna()
    groups = [g[col].to_numpy() for _, g in d.groupby(cluster)]
    if len(groups) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        meds[b] = np.median(np.concatenate([groups[i] for i in pick]))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired_tracers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """(photo_id, lead) pairs with both tracers available; and the unpaired exports."""
    d = leads12(df)
    learned = d[d["tracer_kind"] == "learned"]
    classical = d[d["tracer_kind"] == "classical"]
    pl, pc = set(learned["photo_id"]), set(classical["photo_id"])
    report = {"learned_photos_without_classical": sorted(pl - pc),
              "classical_photos_without_learned": sorted(pc - pl)}
    a = learned[["photo_id", "lead", "pcc_asis"]].rename(columns={"pcc_asis": "learned"})
    b = classical[["photo_id", "lead", "pcc_asis"]].rename(columns={"pcc_asis": "classical"})
    pairs = a.merge(b, on=["photo_id", "lead"]).dropna()
    return pairs, report


def wilcoxon(pairs: pd.DataFrame) -> dict:
    n = len(pairs)
    if n < 1:
        return {"n_pairs": 0, "statistic": None, "p": None, "median_diff": None}
    from scipy.stats import wilcoxon as w
    diff = pairs["learned"] - pairs["classical"]
    if np.allclose(diff, 0):
        return {"n_pairs": n, "statistic": None, "p": None, "median_diff": 0.0}
    r = w(pairs["learned"], pairs["classical"])
    return {"n_pairs": n, "statistic": float(r.statistic), "p": float(r.pvalue),
            "median_diff": float(np.median(diff))}


def robustness_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Each robustness lead minus the median over the same record's ideal photos (same lead, tracer)."""
    d = leads12(df)
    ideal = (d[d["set"] == "ideal"].groupby(["ecg_id", "lead", "tracer_kind"])["pcc_asis"]
             .median().rename("ideal_pcc").reset_index())
    rob = d[d["set"] == "robustness"]
    out = rob.merge(ideal, on=["ecg_id", "lead", "tracer_kind"], how="left")
    out["diff"] = out["pcc_asis"] - out["ideal_pcc"]
    return out


def choose_example(df_set: pd.DataFrame) -> pd.Series | None:
    """The (photo, lead) whose LEARNED pcc_asis is closest to that set's learned median.
    Ties: export name, then lead in standard order (deterministic)."""
    d = leads12(df_set)
    d = d[(d["tracer_kind"] == "learned") & d["available"]].copy()
    if d.empty:
        return None
    med = float(np.median(d["pcc_asis"]))
    d["dist"] = (d["pcc_asis"] - med).abs()
    d["lead_order"] = d["lead"].map({n: i for i, n in enumerate(LEADS12)})
    return d.sort_values(["dist", "export", "lead_order"]).iloc[0]


# ============================================================ output helpers

def clean_json(o):
    """NaN / inf -> None, recursively (json.dumps would write invalid 'NaN')."""
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


def git(*a) -> str:
    return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()


class Out:
    """Writes outputs and a JSON sidecar per output stem."""

    def __init__(self, root: Path, params: dict):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        import matplotlib
        self.prov = {
            "generated_by": "python/make_report_figures.py",
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--", "python")),
            "params": params,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        }
        self.written = []

    def sidecar(self, stem: str, files: list[str], extra: dict | None = None):
        (self.root / f"{stem}.json").write_text(json.dumps(
            clean_json({**self.prov, "files": files, **(extra or {})}), indent=2, default=str), encoding="utf-8")
        self.written += files

    def csv(self, name: str, frame: pd.DataFrame):
        frame.to_csv(self.root / name, index=False)

    def tex(self, name: str, text: str):
        (self.root / name).write_text(text, encoding="utf-8")


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.bbox": "tight"})
    return plt


def no_data(ax, text="no data yet"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="#666666")


def fmt(v, p=3):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{p}f}"


# ============================================================ figures

def fig_example(out: Out, plt, df: pd.DataFrame, set_name: str, pcc_lim) -> dict | None:
    """F1: one lead (chosen by rule), truth under each tracer, identical axes."""
    import score_device_sheets as SDS
    d = df[df["set"] == set_name]
    pick = choose_example(d)
    if pick is None:
        return None
    lead, photo = pick["lead"], pick["photo_id"]
    traces, rows = {}, {}
    for tk in TRACERS:
        r = d[(d["photo_id"] == photo) & (d["lead"] == lead) & (~d["rhythm"]) & (d["tracer_kind"] == tk)]
        if r.empty:
            continue
        r = r.iloc[0]
        meta, cols = SDS.read_export(EXPORTS / r["export"])
        for name, rhythm, row, col, sig, tr in SDS.export_leads(EXPORTS / r["export"], meta, cols):
            if name == lead and not rhythm:
                t0 = col * 2.5
                traces[tk] = (t0, sig, tr)
                rows[tk] = r
    fs = 500.0
    t0_any, _, tr_any = next(iter(traces.values()))
    xlim = (t0_any, t0_any + (len(tr_any) - 1) / fs)
    allv = []
    for t0, sig, tr in traces.values():
        allv += [sig - np.nanmedian(sig), tr - np.nanmedian(tr)]
    v = np.concatenate([x[np.isfinite(x)] for x in allv])
    ylim = (float(np.min(v)) - 0.1, float(np.max(v)) + 0.1)
    data = []
    for tk in TRACERS:
        fig, ax = plt.subplots(figsize=(HALF_W, 1.9))
        if tk in traces:
            t0, sig, tr = traces[tk]
            t = t0 + np.arange(len(tr)) / fs
            ax.plot(t, tr - np.nanmedian(tr), color=COLOUR["truth"], lw=1.6, label="PTB-XL")
            ax.plot(t[:len(sig)], sig - np.nanmedian(sig), color=COLOUR[tk], lw=0.9, label=tk)
            ax.text(0.98, 0.95, f"r = {rows[tk]['pcc_asis']:.3f}", transform=ax.transAxes,
                    ha="right", va="top")
            for i in range(len(tr)):
                data.append({"tracer": tk, "t_s": round(t[i], 4), "truth_mv": tr[i] - np.nanmedian(tr),
                             "digitised_mv": (sig[i] - np.nanmedian(sig)) if i < len(sig) else np.nan})
        else:
            no_data(ax, f"no {tk} export of this photo")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mV")
        fig.savefig(out.root / f"example_{set_name}_{tk}.pdf")
        plt.close(fig)
    out.csv(f"example_{set_name}.csv", pd.DataFrame(data))
    files = [f"example_{set_name}_{tk}.pdf" for tk in TRACERS] + [f"example_{set_name}.csv"]
    ppath = str(pick.get("photo_path", "") or "")
    if ppath:
        p = Path(ppath) if Path(ppath).is_absolute() else REPO / ppath
        if p.is_file():
            from PIL import Image
            im = Image.open(p)
            im.thumbnail((1600, 1600))
            im.convert("RGB").save(out.root / f"example_{set_name}_photo.jpg", quality=90)
            files.append(f"example_{set_name}_photo.jpg")
    info = {"set": set_name, "photo_id": photo, "export": pick["export"], "lead": lead,
            "ecg_id": int(pick["ecg_id"]), "learned_pcc": float(pick["pcc_asis"]),
            "set_learned_median": float(np.median(leads12(d)[(leads12(d)["tracer_kind"] == "learned")
                                                             & leads12(d)["available"]]["pcc_asis"])),
            "classical_export_found": "classical" in traces}
    out.sidecar(f"example_{set_name}", files, {"chosen": info})
    print(f"  F1 example ({set_name}): photo {photo}, lead {lead}, learned r {info['learned_pcc']:.3f} "
          f"(set median {info['set_learned_median']:.3f})")
    return info


def categories(df: pd.DataFrame) -> list[tuple]:
    """x categories of F2: clean, ideal, then each robustness condition present."""
    cats = [("clean", "clean"), ("ideal", "ideal")]
    for c in CONDITIONS[1:]:
        cats.append(("robustness", c))
    return cats


def fig_pcc_by_set(out: Out, plt, df: pd.DataFrame, pcc_lim):
    d = leads12(df)
    cats = categories(df)
    fig, ax = plt.subplots(figsize=(FULL_W, 2.6))
    rows = []
    for i, (s, c) in enumerate(cats):
        for j, tk in enumerate(TRACERS):
            sel = d[(d["set"] == s) & (d["tracer_kind"] == tk) & ((d["condition"] == c) if s == "robustness" else True)]
            vals = sel["pcc_asis"].to_numpy(float)
            sm = summary(vals)
            rows.append({"set": s, "condition": c, "tracer": tk, **sm})
            x = i + (j - 0.5) * 0.36
            ok = vals[np.isfinite(vals)]
            if len(ok):
                ax.boxplot(ok, positions=[x], widths=0.3, showfliers=True, patch_artist=True,
                           boxprops={"facecolor": "white", "edgecolor": COLOUR[tk]},
                           medianprops={"color": COLOUR[tk], "lw": 1.5},
                           whiskerprops={"color": COLOUR[tk]}, capprops={"color": COLOUR[tk]},
                           flierprops={"markeredgecolor": COLOUR[tk], "markersize": 3})
    ns = {(r["set"], r["condition"], r["tracer"]): r["n"] for r in rows}
    ax.set_xticks(range(len(cats)), [f"{c.replace('_', ' ')}\n{ns[(s, c, 'learned')]} | {ns[(s, c, 'classical')]}"
                                      for s, c in cats])
    ax.text(1.0, -0.22, "n leads: learned | classical", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_xlim(-0.6, len(cats) - 0.4)
    ax.set_ylim(*pcc_lim)
    ax.set_ylabel("lead PCC")
    ax.axhline(GOOD, color="#DDDDDD", lw=0.8, zorder=0)
    for tk in TRACERS:
        ax.plot([], [], color=COLOUR[tk], lw=2, label=tk)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.savefig(out.root / "pcc_by_set.pdf")
    plt.close(fig)
    out.csv("pcc_by_set.csv", pd.DataFrame(rows))
    out.sidecar("pcc_by_set", ["pcc_by_set.pdf", "pcc_by_set.csv"])


def fig_pcc_by_lead(out: Out, plt, df: pd.DataFrame, pcc_lim):
    d = leads12(df[df["set"] == "ideal"])
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    rows = []
    any_data = False
    for j, tk in enumerate(TRACERS):
        for i, ld in enumerate(LEADS12):
            sm = summary(d[(d["tracer_kind"] == tk) & (d["lead"] == ld)]["pcc_asis"])
            rows.append({"tracer": tk, "lead": ld, **sm})
            if np.isfinite(sm["median"]):
                any_data = True
                x = i + (j - 0.5) * 0.3
                ax.plot([x, x], [sm["q1"], sm["q3"]], color=COLOUR[tk], lw=1.2)
                ax.plot([x], [sm["median"]], "o", color=COLOUR[tk], ms=3.5)
    ax.set_xticks(range(12), LEADS12)
    ax.set_ylim(*pcc_lim)
    ax.set_ylabel("lead PCC (median, IQR)")
    ax.axhline(GOOD, color="#DDDDDD", lw=0.8, zorder=0)
    for tk in TRACERS:
        ax.plot([], [], "o-", color=COLOUR[tk], ms=3.5, label=tk)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    if not any_data:
        no_data(ax)
    fig.savefig(out.root / "pcc_by_lead.pdf")
    plt.close(fig)
    out.csv("pcc_by_lead.csv", pd.DataFrame(rows))
    out.sidecar("pcc_by_lead", ["pcc_by_lead.pdf", "pcc_by_lead.csv"])


def fig_plain_vs_aligned(out: Out, plt, df: pd.DataFrame, pcc_lim):
    d = leads12(df[(df["set"] == "ideal") & (df["tracer_kind"] == "learned")])
    both = d.dropna(subset=["pcc_asis", "pcc_aligned"])
    fig, ax = plt.subplots(figsize=(HALF_W, HALF_W))
    ax.plot(pcc_lim, pcc_lim, color="#BBBBBB", lw=0.8, zorder=0)
    if len(both):
        ax.scatter(both["pcc_asis"], both["pcc_aligned"], s=8, color=COLOUR["learned"], alpha=0.7, lw=0)
    else:
        no_data(ax)
    ax.set_xlim(*pcc_lim); ax.set_ylim(*pcc_lim)
    ax.set_xlabel("PCC as recorded"); ax.set_ylabel("PCC aligned")
    ax.set_aspect("equal")
    fig.savefig(out.root / "plain_vs_aligned.pdf")
    plt.close(fig)
    out.csv("plain_vs_aligned.csv", d[["export", "photo_id", "lead", "pcc_asis", "pcc_aligned"]])
    out.sidecar("plain_vs_aligned", ["plain_vs_aligned.pdf", "plain_vs_aligned.csv"],
                {"n_points": int(len(both)), "n_aligned_not_measurable": int(d["pcc_aligned"].isna().sum())})


def fig_pcc_by_quality(out: Out, plt, df: pd.DataFrame, pcc_lim):
    d = leads12(df[df["set"].isin(["ideal", "robustness"]) & (df["tracer_kind"] == "learned")]).copy()
    d["verdict"] = d["quality"].fillna("").replace("", "not recorded")
    order = ["ok", "check", "unreliable", "not recorded"]
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(HALF_W, 2.4))
    rows = []
    for i, v in enumerate(order):
        vals = d[d["verdict"] == v]["pcc_asis"].to_numpy(float)
        sm = summary(vals)
        rows.append({"verdict": v, **sm})
        ok = vals[np.isfinite(vals)]
        if len(ok):
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(ok)), ok, s=6, color=COLOUR["learned"], alpha=0.6, lw=0)
            ax.plot([i - 0.25, i + 0.25], [sm["median"]] * 2, color="black", lw=1.2)
    ax.set_xticks(range(len(order)), [f"{v}\nn = {r['n']}" for v, r in zip(order, rows)])
    ax.set_ylim(*pcc_lim)
    ax.set_ylabel("lead PCC (learned)")
    fig.savefig(out.root / "pcc_by_quality.pdf")
    plt.close(fig)
    out.csv("pcc_by_quality.csv", pd.DataFrame(rows))
    out.sidecar("pcc_by_quality", ["pcc_by_quality.pdf", "pcc_by_quality.csv"])


def robustness_rows(df: pd.DataFrame) -> pd.DataFrame:
    pairs = robustness_pairs(df)
    rows = []
    for c in CONDITIONS[1:]:
        for tk in TRACERS:
            sel = pairs[(pairs["condition"] == c) & (pairs["tracer_kind"] == tk)]
            sm = summary(sel["pcc_asis"])
            dsel = sel.dropna(subset=["diff"])
            lo, hi = cluster_ci(dsel, "diff") if len(dsel) else (np.nan, np.nan)
            ssnr = summary(sel["shifted_snr_db_asis"])["median"]
            rows.append({"condition": c, "tracer": tk, "median_pcc": sm["median"], "pct_ge_09": sm["pct_ge_09"],
                         "n_leads": sm["n"], "n_unavailable": sm["n_unavailable"],
                         "median_diff": float(np.median(dsel["diff"])) if len(dsel) else np.nan,
                         "diff_ci_lo": lo, "diff_ci_hi": hi, "n_paired": int(len(dsel)),
                         "median_shifted_snr": ssnr})
    return pd.DataFrame(rows)


def fig_robustness(out: Out, plt, rob: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    conds = list(CONDITIONS[1:])
    any_data = False
    for j, tk in enumerate(TRACERS):
        for i, c in enumerate(conds):
            r = rob[(rob["condition"] == c) & (rob["tracer"] == tk)].iloc[0]
            if np.isfinite(r["median_diff"]):
                any_data = True
                x = i + (j - 0.5) * 0.3
                if np.isfinite(r["diff_ci_lo"]):
                    ax.plot([x, x], [r["diff_ci_lo"], r["diff_ci_hi"]], color=COLOUR[tk], lw=1.2)
                ax.plot([x], [r["median_diff"]], "o", color=COLOUR[tk], ms=4)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(range(len(conds)), [c.replace("_", " ") for c in conds])
    ax.set_xlim(-0.6, len(conds) - 0.4)
    ax.set_ylabel(r"$\Delta$ lead PCC vs ideal")
    for tk in TRACERS:
        ax.plot([], [], "o-", color=COLOUR[tk], ms=4, label=tk)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    if not any_data:
        ax.text(0.5, 0.75, "no robustness photos yet", transform=ax.transAxes, ha="center", color="#666666")
    fig.savefig(out.root / "robustness.pdf")
    plt.close(fig)


def per_export(df: pd.DataFrame) -> pd.DataFrame:
    """One row per export: stage times (pipeline / harness / legacy) and memory."""
    e = df.drop_duplicates("export").copy()
    rows = []
    for _, r in e.iterrows():
        t = parse_timings(r.get("timings_ms"))
        base, mem = parse_memory(r.get("memory_peak_mb"))
        pipe = {k: v for k, v in t.items() if stage_kind(k) == "pipeline"}
        rows.append({"export": r["export"], "set": r["set"], "tracer_kind": r["tracer_kind"],
                     "stages": pipe, "harness": {k: v for k, v in t.items() if stage_kind(k) == "harness"},
                     "legacy": {k: v for k, v in t.items() if stage_kind(k) == "legacy"},
                     "pipeline_ms": sum(pipe.values()) if pipe else np.nan,
                     "mem": mem, "base": base,
                     "peak_native_mb": max((v[0] for v in mem.values()), default=np.nan),
                     "pss_end_mb": pd.to_numeric(r.get("pss_end_mb"), errors="coerce")})
    return pd.DataFrame(rows)


def stage_order(pe: pd.DataFrame) -> list[str]:
    order = []
    for s in pe["stages"]:
        for k in s:
            if k not in order:
                order.append(k)
    return order


def fig_latency(out: Out, plt, pe: pd.DataFrame):
    p = pe[pe["set"] == "ideal"]
    stages = stage_order(p)
    fig, ax = plt.subplots(figsize=(FULL_W, 1.6))
    cmap = plt.get_cmap("Greys")
    rows = []
    any_data = False
    for j, tk in enumerate(TRACERS):
        sel = p[p["tracer_kind"] == tk]
        left = 0.0
        for i, s in enumerate(stages):
            vals = [st[s] for st in sel["stages"] if s in st]
            med = float(np.median(vals)) if vals else 0.0
            rows.append({"tracer": tk, "stage": s, "median_ms": med, "n_exports": len(vals)})
            if med > 0:
                any_data = True
                ax.barh(j, med / 1000, left=left / 1000, color=COLOUR[tk], alpha=0.35 + 0.65 * (i + 1) / len(stages),
                        edgecolor="white", lw=0.6)
                if med >= 1500:
                    ax.text((left + med / 2) / 1000, j, s, ha="center", va="center", fontsize=7, color="white")
                left += med
    ax.set_yticks(range(len(TRACERS)), TRACERS)
    ax.set_xlabel("median time per sheet (s), pipeline stages only")
    if not any_data:
        no_data(ax)
    fig.savefig(out.root / "latency_stages.pdf")
    plt.close(fig)
    out.csv("latency_stages.csv", pd.DataFrame(rows))
    out.sidecar("latency_stages", ["latency_stages.pdf", "latency_stages.csv"],
                {"n_exports": {tk: int((p["tracer_kind"] == tk).sum()) for tk in TRACERS},
                 "excluded": "harness stages (score*) and legacy 'cut+score' (mixed with scoring)"})


# ============================================================ tables

FIDELITY_ROWS = (("PCC", "pcc_asis", 3), ("Aligned PCC", "pcc_aligned", 3), ("SNR (dB)", "snr_db_asis", 1),
                 ("Shifted SNR (dB)", "shifted_snr_db_asis", 1), ("RMSE (mV)", "rmse_mv_asis", 3),
                 ("Gain", "gain_ratio", 2), ("R-peak ratio", "r_ratio", 2), ("Coverage", "coverage", 3))


def med_iqr(sm: dict, p: int) -> str:
    return "--" if not np.isfinite(sm["median"]) else f"{sm['median']:.{p}f} [{sm['q1']:.{p}f}, {sm['q3']:.{p}f}]"


def table_fidelity(out: Out, df: pd.DataFrame, set_name: str, wil: dict) -> dict:
    d = leads12(df[df["set"] == set_name])
    rows, csvrows = [], []
    sums = {tk: {} for tk in TRACERS}
    for label, col, p in FIDELITY_ROWS:
        cells = []
        for tk in TRACERS:
            sm = summary(d[d["tracer_kind"] == tk][col])
            sums[tk][col] = sm
            cell = med_iqr(sm, p)
            if col == "pcc_aligned" and sm["n"] and sm["n_unavailable"]:
                cell += f" ({sm['n_unavailable']} n.m.)"
            cells.append(cell)
            csvrows.append({"measure": label, "tracer": tk, **sm})
        rows.append((label, cells))
    for tk in TRACERS:
        pcc = summary(d[d["tracer_kind"] == tk]["pcc_asis"])
        sums[tk]["_pcc"] = pcc
    rows.append((r"Leads with PCC $\geq$ 0.9", [fmt(sums[tk]["_pcc"]["pct_ge_09"], 0) + r"\%" if sums[tk]["_pcc"]["n"] else "--" for tk in TRACERS]))
    rows.append(("Leads", [str(sums[tk]["_pcc"]["n"]) for tk in TRACERS]))
    rows.append(("Leads without a score", [str(sums[tk]["_pcc"]["n_unavailable"]) for tk in TRACERS]))
    w = (f"Wilcoxon signed-rank, PCC paired by photograph and lead: n = {wil['n_pairs']}" +
         (f", W = {wil['statistic']:.1f}, p = {wil['p']:.3g}" if wil.get("p") is not None else ", not computable"))
    tex = [r"\begin{tabular}{@{}lll@{}}", r"\toprule",
           r"\textbf{Measure} & \textbf{Learned} & \textbf{Classical}\\", r"\midrule"]
    tex += [f"{lab} & {a} & {b}\\\\" for lab, (a, b) in rows]
    tex += [r"\midrule", rf"\multicolumn{{3}}{{@{{}}l}}{{\footnotesize {w}}}\\", r"\bottomrule", r"\end{tabular}", ""]
    out.tex(f"fidelity_{set_name}.tex", "\n".join(tex))
    out.csv(f"fidelity_{set_name}.csv", pd.DataFrame(csvrows))
    out.sidecar(f"fidelity_{set_name}", [f"fidelity_{set_name}.tex", f"fidelity_{set_name}.csv"], {"wilcoxon": wil})
    return sums


def table_robustness(out: Out, rob: pd.DataFrame):
    tex = [r"\begin{tabular}{@{}lrlrrr@{}}", r"\toprule",
           r"\textbf{Condition} & \textbf{Median PCC} & \textbf{$\Delta$ vs ideal [95\% CI]} & "
           r"\textbf{PCC $\geq$ 0.9} & \textbf{Shifted SNR} & \textbf{Leads}\\"]
    for tk in TRACERS:
        tex += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{tk.capitalize()} tracer}}}}\\"]
        for _, r in rob[rob["tracer"] == tk].iterrows():
            ci = (f"{r['median_diff']:+.3f} [{r['diff_ci_lo']:+.3f}, {r['diff_ci_hi']:+.3f}]"
                  if np.isfinite(r["diff_ci_lo"]) else (f"{r['median_diff']:+.3f}" if np.isfinite(r["median_diff"]) else "--"))
            pct = f"{r['pct_ge_09']:.0f}\\%" if r["n_leads"] else "--"
            tex.append(f"{r['condition'].replace('_', ' ')} & {fmt(r['median_pcc'])} & {ci} & {pct} & "
                       f"{fmt(r['median_shifted_snr'], 1)} & {int(r['n_leads'])}\\\\")
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    out.tex("robustness.tex", "\n".join(tex))
    out.csv("robustness.csv", rob)
    out.sidecar("robustness", ["robustness.pdf", "robustness.tex", "robustness.csv"])


def table_on_device(out: Out, pe: pd.DataFrame) -> dict:
    p = pe[(pe["set"] == "ideal") & (pe["tracer_kind"] == "learned")]
    stages = stage_order(p)
    rows, csvrows = [], []
    for s in stages:
        ms = [st[s] for st in p["stages"] if s in st]
        mem = [m[s][0] for m in p["mem"] if s in m]
        r = {"stage": s, "median_ms": float(np.median(ms)), "min_ms": min(ms), "max_ms": max(ms),
             "peak_native_mb": float(np.median(mem)) if mem else np.nan, "n": len(ms)}
        csvrows.append(r)
        rows.append(f"{s} & {r['median_ms']:.0f} & {r['min_ms']}--{r['max_ms']} & {fmt(r['peak_native_mb'], 0)}\\\\")
    tot = p["pipeline_ms"].dropna()
    peak = p["peak_native_mb"].dropna()
    harness = sorted({k for h in p["harness"] for k in h} | {k for h in p["legacy"] for k in h})
    n = len(p)
    tex = [f"% caption: time and memory per sheet on the Galaxy A30s, learned tracer, ideal set (n = {n} exports);",
           "% harness stages excluded: " + (", ".join(harness) if harness else "none"),
           r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
           r"\textbf{Stage} & \textbf{Median (ms)} & \textbf{Range (ms)} & \textbf{Peak native (MB)}\\", r"\midrule"]
    tex += [x.replace("_", r"\_").replace("+", "+") for x in rows]
    tex += [r"\midrule",
            (f"\\textbf{{Total}} & {np.median(tot):.0f} & {tot.min():.0f}--{tot.max():.0f} & {fmt(float(np.median(peak)) if len(peak) else np.nan, 0)}\\\\"
             if len(tot) else r"\textbf{Total} & -- & -- & --\\"),
            r"\bottomrule", r"\end{tabular}", ""]
    out.tex("on_device.tex", "\n".join(tex))
    out.csv("on_device.csv", pd.DataFrame(csvrows))
    out.sidecar("on_device", ["on_device.tex", "on_device.csv"], {"n_exports": n, "harness_excluded": harness})
    trace = [st.get("trace") for st in p["stages"] if "trace" in st]
    return {"n_exports": n, "median_total_ms": float(np.median(tot)) if len(tot) else None,
            "model_share": (float(np.median(trace)) / float(np.median(tot))) if len(trace) and len(tot) else None,
            "median_peak_native_mb": float(np.median(peak)) if len(peak) else None,
            "median_pss_end_mb": float(np.median(p["pss_end_mb"].dropna())) if p["pss_end_mb"].notna().any() else None}


# ============================================================ manifest draft

def draft_manifest(path: Path, force: bool) -> int:
    if path.exists() and not force:
        print(f"{path} exists; not overwritten (use --force)")
        return 1
    exports = sorted(p.name for p in EXPORTS.glob("eacg_sheet_*.csv"))
    pd.DataFrame({"export": exports, "photo_id": [Path(e).stem for e in exports], "set": "ideal",
                  "condition": "ideal", "photo_path": ""}).to_csv(path, index=False)
    print(f"wrote {path}: {len(exports)} exports, all set=ideal, condition=ideal, photo_id = export stem")
    return 0


# ============================================================ main

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--draft-manifest", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing manifest with --draft-manifest")
    p.add_argument("--manifest", type=Path, default=MANIFEST)
    p.add_argument("--scores", type=Path, default=SCORES)
    p.add_argument("--out", type=Path, default=REPO / "results" / "report")
    args = p.parse_args(argv)
    if args.draft_manifest:
        return draft_manifest(args.manifest, args.force)

    out_dir = args.out if args.out.is_absolute() else REPO / args.out
    df, rep = load(args.scores, args.manifest)
    for e in rep["skipped_not_in_manifest"]:
        print(f"  skipped (not in the manifest): {e}")
    for e in rep["in_manifest_without_scores"]:
        print(f"  in the manifest but not scored (no target, or not exported): {e}")
    params = {"scores": str(args.scores.relative_to(REPO)) if args.scores.is_relative_to(REPO) else str(args.scores),
              "manifest": str(args.manifest.relative_to(REPO)) if args.manifest.is_relative_to(REPO) else str(args.manifest),
              "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              "n_boot": N_BOOT, "seed": SEED, "good_pcc": GOOD, "dev_ids": DEV_IDS, "held_out_ids": HELDOUT_IDS}
    out = Out(out_dir, params)
    plt = style()

    allp = pd.concat([df["pcc_asis"], df["pcc_aligned"]]).dropna()
    lo = min(0.0, np.floor((allp.min() if len(allp) else 0.0) * 10) / 10)
    pcc_lim = (lo - 0.02, 1.02)

    examples = {}
    for s in ("clean", "ideal"):
        examples[s] = fig_example(out, plt, df, s, pcc_lim)
        if examples[s] is None:
            print(f"  F1 example ({s}): no learned leads in this set -- not drawn")
    fig_pcc_by_set(out, plt, df, pcc_lim)
    fig_pcc_by_lead(out, plt, df, pcc_lim)
    fig_plain_vs_aligned(out, plt, df, pcc_lim)
    fig_pcc_by_quality(out, plt, df, pcc_lim)
    rob = robustness_rows(df)
    fig_robustness(out, plt, rob)
    table_robustness(out, rob)
    pe = per_export(df)
    fig_latency(out, plt, pe)
    device = table_on_device(out, pe)

    pairs, pair_rep = paired_tracers(df)
    for k, v in pair_rep.items():
        if v:
            print(f"  {k.replace('_', ' ')}: {len(v)} (excluded from the Wilcoxon)")
    head = {"sets": {}, "wilcoxon": {}, "examples": examples, "on_device": device,
            "pairing": pair_rep, "skipped": rep}
    for s in SETS:
        dset = df[df["set"] == s]
        pr = pairs[pairs["photo_id"].isin(set(dset["photo_id"]))]
        wil = wilcoxon(pr)
        head["wilcoxon"][s] = wil
        if dset.empty:
            continue
        sums = table_fidelity(out, df, s, wil)
        head["sets"][s] = {}
        for tk in TRACERS:
            d = leads12(dset[dset["tracer_kind"] == tk])
            if d.empty:
                continue
            ci = cluster_ci(d, "pcc_asis")
            rh = summary(dset[(dset["tracer_kind"] == tk) & dset["rhythm"]]["pcc_asis"])
            head["sets"][s][tk] = {"median_pcc": sums[tk]["pcc_asis"]["median"], "ci95": ci,
                                   "pct_ge_09": sums[tk]["_pcc"]["pct_ge_09"],
                                   "median_shifted_snr_db": sums[tk]["shifted_snr_db_asis"]["median"],
                                   "n_leads": sums[tk]["_pcc"]["n"], "n_unavailable": sums[tk]["_pcc"]["n_unavailable"],
                                   "n_photos": int(d["photo_id"].nunique()), "rhythm_median_pcc": rh["median"],
                                   "rhythm_n": rh["n"]}
    lv = leads12(df[(df["set"] == "ideal") & (df["tracer_kind"] == "learned")])
    gd = {g: summary(lv[lv["group"] == g]["pcc_asis"])["median"] for g in ("dev", "held_out")}
    head["dev_vs_held_out"] = {"dev_median": gd["dev"], "held_out_median": gd["held_out"],
                               "difference": (gd["held_out"] - gd["dev"]) if all(np.isfinite(v) for v in gd.values()) else None}
    (out.root / "headline.json").write_text(json.dumps(clean_json({**out.prov, "headline": head}), indent=2,
                                                       default=str, allow_nan=False), encoding="utf-8")
    print(f"wrote {len(out.written) + 1} files to {out.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
