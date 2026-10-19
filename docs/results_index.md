# Where every result is (for the report)

*Written 2026-10-16. Every number below is in a committed file under `results/`; each
CSV has a `.json` sidecar beside it with the git commit, the parameters and the
library versions (`git_dirty: false` on all the files listed). Paths are from the
repository root. The engineering history behind each one is `docs/progress_log.md`
(section numbers in brackets).*

**To look at the numbers, open these workbooks first.** Sets A, B and C use the SAME statistics, made by the same scripts (`make_report_figures.py`, `make_stats_workbook.py`): the lead is the unit, median [IQR], 95% CI of a median by cluster bootstrap over photographs (2,000 resamples, seed 20261012), Wilcoxon signed-rank for learned vs classical paired by (photograph, lead), an empty PCC counted as a failure. Checked: re-running Sets A and B through the Set C inputs reproduces `results/report/` byte for byte (§44).

| workbook | what is in it |
|---|---|
| `results/report/stats_sets_A_B.xlsx` | Sets A and B: *README* (every method), *Summary 12 leads*, *Summary rhythm strip*, *Tracer comparison*, *Per photo*, *Per sheet*, *PCC by lead*, *SNR by lead*, *Learned vs classical*, *Timing & memory*, *Photo quality (B)*, *Quality vs PCC (B)*, *Per lead (all)*, *Excluded (Set B)* (§37-38) |
| `results/report_set_C/set_C_robustness/stats_set_C.xlsx` | Set C, the same sheets as A/B: *README*, *Summary 12 leads* (with 95% CIs), *Summary rhythm strip*, *Summary by condition (C)* (tilt / shadow / crumple, with CIs), *Tracer comparison* (Wilcoxon), *Per photo*, *Per sheet*, *PCC by lead*, *SNR by lead*, *Learned vs classical*, *Timing & memory*, *Per lead (all)* (§44) |
| `results/report_set_C/stats_extras.xlsx` | Everything else: *Set C runs* (one row per export, incl. the rescues), *Rescue before-after*, *Rescue scores (leads)*, *Orientation* (64/68), *Orientation rule v1* (rejected), *Orientation sideways* (76/76), *t0 guard replay*, *t0 guard what-if b3* / *cl* (in-sample) |

---

## The three sets

| set | what | runs | where the exports are | photographs |
|---|---|---|---|---|
| **A** clean | the 12 printed-sheet images, no camera | 12 x (learned + classical) | `data/device_exports/Set A/` | `data/kit_photos/SetA_Clean/` |
| **B** ideal | the 12 sheets photographed in good light, 2 photos each | 24 x 2 | `data/device_exports/Set B/` | `data/kit_photos/setB/` |
| **C** robustness | tilt, shadow, crumple | 6 + 4 + 7 | `data/device_exports/Set C Tilt/`, `Set C Shadow/`, `Set C Crumple/` | `data/kit_photos/SetC_Tilt/`, `SetC_Shadow/`, `SetC_Crumple/` |
| C rescue | the failed Set C photographs re-run with the t0 guard (in-sample) | 5 + 1 | `data/device_exports/Crumple ResQ/`, `Tilt ResQ/` | (the Set C photographs) |

Which runs belong to which set: `python/test_set.csv` (A, B) and the README in each
Set C folder (run -> photograph, matched by the recorded corners). Excluded runs are
in each folder's `_excluded/` with a README saying why. The data itself is not in git.

---

## Chapter 5 section by section

### 5.1 Experiment conditions
- The 12 test sheets -- PTB-XL diagnosis (codes, likelihood, superclass), age, sex,
  fold, the report text, and every photograph of each in Sets A, B and C:
  `results/kitphoto/test_sheets.csv` (`python/list_test_sheets.py`).
- The sets: the table above; Set B photo quality (frame fill, px/mm, tilt, light):
  `stats_sets_A_B.xlsx`, sheet *Photo quality (B)*, and *Quality vs PCC (B)* (§36).
- Metrics, how each is computed: the README sheets of the workbooks; the scorer is
  `python/score_12lead.py` (`score_lead`), RMSE and SNR on median-removed signals (§38).

### 5.2 Page geometry and grid fit
- Grid-fit distortion vs registration, 0.27-1.75 mm -> 0.035-0.094 mm:
  `results/kitphoto/geometry.csv` (§4).
- Effect on lead fidelity (PTB-XL 40, 0.865 -> 0.966): progress_log §4 and
  `results/ml/sheets_b3_corners_auto.csv` / `sheets_b3_corners_truth.csv`.

### 5.3 Layout accuracy
- The nine ticks, why some are missed: `results/kitphoto/ticks.csv` (§6).
- One-pixel mask offset: `results/kitphoto/halfshift_markers*.csv` (§11).
- Quality warnings measured: `results/kitphoto/warnings.csv` (§14).
- Rejected alternatives: `results/kitphoto/timemap_leads.csv`, `timemap_rows.csv`,
  `timing_b3.csv` (§5, §11).
- Orientation (64 / 68 session photographs right; old rule 58 or 64):
  `results/kitphoto/orientation_sessions.csv`; the rejected first rule
  `orientation_sessions_rule_v1.csv`; the quarter-turn test 76 / 76
  `orientation.csv` (§35, §39).
- Guarded time-zero fallback: where it fires `results/kitphoto/t0_guard_replay.csv`;
  what-if (in-sample) `t0_guard_whatif.csv` (classical), `t0_guard_whatif_b3.csv` (§41).

### 5.4 Trace extraction (learned vs classical)
- Validation of the learned tracer (row error 0.33-0.44 px): `results/ml/train_b3.csv`
  (+ `train_b0..b2`, model A `train_a1..a3`); label ceiling `results/ml/label_ceiling.csv`.
- **Set A and Set B, learned vs classical**:
  - tables `results/report/fidelity_clean.tex/.csv`, `fidelity_ideal.tex/.csv`
    (every measure, median [IQR], Wilcoxon);
  - figures `results/report/pcc_by_lead.pdf`, `plain_vs_aligned.pdf`,
    `pcc_by_quality.pdf`, `pcc_by_set.pdf`, examples `example_clean_*.pdf`,
    `example_ideal_*.pdf` (+ `example_ideal_photo.jpg`);
  - headline numbers `results/report/headline.json`;
  - every lead: `results/device/sheet_scores.csv`.

| | learned | classical |
|---|---|---|
| Set A: median lead PCC [IQR] | 0.994 [0.99, 1.00] | 0.971 [0.95, 0.98] |
| Set A: leads >= 0.9 | 99% | 93% |
| Set B: median lead PCC [IQR] | 0.947 [0.91, 0.97] | 0.646 [0.39, 0.88] |
| Set B: aligned PCC / SNR / shifted SNR | 0.992 / 10.0 / 15.5 dB | 0.720 / 2.0 / 2.4 dB |
| Set B: leads >= 0.9 | 80% | 21% |

### 5.5 Robustness analysis (Set C)
- **Set C on its own** (the same tables and figures as A and B):
  `results/report_set_C/set_C_robustness/` -- `fidelity_robustness.tex/.csv` (every
  measure, median [IQR], Wilcoxon), `pcc_by_lead.pdf`, `plain_vs_aligned.pdf`,
  `pcc_by_quality.pdf`, `latency_stages.pdf`, `on_device.tex`, examples
  `example_robustness_*.pdf`, `headline.json`, and the workbook `stats_set_C.xlsx`.
- **Set C against A and B**: `results/report_set_C/compare_sets/` --
  `robustness.tex/.csv/.pdf` (each Set C lead minus the median of the same record's
  Set B photos, same lead and tracer: median difference with a 95% cluster-bootstrap
  CI, per condition), `compare_sets.tex/.csv` (A, B, C side by side), `pcc_by_set.pdf`,
  `headline.json`. The figures and `robustness.tex` show only the conditions that have
  photographs (`--present-conditions-only`, §45); the CSVs keep every row.
- **Per condition, with the gate**: `results/report_set_C/setC_condition_summary.csv/.tex`
  (`make_setC_condition_summary.py`, §45) -- photos, learned runs, reliable runs
  (ok/check), failures by cause, learned and classical median PCC over the reliable
  runs and their change against the same records' Set B, and the gate row (7 of 7
  failures caught, 3 of 10 good runs false alarms).
- Inputs: `results/device/sheet_scores_all_sets.csv` + `python/test_set_all.csv`
  (`make_setC_inputs.py`: the A/B rows unchanged + the Set C score files); every
  lead per condition: `results/device/sheet_scores_setC_tilt.csv`, `_setC_shadow.csv`,
  `_setC_crumple.csv`.
- Every run (facts: verdict, warnings, ticks, t0 source, guard): `results/device/setC_runs.csv`.
- The rescue (IN-SAMPLE, not in the Set C statistics): before / after per photograph
  `results/device/setC_crumple_rescue_before_after.csv`; every lead
  `sheet_scores_setC_crumple_rescue.csv`, `_setC_tilt_rescue.csv` (§42).
- Do not use `results/report/robustness.*` (made before Set C existed; empty) or
  `results/device/setC_crumple_summary.csv` (§40; medians without CIs -- superseded
  for statistics by the files above, the medians agree).

| Set C (lead = unit), median [95% CI] | learned | classical |
|---|---|---|
| all conditions: lead PCC | 0.848 [0.15, 0.94] | 0.448 [0.19, 0.56] |
| leads >= 0.9 | 40% | 4% |
| learned vs classical (Wilcoxon, 154 pairs) | p = 1.3e-15 | |
| tilt: lead PCC | 0.946 [0.19, 0.96] | 0.563 [0.44, 0.59] |
| shadow: lead PCC | 0.932 [0.10, 0.97] | 0.556 [0.55, 0.75] |
| crumple: lead PCC | 0.211 [0.01, 0.88] | 0.183 [0.05, 0.51] |

| change from the same record's Set B photos, median [95% CI] | learned | classical |
|---|---|---|
| tilt | -0.004 [-0.77, +0.01] | -0.10 [-0.27, +0.03] |
| shadow | -0.02 [-0.86, -0.001] | -0.20 [-0.28, -0.14] |
| crumple | -0.76 [-0.94, -0.08] | -0.49 [-0.72, -0.22] |

The CIs are wide because there are few photographs (4-7 per condition): describe
the failure modes and give the CIs; do not present Set C rates as precise. All 7
failed learned runs (median lead PCC < 0.5) were marked unreliable (`setC_runs.csv`).

### 5.6 Whole-sheet performance and published work
- Per sheet and per lead name: `stats_sets_A_B.xlsx` (*Per sheet*, *PCC by lead*, *SNR by lead*, *Per lead (all)*);
  `results/report/pcc_by_lead.csv`.

### 5.7 Performance on edge hardware (the A30s)
- Time and peak memory per stage (learned): `results/report/on_device.tex/.csv`;
  per-stage figure `results/report/latency_stages.pdf/.csv`; per export: the
  workbook's *Timing & memory* sheet. Set B: ~21.5 s per sheet (trace 19.6 s),
  peak native ~506 MB; classical 2.9 s, 150 MB (§33, §36).
- Tracer benchmark per tile (FP16 1.56 s, INT8 0.95 s): `results/device/tracer_bench_20261004_141736.csv`;
  model export checks `results/ml/export_b3.csv`.
- Orientation check time on the phone: the `# orient_ms:` header line of every export
  since 593a269 (3.3-4.5 s, Set C Crumple; `setC_runs.csv` column `orient_ms`).

### Diagnosis classifier (if it goes in)
- The 7 training runs: `results/ml/clf_*.csv/.json` (test macro AUROC in each sidecar);
  deployed model parity `results/ml/clf_export.*`; degradation curve
  `results/ml/clf_degradation.*`; on the phone's sheets `results/ml/clf_sheets.*` (§29-33).

---

## How each file was made (to regenerate or cite)

| files | script (from `python/` unless stated) |
|---|---|
| `results/device/sheet_scores.csv` (Sets A, B) | `score_device_sheets.py` -- do **not** re-run before the report: it re-scores every folder |
| `results/report/*` | `make_report_figures.py` (+ `make_stats_workbook.py`) |
| `results/device/sheet_scores_setC_*.csv` | `score_device_set.py "<folder>"` (the same scorer, one folder; `--targets target_corrections.csv` for the one mis-labelled run) |
| `results/report_set_C/*` | `make_setC_inputs.py`, then `make_report_figures.py --scores results/device/sheet_scores_all_sets.csv --manifest test_set_all.csv --sets robustness` (and `--sets clean ideal robustness`), `make_stats_workbook.py --sets robustness --scores ... --manifest ...`; `make_extras_workbook.py`; `compare_sets/` with `--present-conditions-only` (§45); `make_setC_condition_summary.py` |
| `results/device/setC_runs.csv`, `setC_crumple_*.csv`, `setC_crumple_rescue_before_after.csv` | `summarise_set_c.py`, `summarise_crumple.py`, `rescue_before_after.py` |
| `results/kitphoto/orientation_sessions*.csv` | `run_orientation_sessions.py` |
| `results/kitphoto/t0_guard_*.csv` | `run_t0_guard.py` (`--whatif`; b3 in WSL) |
