# Engineering log — whole-sheet digitisation (from 2026-10-02)

What was done, in order, why, what it measured, and where the evidence is.
Every number here is in a committed CSV (+ JSON sidecar with the git commit
and parameters) under `results/`; the commit hash after each entry is the
code or result commit that holds it. Experiments that **failed** are logged
too, with the reason they were rejected — they belong in the thesis.

Abbreviations: **PCC** median lead Pearson correlation against the PTB-XL
truth, as-is (no alignment) unless stated; **aligned** = after a bounded
±100 ms lag / ±10% scale search; **SNR / shifted** = dB, median-removed,
shifted allows ±50 ms (PhysioNet Challenge 2024 convention); **R** = R-peak
height ratio (target ≥ 0.90); **b3** = the learned tracer (model B, run 3),
exported FP16 LiteRT exactly as the app bundles it.

---

## 0. Where it stands (2026-10-05)

Four PTB-XL records printed with ecg-image-kit and photographed on the
Galaxy A30s; the app digitises all 12 leads + rhythm on the phone.

| record | phone (A30s) PCC | desktop replay PCC | aligned | SNR / shifted | R |
|---|---|---|---|---|---|
| PTB-XL 38 | **0.921** | 0.925 | 0.987 | 8.3 / 13.9 | 0.91 |
| PTB-XL 57 | **0.908** | 0.919 | 0.990 | 7.8 / 13.4 | 0.87 |
| PTB-XL 9  | **0.910** | 0.869 | 0.990 | 8.1 / 15.3 | 0.96 |
| PTB-XL 40 (6 photos) | 0.832 *(10-04, before the fixes)* | 0.955 | 0.992 | 10.6 / 16.2 | 0.93 |

Phone: `results/device/sheet_scores.csv` (fc66d65). Desktop replay of the
phone's exact path (3,200 px decode, automatic corners, grid fit, b3 FP16):
`results/ml/phone_path_b3_512x1024_fp16_grid.csv` (3e555bb). Classical
tracer on the same path: PCC 0.46–0.87, R 0.15–0.83.

**What limits it now is timing, not the tracer**: aligned PCC is 0.99 on
every record, shifted SNR is 4–9 dB above plain SNR. Two systematic timing
errors recur across records — V1–V3 read 6–14 ms early, and the first
column (I/II/III) shifts — plus a fragile time zero (calibration pulses
missed on darker rows).

Time per sheet on the A30s ≈ 30 s: grid fit 1.8–2.8 s, model 19–20 s,
cut+score ~7 s (unoptimised Kotlin). Memory not yet measured.

---

## 1. 2026-10-02 — The goal changes to the whole sheet

The user and supervisor moved the target from the lead II rhythm strip to
all 12 leads of a standard 3×4 + rhythm printout, with the learned tracer
on the phone. Standard 3×4 only; lead identity by position.

## 2. 2026-10-03/04 — Whole-sheet pipeline, desktop then phone

- **Learned tracer on the phone.** b3 exported to LiteRT (FP16 exact
  against Keras; INT8 + float decode +0.3 px row error). A benchmark button
  on the A30s: FP16 1.56 s per tile, ~18.7 s per sheet; INT8f 0.95 s.
  Exact against the desktop. (5a80f5f, d9889b7;
  `results/device/tracer_bench_*.csv`)
- **Accuracy over speed** (user decision, 10-04): latency and memory became
  reported characterisation, not gates. (08b68a5; `docs/ml_plan.md` §4)
- **Separator ticks set each cell's time scale.** The nine lead-change bars
  (3 rows × 2.5/5.0/7.5 s) are knots in a per-row time map; a single
  page-wide scale had left later columns tens of ms early. A missing tick
  borrows its column-mates' position. (4ce4c9d, 5c9b7d5)
- **Automatic corners** (`page.find_sheet_corners`: paper mask → grid
  texture → RANSAC line per edge → edge refinement) and a **registration
  truth** to score corners against (`corner_truth`: ECC homography to the
  printed PNG — an evaluation instrument only). (b040b1f)
- **One reference for the phone**: `ecgproto.sheet.digitise_sheet` does on
  the desktop exactly what the app does; the Kotlin port
  (`SheetLayout/SheetCanonical/SheetCorners/SheetPipeline.kt`) is
  golden-vector tested against it. The port found a NaN bug in the Python
  reference (`time_at`), fixed in both. (be45911, b85ac1d, 77ef289)

## 3. 2026-10-04 — SNR added to the scorer

`score_lead` reports plain and shifted SNR on median-removed signals (a
printed trace has no absolute baseline). First phone CSV (PTB-XL 40):
median SNR 5.5 dB, shifted 8.0 dB — the gap is timing. (bc909d6)

## 4. 2026-10-04 — Grid-fit geometry (the biggest single gain)

**Why:** four corners fix a homography from four points, so corner error
spreads over the page as scale and keystone — a time base that drifts lead
by lead. Registered (true) corners had lifted median lead PCC 0.913 → 0.969.

**What:** `ecgproto/gridfit.py` fits the homography to every 5 mm grid-rule
sample (~55 × 37 rules): redness response, band median profiles, peaks,
rules indexed and chained between bands, Huber-reweighted fit; the photo is
warped once onto the regular page.

**Measured** (`results/kitphoto/geometry.csv`, against registration):
page distortion 0.27–1.75 mm RMS → **0.035–0.094 mm**; 95th-percentile time
error 16–103 ms → 2–6 ms; automatic corners + grid = true corners + grid to
0.002 mm (the corners stop mattering). End to end, b3, PTB-XL 40, 72 leads:
PCC 0.865 → **0.966**, leads ≥ 0.9 30 → 53, SNR 5.8 → **11.4 dB**.
(3aa64d2, 16f9dd0, 7931217)

- A first version failed on two photos (60% inliers): one spurious peak
  shifted every later rule's index. Fixed by dropping off-pitch peaks
  before counting.
- **Scale hint** (b3f8e96): the grid's px/mm now places the pulse and tick
  searches; the duration anchor (needs the rhythm strip's last ink) had
  misplaced them on one photo (0/9 ticks).
- **Desktop-replay correction** (38258e3): `eval_phone_path.py` had been
  run on full-resolution files; the phone decodes to 3,200 px. At 3,200 px
  one photo collapsed with corners only — hidden before.

## 5. 2026-10-04 — Per-row time maps: diagnosed, **not adopted**

Asked: is there a time-scale error *inside* each row? (`ecgproto/timemap.py`,
`ml/run_timemap.py`, `results/kitphoto/timemap_*.csv`; 5e5cdd7, 5d1d80a)

- **Diagnosis**: per-lead best shift, fitted against column (0–3). On the
  grid page, b3: median |slope| 1.8 ms/cell (0.07%), photo-specific sign —
  **no systematic within-row scale error**. On the corner page: up to
  31.6 ms/cell, varying by row — perspective, which the grid fit removes.
- **Row start + row end pinned to 10 s** (the proposed fix): 14 leads worse
  by > 1 dB vs 5 better; 24/78 leads unavailable (row end or own pulse not
  found); the last column's V5/V6 shifted 56–200 ms. **Rejected.**
- **Uniform grid scale**: SNR −4.5 dB. Revealed that time zero (the pulse
  edge) sits ~10 ms early; ticks absorb it, a uniform map cannot.
- **Time zero from the ticks at the grid scale** (4b0a36c, bf220b8): mixed
  (30 worse / 38 better); the first-column hypothesis not supported.
  **Rejected.** A side finding kept: cv2's opening with an EVEN kernel
  shifts the mask +1 px (worth +0.8 dB when undone).
- **Open**: V1–V3 read 6–10 ms early on photos (0 ms on the clean print),
  both tracers, every map.

Tests: `tests/test_timemap.py`, mutation check `tests/mutate_timemap.py`
(21 of 21 mutations caught).

## 6. 2026-10-04 — The nine ticks, explained

`python/run_ticks.py` mirrors the tick search window by window
(`results/kitphoto/ticks.csv`). Missing ticks are the faintest bars: e.g.
darkness 154–165 against thresholds 141–151, binarised 0.36–0.42 mm wide,
under the 0.45 mm the search requires. All nine are printed identically
(0.76 mm on the clean image) — it is the print or photo, almost always the
top-right 7.5 s tick (row 1, before V4): printer fading at that edge.
(5acb972, ef78943)

## 7. 2026-10-04 — Records tab

Every digitisation is saved privately on the phone (photo, rectified page,
row traces, 12-lead CSV, metadata) and can be reopened: Leads, Page + trace
(the digitised trace over the page, pinch-zoom), Photo; export and delete.
`RecordStore.kt`, `RecordsScreen.kt`, `RecordStoreTest` (round-trip through
the exported CSV). (9143fca)

## 8. 2026-10-05 — Second record; a phone failure, diagnosed and fixed

- **PTB-XL 9, first photo on the phone**: median PCC 0.807, aligned 0.989 —
  the tracer generalised; the loss was timing (last column stretched 1–2%:
  corner geometry, grid fit not yet on the phone).
- **PTB-XL 9, new photo: every lead lost** (phone: 3/9 ticks, 6.52 px/mm,
  PCC −0.01). Cause: `locate_sheet` cleaned its mask with a **3×3
  opening**, which shredded the 2–3 px trace (56% of rhythm-band columns
  kept ink vs 94% at 2×2); the rhythm strip seemed to end three-quarters of
  the way across, the scale read 7.13 px/mm (true ~9.9), and every search
  placed with it missed. **Fix**: 2×2 opening, in Python and Kotlin. Same
  photo, b3: PCC 0.04 → 0.843 (corners), 0.940 (grid). Test fails under the
  old rule. (c1168e6, 65b2973, fab95ff)
- `python/score_device_sheets.py` scores every phone 12-lead export the
  same way as the desktop (`results/device/sheet_scores.csv`).

## 9. 2026-10-05 — Grid fit on the phone

`GridFit.kt`, a step-for-step port; golden vectors from the PTB-XL 9 photo
(band profiles in, every peak, index, chain offset and the homography
checked). One reference change for portability: SciPy's `find_peaks` breaks
ties with an unstable sort, which no port can reproduce; `gridfit` now
breaks them stably (geometry unchanged to 0.001 mm). (c8195c1)

**Cost of the 2×2 locate fix, measured** (f62017e): on PTB-XL 40 the grid
path moved 0.966 → 0.955 PCC, 11.4 → 10.6 dB — the 2×2 mask keeps grid-rule
fragments that fall in the pulse search on darker rows (pulses then missed;
PTB-XL 9: 2 of 4 found). Worth it: no sheet is lost entirely any more.
Tried and **rejected**: "first ink run of pulse length" — grid fragments
can be pulse-length too.

## 10. 2026-10-05 — Records 3 and 4, and the phone matches

PTB-XL 38 and 57 photographed, registered (ECC 0.947 / 0.940 against
0.695 / 0.645 for a wrong record's sheet — identity confirmed). Desktop
replay with grid fit: 0.925 / 0.919 (corners only: 0.769 / 0.595). The
phone: **0.921 / 0.908**. (304cf91, 3e555bb, fc66d65)

The automatic corner finder puts the **top** corners 5–7 mm off on all
three new photos (refinement fails, "coarse only"); cause not yet
diagnosed; the grid fit makes it irrelevant to the result.

---

## Open items (ordered)

1. **Timing**: V1–V3 6–14 ms early (3 records), first-column shifts,
   fragile pulse-based time zero. Target: close the 4–9 dB plain-vs-shifted
   SNR gap.
2. Memory measurement on the A30s; cut+score (~7 s) is slow Kotlin.
3. More photos per record in worse conditions (side light, angle).
4. Code freeze ~2026-10-17, regenerate everything from one commit, merge
   `prototype-and-dataset-repair` → `main`.
