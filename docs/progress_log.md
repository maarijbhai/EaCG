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

## 0. Where it stands (2026-10-05; desktop numbers updated after §11)

Four PTB-XL records printed with ecg-image-kit and photographed on the
Galaxy A30s; the app digitises all 12 leads + rhythm on the phone.

| record | phone (A30s) PCC | desktop replay PCC | aligned | SNR / shifted | R |
|---|---|---|---|---|---|
| PTB-XL 38 | **0.921** | 0.955 | 0.987 | 8.3 / 13.9 | 0.91 |
| PTB-XL 57 | **0.908** | 0.927 | 0.990 | 7.8 / 13.4 | 0.87 |
| PTB-XL 9  | **0.910** | 0.915 | 0.990 | 8.1 / 15.3 | 0.96 |
| PTB-XL 40 (6 photos) | 0.832 *(10-04, before the fixes)* | 0.955 | 0.992 | 10.6 / 16.2 | 0.93 |

Phone: `results/device/sheet_scores.csv` (fc66d65) — run BEFORE the §11 fix; the desktop column is after it (a972d41), so re-run on the phone to compare. Desktop replay of the
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

## 11. 2026-10-05 — Timing: where the error comes from, and the first fix

**Question:** the tracer reads every record's shape at aligned PCC 0.99, but
plain SNR sits 4–9 dB under shifted SNR. Which part of the time map is off?

- **Registration as truth — inconclusive** (`python/run_timing.py`,
  `results/kitphoto/timing_b3.csv`; c502e1c). Mapping every printed
  sample's true time through the registration onto the page gives the time
  map's own error per lead; it does **not** predict each lead's measured
  shift (r = 0.08 over 106 leads). At a few milliseconds the registration
  homography is not a precise enough truth (the phone lens's slight
  distortion, which a homography cannot model). Kept as a negative result.
- **The traces as the instrument** (`python/run_halfshift.py`,
  `results/kitphoto/halfshift_markers*.csv`; c502e1c). Each half cell's
  measured shift is fitted as errors of the row's four time markers (pulse
  edge, 2.5 / 5.0 / 7.5 s ticks); fit RMS 0.5–5 ms against raw shifts of
  4–10 ms, so the markers explain it. On the **printed PNGs** every marker
  read a uniform **+4 ms**; on photos the pulse ~+10, 2.5 s ~−2, 5.0 s
  +6.5..+12.5 on every photo and row, 7.5 s ~+4.
- **Cause of the uniform part:** cv2's opening with an even (2×2) kernel
  shifts the mask one pixel right. Since locate moved to 2×2 (§8), every
  pulse edge and tick was one pixel late, while b3 — which reads the image,
  not a mask — was not; so the map and b3's trace disagreed by a pixel.
- **Fix** (4c47bd6): `segment.mask_offset_px` subtracted from pulse edges,
  fallback t0, tick positions and the classical tracer's columns; Python
  and Kotlin; golden vectors regenerated; a test that the same drawing read
  through 2×2 and 3×3 openings gives identical markers (fails under the old
  rule). Measured before adopting: printed PNGs plain SNR 9.5–12 →
  17.5–23.7 dB (= shifted SNR).
- **Result** (a972d41), b3, 9 photos / 108 leads: grid path PCC 0.943 →
  **0.952**, leads ≥ 0.9 76 → **92**, SNR 9.7 → **10.3 dB**; corners path
  0.887 → 0.926, SNR 6.7 → 8.6. Classical unchanged (its markers and trace
  had been shifted together). Per record (grid): PTB-XL 38 **0.955**, 57
  **0.927**, 9 **0.915**, 40 0.933–0.975.
- **Still open — the photo-specific part:** refitted markers on photos:
  pulse **+6.3 ms**, 2.5 s tick **−6.4**, 5.0 s **+4.4**, 7.5 s +0.1
  (printed PNGs: all ~0). Not geometry (grid-fit page), not the tracer
  (classical shows it too), not the drawing (absent on the PNGs). Next
  hypotheses: the bar centre vs the trace on a photographed (blurred,
  sharpened) bar; the pulse edge's width rule (`0.25 mm` line assumed).

## 12. 2026-10-06 — The 7 s "cut+score" was one line of the scorer

**Found:** the phone's `cut+score` stage (7,328 ms on PTB-XL 38,
`eacg_sheet_ptbxl38_..._164847_915.csv`) was not the cutting. In
`SheetPipeline.score` the RMSE took each signal's median **inside a
per-sample map**, so every sample re-sorted the whole window: O(n² log n),
5,000 sorts of 5,000 values for the rhythm strip.

**Fix** (Kotlin only; the Python scorer was never affected): medians taken
once. Cutting (`cut`) and truth scoring (`score*`) are now timed separately;
`score*` is harness (limitation 9) and is excluded from the "Done in" total
and the Records tab's total (`Result.pipelineMs`); the CSV still lists it.
No change to any lead's signal or score. `SheetScoreTest`: matches a naive
reference to 1e-12, and 13 rhythm-length scores take 25 ms where the old
rule took **42.6 s** on the desktop JVM (checked: the test fails under it).
98 JVM tests. Expected on the A30s: ~30 s → ~23 s per sheet; to confirm with
the phone re-runs.

## 13. 2026-10-07 — Memory per stage on the phone (NFR2)

`StageMemory.kt`: a background sampler (every 20 ms) reads the native heap
(`Debug.getNativeHeapAllocatedSize`: OpenCV Mats, the LiteRT arena, Bitmap
pixels) and the Java heap in use; each timing lap files the peak since the
previous lap under that stage. The sheet CSV gains `memory_peak_mb`
(`baseline=native/java` then `stage=native/java`, absolute MB) and
`pss_end_mb` (process PSS once at the end); the result screen shows the
native peak; `score_device_sheets.py` carries both into
`results/device/sheet_scores.csv`. Limits: a sampled peak can miss an
allocation shorter than 20 ms (each lap also samples); Java figures include
uncollected garbage. `StageMemoryTest` (4 tests) pins the bookkeeping,
including that a transient in one stage is not charged to the next.
102 JVM tests. **Not yet measured on the A30s** -- next phone runs.

Also from 10-07: a new PTB-XL 57 photo on the phone (sheet filling the
frame, corners at/over the frame edge; grid-fit inliers 0.965 vs 0.996):
median PCC 0.911 (old photo, pre-fix app 0.908), SNR 7.9 / 13.7 dB, rhythm
0.911 (was 0.777); III, aVL, V5 lower, II, V2, V3 higher. Not comparable
like for like (photo and app both changed); `cut` + `score*` 62 + 96 ms on
the phone, confirming §12.

## 14. 2026-10-07 — Whole-sheet quality warnings (FR8)

`ecgproto/sheet_quality.py` + `SheetQuality.kt` (golden-vector port,
`make_sheet_quality_fixtures.py`, 21 cases). Checks the time and geometry
references the sheet was cut with -- anchored on printed features (grid,
ticks, pulses) or physiology, never on the pipeline agreeing with itself:

| code | severity | rule | measured on the 9 photos (3,200 px, grid) |
|---|---|---|---|
| GRID_REFUSED | warn | grid fit asked for, refused | never |
| NO_TICK_SCALE / FEW_TICKS | severe / warn | < 3 / < 6 of 9 ticks | 8-9 found everywhere; the 3x3 phone failure 3 |
| SCALE_DISAGREE | severe | tick vs grid scale > 1% | <= 0.28%; the 3x3 failure ~ -33% |
| TICK_OUTLIER | severe | a tick > 1 mm from the median of that tick over the rows | <= 0.24 mm; ptbxl40_20260929_02 5.49 mm |
| NO_PULSE | warn | time zero from the fallback | 1-4 pulses found everywhere (PTB-XL 38 scores 0.955 from one) |
| LEAD_COVERAGE | warn | < 80% of a lead traced | b3 >= 0.979 |
| LEAD_AMPLITUDE | warn | p-p outside 0.05-6 mV (quality.REJECT_PP_MV) | 0.24-4.11 mV on every phone lead |

Verdict: ok / check (any warn) / unreliable (any severe); the phone shows it
as a green/amber/red banner and writes `quality` + `warnings` into the CSV;
`score_device_sheets.py` and `ml/eval_phone_path.py` carry both.

**Found while measuring: what broke ptbxl40_20260929_02 is one tick.** Row
3's 7.5 s tick was found 52 px (5.5 mm, ~220 ms) early (x 1,873.5 against
1,926 in rows 1-2); that knot squeezed V3, stretched V6, and the rhythm strip
(knots = the rows' mean) inherited it -- the three leads that scored PCC
~0.1-0.5. Not the tracer. TICK_OUTLIER now flags it. The obvious FIX --
drop the outlier and borrow its mates' position -- is a separate change, to
be measured end to end before adoption.

A design error caught by the tests: comparing a tick with the MEAN of its
mates let the one bad tick drag the two good ones 2.7 mm and flag them too;
the median of the three cannot be moved by one. Kotlin mutation check:
mean-for-median fails SheetQualityPortTest.

Tests: `tests/test_sheet_quality.py` (11), `SheetQualityPortTest` (3).
Python 85, JVM 105. Evidence: `results/kitphoto/warnings.csv`
(`python/run_sheet_warnings.py`); lead-level with b3 in
`results/ml/phone_path_b3_512x1024_fp16_grid.csv` (rerun at 098f91d: all 234
lead scores identical to before, max |diff| 0.0). With b3: 8 photos ok, 0
lead-level warnings (no false flags), _29_02 unreliable. The classical
control: 6 of 9 "check" (LEAD_COVERAGE -- it really does lose trace), 2 ok,
_29_02 unreliable.

## 15. 2026-10-07 — A misplaced tick is replaced by its column-mates

**Change:** `segment.reject_tick_outliers` / `SheetLayout.rejectTickOutliers`,
applied in `tick_scale`: when all three rows found boundary k, a tick more
than `TICK_OUTLIER_MM` = 1 mm from the median of the three is rejected (not
counted, not in the scale median) and `segment_leads` borrows its
column-mates' position, the existing rule for a missing tick. With two rows
nothing says which is wrong: kept, and sheet_quality's TICK_OUTLIER (severe)
still fires. A rejection is reported as TICK_REPLACED (warn, naming the
leads). Threshold: good boundaries agree within 0.21 mm (§14).

**Measured before adopting** (b3 desktop replay, 9 photos, grid): only
ptbxl40_20260929_02 moved -- V3 0.099 -> **0.977**, V6 0.103 -> **0.941**,
rhythm 0.505 -> **0.813**, its other leads within +-0.003; the other eight
photos bit-identical. All 108 leads: median PCC 0.952 -> **0.955**, >= 0.9
92 -> **94**, SNR 10.33 -> 10.44 dB (shifted 16.00). Classical unchanged.

Tests: `TickOutlierIsReplaced` (a synthetic sheet with one bar 5.5 mm out;
fails under the old rule, 9 != 8), the rule's thresholds both sides;
golden cases `reject.*` in sheet_port_fixtures (only additions; no existing
value moved), TICK_REPLACED cases in sheet_quality_fixtures. Python 89,
JVM 106.

## 16. 2026-10-07 — UI redesign (docs/ui_plan.md), step 1: theme

UI only; no pipeline code touched. `ui/theme/Color.kt`: the plan's §2.1
tokens (maroon `#7A1F2B` accent, ground `#F7F3F1`, teal `#0F766E` for
digitised data, amber/red verdict colours, pink ECG paper); four renamed to
avoid clashing with Material composables (`CardSurface`, `CardBorder`,
`RowDivider`, `ButtonOutline`). `Type.kt`: IBM Plex Sans 400/500/600/700 and
Mono 400/500 as static TTFs in `res/font` (IBM/plex v6.4.0, OFL 1.1, licence in
`assets/licenses/`), mapped to the mock-up's sizes; `DataText` (Mono 13) for
numbers. `Theme.kt`: one light scheme, `dynamicColor` and dark theme removed.
Window background set to the ground colour. Existing screens now render in
the new colours and type; nothing else changed. Build and 106 JVM tests
green; NOT verified on the A30s.

## 17. 2026-10-07 — UI step 2: navigation shell, Settings, Lab

UI only. `MainActivity.AppShell`: bottom bar Scan / Records in the mock-up's
style (maroon icon + label in a tint pill), Settings and Lab as full screens
over the tabs with Back handling. The Scan tab stays composed at zero size
whenever it is not shown (a digitisation in flight survives); the Lab, once
opened, is kept the same way (a running benchmark survives). `AppSettings`:
prototype mode, SharedPreferences, ON by default, provided as
`LocalAppSettings`; it only hides UI. `SettingsScreen`: the switch, the Lab
entry (prototype mode only), About (version, model file, font licence).
`LabScreen` = `DigitiseScreen(includeWholeSheet = false)`, unchanged in
function (fixtures, deskew + benchmark, rhythm-strip path, tracer
benchmark). Interim Scan tab: the new header + Settings gear over the
existing `SheetSection`, until steps 3-6 replace it. New `ui/Icons.kt` (the
mock-up's own SVG paths as ImageVectors) and `ui/Components.kt` (TopBar,
EaCard, 52 dp full-radius buttons, section label, disclaimer). Build and 106
JVM tests green; NOT verified on the A30s.

## 18. 2026-10-07 — UI step 3: Home, and the Scan flow's controller

`ScanController`: the former SheetSection's state and logic in one holder
at the app shell (outlives tab switches) -- load at 3,200 px
(`UriSource(maxDimension = 3200)`), corners, digitise, save, export; the
pipeline calls, CSV text, export file name (`eacg_sheet_<target|photo>_
<tracer>_<stamp>.csv`, `ScanController.exportName`) and record saved are
unchanged. With prototype mode off a scan uses the defaults (learned, grid
fit, no benchmark). The benchmark target is remembered across launches.
`HomeScreen`: illustration card, Take photo / Choose from gallery (both the
picker until step 9), three tip tiles, Recent (newest 3 records with a
pink-grid thumbnail of their own rhythm strip and the verdict as icon +
text), disclaimer. Until steps 4-6, picking a photo opens an interim panel
with the old controls and result.

Record format: **fields added, none renamed** (quality, warnings,
corners_refined, grid_rms_mm, grid_rules, grid_note, timings, memory,
pss_end_mb, photo_source, t0_x, row_y, ticks, ticks_rejected, cells), all
optional; `SheetPipeline.Result` gained the matching UI-only copies (t0,
row centres, ticks kept / replaced, cell boxes). Tests: a pre-redesign
record still loads; the new fields and the warnings round-trip. 108 JVM
tests green; NOT verified on the A30s.

## 19. 2026-10-07 — UI step 4: Check corners

`CornersScreen` (mock-up artboard 2): the photograph with the detected quad
on the dark ground (`CornerEditor`, restyled: white handles with a 3 dp
corner-red ring, red quad with a 10% fill, the dragged handle and loupe
crosshair in teal; grab radius unchanged at 36 dp, geometry unchanged);
a white sheet with the status ("found automatically" + refined / coarse,
or "could not find the sheet"), the four labelled close-ups
(`CornerCloseUps`, the existing loupe drawing), and in prototype mode
Research options (Learned / Classical segmented control, Grid fit switch)
and Benchmark against (None + every target in assets/kit/targets.txt,
remembered); Retake (back to Home) and Digitise. Help explains where a
corner belongs. The Lab's rhythm-strip panel shares the editor and gets
the same colours (restyle only). Build and 108 JVM tests green; NOT
verified on the A30s.

## 20. 2026-10-07 — UI step 5: progress callback + Digitising screen

`SheetPipeline.run(onProgress = null)`: Stage (key, ms, detail), Page
(preview), Rows (centres), Tile (n of total), Row (trace). Null = the run
exactly as before (every addition guarded). With a listener, time spent in
callbacks (incl. the page preview) is measured and subtracted from the stage
it fell in; cancellation is checked between stages and tiles. Save time is
measured by the controller, outside the pipeline. A cancelled run saves no
record (save comes after the run; a cancellation is re-thrown past
runCatching). `ProcessingScreen`: page with rows in teal as read, band on the
active row, tiles / rows progress, stage list with times. `TileCountTest`
pins tileCount to predictTiled's calls. 109 JVM tests; NOT verified on the
A30s.

## 21. 2026-10-08 — UI step 6: Result screen (Scan and Records)

`ResultScreen` + `ResultData` (built from a live Result or a saved record,
display copies only): top bar (title "PTB-XL n" / "Photo", date, Share via
the share sheet, More -> Delete), underline tabs Leads / Page overlay /
Photo; quality banner (none / amber / red) with "Compare on the page
overlay"; the leads as printed on pink paper at the paper's own scale,
labels at each cell start, ticks as bars (amber if not found or replaced;
plain on records saved before the tick field); details list (grid fit,
lead markers, time zero, tracer and time); Export CSV. Prototype mode
adds per-lead r on the labels, the PTB-XL truth in grey, a Benchmark card
(median lead r, leads >= 0.9) and an expandable Prototype details card
(everything the old result panel printed: corners, calibration, ticks, t0,
geometry, stage times, memory). Export is unchanged for both (the scan:
ScanController.export; a record: its own leads.csv as
`eacg_sheet_<target|photo>_<tracer>_<record id>.csv`). Share writes the
same text to cache/share and hands it over through a FileProvider (cache
paths only). Records now opens records here; the old detail view is gone.
The Page overlay tab shows the page for now (step 7). 112 JVM tests
(ResultDataTest: ticks, cells, memory summary, an old record); NOT verified
on the A30s.

## 22. 2026-10-08 — UI step 7: Page overlay

`PageOverlay` (mock-up artboard 6): the straightened page in its own view
transform (pinch, double-tap, + / − / fit), the digitised rows drawn thin
and dotted in teal (1.8 dp, dash [0.1, 3.2 dp], constant on screen at any
zoom) over the printed trace, t0 and the nine ticks as dashed teal lines
("tick 7.5 s" when zoomed; amber, "borrowed", where a tick was missing or
replaced), lead labels at each cell, a legend, and a lead card ("Lead V5 ·
7.5 to 10.0 s · Timed from the 7.5 s tick") whose previous / next zoom to
each cell. Prototype mode: "Show truth" draws the PTB-XL signal in grey,
placed by cell extent (APPROXIMATE: a record keeps cell boxes, not the
exact time map; labelled so). Records saved before the overlay fields show
the page and the dotted trace only. Display only. 112 JVM tests; NOT
verified on the A30s.

## 23. 2026-10-08 — UI step 8: Records list

`RecordsScreen` (mock-up artboard 7): title, search (name, target, date),
underline filters All (n) · Check timing (n: check or unreliable) ·
Classical, plus Benchmarked in prototype mode; grouped by day (Today,
Yesterday, date); each row a pink-grid thumbnail of the record's own
rhythm strip, name, time · tracer · geometry, the verdict as icon + text,
and in prototype mode the median r in mono. Footer: what a record keeps,
and the count and size on the phone. Tapping opens the shared Result
screen (step 6); Delete is in its More menu. Storage untouched. 112 JVM
tests; NOT verified on the A30s.

## 24. 2026-10-08 — UI step 9: in-app camera (the redesign complete)

Home's Take photo: `ActivityResultContracts.TakePicture` into
`cache/camera/` through the FileProvider (the system camera app; no camera
permission), then exactly the gallery's path -- `UriSource` at 3,200 px with
EXIF orientation. The CSV's existing `photo` field reads `camera WxH`
(gallery: `picked WxH`, as before); the record's `photo_source` says the
same. No camera app: falls back to the picker. One shot is kept in the cache
at a time (the record keeps its own copy).

**The UI plan (docs/ui_plan.md §6) is done: steps 1-9, commits 3a37bb1 ..
this one.** Unchanged throughout: pipeline numerics (golden vectors pass),
the 3,200 px decode, the CSV format / file names / location, RecordStore
(fields only added; old records open), the classical control (prototype
mode), the Lab. The old SheetSection composable is left in SheetPanels.kt,
unused. 112 JVM tests; NOTHING visual verified on the A30s -- see the
plan's §7 acceptance checks.

## 25. 2026-10-08 — Camera photos must leave the phone

Found on the first deliberately bad camera photo (eacg_sheet_photo_model_
20261008_111231.csv: row 1 -- I, aVR, V1, V4 -- 0% traced): an in-app camera
photo lived only in the app cache (cleared at the next shot) and the private
record, so it could not reach data/kit_photos for the desktop replay. Now the
camera writes the ORIGINAL to the gallery (MediaStore Pictures/EaCG, Android
10+, no permission; a cancelled shot's empty entry is deleted; cache below
10), and the Result screen's More menu has "Save photo and page to
Downloads" (the record's photo.jpg as digitised and page.jpg). Formats
untouched. 112 JVM tests; NOT verified on the A30s.

---

## Open items (ordered)

1. **Timing, photo-specific part** (§11): pulse ~+6 ms, 2.5 s tick ~−6,
   5.0 s ~+4 on photos only; and fragile pulse detection on darker rows.
   Plain-vs-shifted SNR gap now ~5.7 dB on the grid path (10.3 vs 16.0).
2. Memory: instrumented (§13), measure on the A30s. (cut+score: fixed and
   confirmed on the phone, §12-13.)
3. More photos per record in worse conditions (side light, angle).
4. Code freeze ~2026-10-17, regenerate everything from one commit, merge
   `prototype-and-dataset-repair` → `main`.
