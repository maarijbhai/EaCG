# CLAUDE.md — Project context

## What this project is

Final-semester engineering thesis. The goal is an **Android application
that photographs a paper 12-lead ECG printout, digitises it into a
waveform, and optionally classifies it** — with as much of the pipeline
as possible running **on-device**, rather than on a server.

The thesis contribution is a **deployment characterisation**, not a novel
architecture: existing work solves ECG digitisation offline with heavy
models, and the open question is whether that pipeline survives
compression onto consumer mobile hardware, and at what cost in latency,
memory and reconstruction fidelity.

## Current state — read this first (updated 2026-10-06)

**Read these too.** `docs/progress_log.md` is the dated engineering log since
2026-10-02 (every change, its measured reason, commit and results file, and
the REJECTED experiments) -- update it with every change (the user's
standing request). `docs/executive_summary.md` is the plain-language summary
(with analogies) the user is writing the report from. Pipeline diagram
(published): https://claude.ai/artifact/FxMTo4UfoKaPh1YnuMmqNk

**Requirement numbering — read this before citing an FR.** Two schemes are
in play and they collide. The thesis report uses a flat **FR1–FR11**
(FR1 acquisition, FR2 deskew, FR3 grid, FR4 lead segmentation, FR5 trace
segmentation, FR6 vectorisation, FR7 calibration, FR8 quality gate,
FR9 on-device execution, FR10 export, FR11 latency reporting), and **that
is the authoritative one.** `docs/functional_requirements.tex` uses a
nested FR-1.1…FR-8.7 in which FR-8.x is *deployment characterisation* and
the quality gate is FR-3.4. So "FR8" means the quality gate in the report
and something entirely different in the tex table. Say which you mean.

### Goal and calendar

Since 2026-10-02 the goal is the **whole standard 3x4 + lead II rhythm
printout -- all 12 leads -- with the learned tracer, on the phone** (Galaxy
A30s, SM-A307FN). Supervisor agreed. Standard 3x4 only (lead identity by
position). **Supervisor meeting Wednesday 2026-10-07. Code freeze ~10-17,
write-up after; deadline end of October (movable).** The user writes the
report from `docs/executive_summary.md`; `docs/thesis/methods_draft.md` is
their own draft (untracked -- do not commit or edit it without asking).

### Where it stands (numbers)

Median of each record's 12 leads, b3, grid fit, automatic corners:

| record | phone (A30s)* | desktop replay, latest | aligned | SNR / shifted |
|---|---|---|---|---|
| PTB-XL 9  | 0.910 | 0.915 | 0.99 | ~8 / ~15 dB |
| PTB-XL 38 | 0.921 | 0.955 | 0.99 | ~8-11 / ~13-14 |
| PTB-XL 57 | 0.908 | 0.927 | 0.99 | ~8 / ~13 |
| PTB-XL 40 (6 photos) | -- | 0.933-0.975 | 0.99 | ~9-13 / ~15-18 |

\*Phone runs (`results/device/sheet_scores.csv`) were made BEFORE the
one-pixel fix (4c47bd6); the desktop column is after it
(`results/ml/phone_path_b3_512x1024_fp16_grid.csv`, a972d41). All 9 photos /
108 leads, grid path: PCC **0.952**, >= 0.9 **92/108**, SNR **10.3 / shifted
16.0 dB**. Classical (the control): PCC 0.46-0.87, R 0.15-0.83. The phone
matches the desktop to ~0.01 PCC on the same method. **What limits it is
timing, not reading:** aligned PCC 0.99 everywhere; plain SNR ~5.7 dB under
shifted. Time per sheet on the A30s ~30 s (grid fit 1.8-2.8 s, model 19-20 s,
cut+score ~7 s -- a per-sample median in the scorer, fixed 10-06, progress_log
§12; confirmed on the phone 10-07: cut 62 ms). Memory instrumented 10-07
(§13), not yet measured.

### The app UI (redesigned 2026-10-07/08, docs/ui_plan.md, progress_log §16-24)

Scan tab: Home → Check corners → Digitising → Result (`ScanController` holds
the flow; same pipeline calls, CSV and record as the old SheetSection).
Records tab opens the same `ResultScreen` (Leads / Page overlay / Photo).
Settings: **prototype mode** (ON by default; hides research UI only) and the
**Lab** (the old DigitiseScreen: fixtures, deskew, rhythm path, tracer
benchmark). Benchmark target, tracer and grid fit are chosen on the Corners
screen in prototype mode. `SheetPipeline.run(onProgress)` is display-only and
null by default. Record format: optional fields added, none renamed.

### The pipeline as it is

Python reference = `ecgproto.sheet.digitise_sheet`; phone = `SheetPipeline.kt`,
golden-vector tested against it.

1. Decode to 3,200 px long side (`ImageSource.kt`). Desktop replays MUST
   resize the same way -- a full-resolution replay once hid a failure.
2. Corners: `page.find_sheet_corners` / `SheetCorners.kt` (paper mask ->
   grid texture -> RANSAC lines -> refine). Top corners come out 5-7 mm off
   on the new photos ("coarse only"); irrelevant after the grid fit.
3. Rectify from the corners, then the **grid fit** (`gridfit.py` /
   `GridFit.kt`): homography to every 5 mm rule (redness, band medians,
   stable-tie peaks, indexing + chaining, Huber), the photo warped once.
   Distortion 0.27-1.75 mm -> 0.035-0.094 mm (`results/kitphoto/geometry.csv`).
   On by default ("Grid fit" chip); a refused fit falls back to corners and
   says why.
4. Locate (`segment.locate_sheet`): **2x2 opening** (`LOCATE_OPEN_KERNEL`; a
   3x3 one shredded PTB-XL 9's thin trace and the phone lost every lead),
   rows by layout spacing, time zero = median pulse falling edge; the grid's
   px/mm is the SEARCH scale (`scale_hint`).
5. Ticks (`segment.tick_scale`): nine separator bars (3 rows x 2.5/5.0/7.5 s)
   are the per-row time knots; a missing tick borrows its column-mates. The
   usual miss is row 1 at 7.5 s (top right): faint print, not code.
6. **Mask offset**: cv2's opening with an EVEN kernel shifts the mask +1 px;
   `segment.mask_offset_px` / `SheetLayout.maskOffsetPx` is subtracted from
   pulse edges, ticks and classical trace columns (b3 reads the image, so it
   was never shifted). Clean-print plain SNR ~11 -> ~22 dB.
7. Trace per row: **b3** (canonical 8 px/mm, 64 mm window, tiles 512x1024,
   LiteRT FP16 `assets/ml/b3_512x1024_fp16.tflite`) or classical (control).
8. Cut into 13 leads on the knots (`cell_signal`), mV about each cell's
   median; the Records tab (`RecordStore`/`RecordsScreen`) saves every run.

### The model

**b3 is FROZEN** (`out/ml_runs/b3/best.weights.h5`, epoch 4; 389,580 params;
trained on 2,000 fold 1-8 records = 8,000 row bands, validated on 300 fold-9
records = 1,200 bands; seven grid styles; camera + scan augmentation).
Lineage b0 -> b1 (camera aug) -> b2 (all rows, 64 mm) -> b3 (scan path).
Model A (a1-a3, mask + centroid) is the ablation (ceiling R 0.81). Val row
error 0.33-0.44 px, R 0.97. Export: FP16 exact (0.82 MB); INT8 + float decode
("int8f") +0.3 px. Phone benchmark: FP16 1.56 s/tile, int8f 0.95 s/tile,
exact against the desktop.

### Decisions in force

- **Accuracy over speed** (user, 10-04): latency/memory are reported, not
  gates; do not trade accuracy for speed. Raise with the supervisor.
- ML Kit Document Scanner dropped (pending supervisor agreement).
- The classical path is the control; never delete or "improve away" it.
- One change at a time; measure end to end on ALL photos (desktop replay of
  the phone's path, b3) before adopting; port to Kotlin in the same commit
  with regenerated golden vectors and a test that fails under the old rule.

### Rejected experiments (details: progress_log §5, §9, §11)

Row end pinned to 10 s (14 leads worse, 24/78 unavailable); uniform grid time
scale; time zero from the ticks (mixed); "first pulse-length run" pulse
detection (picks grid fragments); registration as a ms-level timing truth
(r = 0.08 with measured shifts -- lens distortion).

### Open problems (ordered)

1. **Photo-only time-marker errors** (`python/run_halfshift.py`, a972d41):
   pulse edge ~+6 ms late, 2.5 s tick ~-6, 5.0 s ~+4, 7.5 s ~0 on photos;
   ~0 on the printed PNGs. Not geometry, not the tracer, not the drawing.
2. **Pulses missed on darker rows**: the 2x2 mask keeps grid-rule fragments
   in the pulse search slab (PTB-XL 9: 2 of 4 found), so time zero is fragile.
3. ~30 s per sheet; memory unmeasured; the phone runs predate the offset fix.
4. Evidence: 4 records; 9, 38 and 57 have one good-light photo each.

### NEXT -- the plan agreed with the user for Mon-Tue (before the Wed meeting)

1. **User**: reinstall the APK and re-run PTB-XL 9, 38, 57 on the phone;
   CSVs -> `data/device_exports/`; then `python python/score_device_sheets.py`
   and commit. Also 4-6 harder photos (side light, hand shadow, ~20 deg
   tilt, curled sheet).
2. ~~Whole-sheet quality warnings~~ **built 10-07** (`sheet_quality.py` /
   `SheetQuality.kt`, progress_log §14). ptbxl40_20260929_02 was broken by ONE
   misplaced tick; **fixed 10-07** (outlier tick replaced by its column-mates,
   §15): V3 0.10 -> 0.98, V6 0.10 -> 0.94; all 108 leads 0.952 -> 0.955, 94/108.
3. ~~cut+score from ~7 s to < 1 s~~ **done 10-06** (progress_log §12; the scorer's
   per-sample median; `score*` is now timed as harness).
4. ~~Memory instrumentation~~ **built 10-07** (`StageMemory.kt`, progress_log §13);
   the phone runs now record it -- not yet measured.
5. **One-page supervisor summary** (an artifact) from committed numbers once
   the phone reruns are in.
6. Robust pulse detection (open problems 1-2); present as in progress.

Questions for the supervisor: accuracy over speed; ML Kit dropped; are 4
records enough; what counts as "good enough" (e.g. phone median PCC > 0.9);
the timeline.

### How to do things (exact)

- **New photo**: the user drops `ptbxl<id>_<date>_<nn>.jpg` in
  `data/kit_photos/` (rename it if it doesn't follow that, and say so).
  Register with REGISTRATION corners: automatic corners at 3,200 px scaled
  back to full size, then `corner_truth.find_true_corners(full_photo,
  Target(name).sheet, auto, Target.px_per_mm)` (also try a wrong record:
  ECC ~0.65-0.70 against ~0.94 confirms identity), then
  `python score_kit_photo.py --add FILE --target ptbxl<id>_kit_500hz_300dpi
  --corners "x,y x,y x,y x,y" --note "..."`. Commit `python/kit_photos.json`
  before scoring.
- **Desktop replay of the phone** (needs TF, so WSL): from Git Bash
  `export MSYS_NO_PATHCONV=1; wsl -d Ubuntu -- bash /mnt/c/dev/ECG-App/ml/wsl_run.sh python ml/eval_phone_path.py --geometry grid`
  (and `--geometry corners`). NOT `bash ml/wsl_run.sh ...` from Git Bash --
  that runs outside WSL and the venv is missing.
- **Phone exports**: `python python/score_device_sheets.py` ->
  `results/device/sheet_scores.csv` (every `eacg_sheet_*.csv`).
- **Diagnostics**: `python run_geometry.py` (geometry vs registration),
  `python run_ticks.py` (each of the nine ticks), in WSL:
  `python/run_halfshift.py [--clean]` (marker errors from half-cell shifts),
  `python/run_timing.py --b3`, `ml/run_timemap.py` (time-map comparisons).
- **Golden vectors**: `python make_sheet_port_fixtures.py` and
  `python make_grid_port_fixtures.py` (from `python/`), then the JVM tests.
- **Results discipline**: commit code first, then regenerate (sidecar
  `git_dirty: false`); every result CSV has a JSON sidecar.
- **Tool quirks**: the Bash tool's heredocs collapse backslash escapes and
  choke on very long bodies -- write files with the Write/Edit tools.
  Foreground `sleep` is blocked. Long training runs: the user runs them by
  hand in a WSL terminal; no GPU jobs while one trains (~1.8 GB usable).

### Earlier: the rhythm-strip photo path (still valid; measured reasons)

**The photo path, and the measured reasons for each of its steps** (each
has a commit message with the numbers):

- **Print targets** (`make_kit_target.py`): the kit's minor grid
  (254,223,219) is dropped by an office printer — the first print had no
  1 mm rules and the grid read 5× wrong. Targets now redraw the minor rules
  in `MINOR_GRID_RGB` (242,160,160); on the reprint the raw grid
  calibration came within 0.28% on a sharp photo. Prints made before
  2026-09-28 (the `ptbxl40_20260928_01.jpg` photo) have no 1 mm rules; kept
  as a real-world condition.
- **Corners**: the phone decodes photos to 3200 px and a 720 px screen shows
  4–5 photo pixels per screen pixel, so taps were 92 px off. Now
  `CornerEditor.kt`: drag handles, a ±100 px full-resolution magnifier, and
  four always-visible corner close-ups. `PageCorners`/`page.refine_corners`
  finds edges as the grid→paper step, which needs the 1 mm grid; above 3 px
  edge-fit RMS the **hand corners are used as placed** (`corners_used` in
  the CSV). Users tend to place corners inside the sheet — tell them "on the
  very tips of the grid lines". A prototype automatic corner finder (paper
  mask → grid region → line per edge, within ~1–3 mm on the 29 Sep photos)
  lives only in the 2026-09-29 session scratchpad; rebuild it if needed.
- **Binarisation**: the photo path uses a **2×2** opening (`OPEN_KERNEL`);
  at ~10 px/mm the trace is 2–3 px and 3×3 erased it (anchor −35.6% → +5.8%).
  2×2 keeps thin grid rules, which touch QRS peaks; no pixel rule separates
  them on this print (colour and threshold sweeps). An area filter in place
  of the opening was tried and rejected.
- **Pulse**: `segment.locate_rhythm_strip` skips a thin lead-in (≤5% of the
  width, `PULSE_LEADIN_FRACTION`) before the pulse, and after a lead-in
  requires the pulse's edges ≥1.2% apart (`PULSE_MIN_WIDTH_FRACTION`).
  With accurate corners the pulse starts ~8 mm in behind grid-rule ink.
- **Levelling**: `segment.level_strip` estimates the strip band's own tilt
  (imperfect corners leave the bottom of the page tilted while the top is
  level) and removes it if margin ≥1.5 and |angle| ≤2°. Drift vs truth
  1.55 mm → 0.25 mm.
- **Text removal** (`isolate_trace`): widest components first; narrow
  pieces (<3% width) are kept when they fill a hole inside the trace or
  continue it past an end at its height — both were real lost beats (a
  0.24 s gap; the tenth R wave).
- **Calibrations reported**: `duration` (10 s anchor), `grid` (raw comb —
  the control, usually wrong on photos), `grid_anchored` (the anchor picks
  the harmonic factor 1/2/5 within 10%, or refuses).
- **Honest reporting** (after a header that overstated results):
  `px_per_mm_from_corners` / `error_vs_corners_pct` are only as true as the
  corners; `truth_px_per_mm_error_pct` / `pcc_aligned` come from aligning
  the trace to the truth waveform (`Fidelity.bestAlignment`, a port of
  `run_photo.best_alignment`) and are **withheld unless the alignment spans
  ≥ 8.5 s** (`run_photo.MIN_OVERLAP_S`) — a 5×-wrong trace is 2 s long and
  the ±10% search reports a spurious local match otherwise.

**Device workflow.** Phone CSVs (`Downloads/EaCG/eacg_*.csv`) go in
`data/device_exports/`; `python ingest_device_exports.py` writes
`results/device/photo_scores.csv`, mapping old header names and
recomputing the truth alignment from each export's own waveform (the
phone's figures have matched the desktop recomputation exactly). Photos go
in `data/kit_photos/` as `ptbxl<id>_<date>_<nn>.jpg`, corners in
`python/kit_photos.json` (committed). `data/a30_pics/` held the six
`ecg00001` photos behind `results/photo/`; **they were deleted around
2026-09-29 and are not on the PC** — `results/photo/` cannot be regenerated
unless the user restores them from the phone.

**Older measurements, still valid:** grid pitch on the A30s 5.941 / 7.899 /
11.82 px/mm (+0.61 / +0.32 / +0.08%) on the three bundled sheets, 205 ms at
8.41 MP (`results/device/a30s_2026-09-18.csv`); `ecg00001` photographs
(limitation 15): repeatability overstates accuracy by about two.

## Repository layout

```
ECG-App/
├── android/app/src/main/java/com/maarijalam/eacg/
│   ├── DigitiseScreen.kt   one scrolling screen: fixtures, deskew check,
│   │                       digitise, and the photo section
│   ├── PhotoPanels.kt      the photo section UI (pick, record, corners,
│   │                       score, results, export)
│   ├── CornerEditor.kt     drag handles, magnifier, corner close-ups;
│   │                       CornerGeometry (unit-tested)
│   ├── PhotoPipeline.kt    corners -> rectify -> locate -> level -> binarise
│   │                       -> isolate -> 3 calibrations -> trace -> gate
│   │                       -> truth scoring; OPEN_KERNEL, anchor resolution
│   ├── PageCorners.kt      port of page.refine_corners
│   ├── StripLocator.kt     port of segment.locate_rhythm_strip/isolate_trace
│   ├── SheetPipeline.kt    the whole sheet on the phone: corners -> rectify ->
│   │                       grid fit -> locate -> ticks -> segment -> b3 or
│   │                       classical per row -> 12 leads (port of ecgproto.sheet)
│   ├── GridFit.kt          grid-fit geometry (port of ecgproto.gridfit)
│   ├── SheetLayout.kt / SheetCanonical.kt / SheetCorners.kt  pure ports
│   ├── SheetPanels.kt      Scan tab: whole-sheet UI; RecordsScreen.kt +
│   │                       RecordStore.kt the Records tab (private storage)
│   ├── Exporter.kt         FR10 CSV to Downloads/EaCG/
│   ├── KitTargets.kt       the printed sheets (assets/kit/targets.txt)
│   ├── Fidelity.kt         PCC/RMSE/gain, fillNan, bestAlignment (FFT)
│   ├── TraceExtractor.kt   FR5-FR7; midpointThreshold; binarise(openKernel)
│   ├── GridEstimator.kt    FR3 with detrend + resolveHarmonic
│   ├── SkewEstimator.kt    FR2; QualityGate.kt FR8; ImageWarp.kt; ImageSource.kt
│   │                       (decode to maxDim exactly, EXIF orientation)
│   └── ...                 DeskewPipeline, DigitisePipeline, TimingStats, etc.
├── android/app/src/main/assets/  reference/ (6 fixtures), kit/ (truth for the
│                                 four printed sheets, lead II at 500 Hz)
├── android/app/src/test/   75 JVM tests: port fidelity (PhotoPortFixturesTest,
│                           golden vectors in resources/photo_port_fixtures.txt
│                           from python/make_photo_port_fixtures.py), corner
│                           geometry, anchor resolution, alignment
├── python/ecgproto/
│   ├── digitise.py   FR3/FR5-FR7: detrend, _resolve_harmonic, midpoint
│   │                 threshold, binarise_trace(method, open_kernel)
│   ├── segment.py    locate_rhythm_strip, level_strip, isolate_trace; FR4:
│   │                 locate_sheet, tick_scale, segment_leads, trace_row
│   ├── sheet.py      digitise_sheet: THE reference the phone mirrors
│   ├── gridfit.py    grid-fit geometry;  timemap.py per-row time maps +
│   │                 the shift diagnosis;  corner_truth.py registration (eval only)
│   ├── page.py       refine_corners, find_sheet_corners;  pulse.py  1 mV pulse height (not wired
│   │                 into the gate yet);  quality.py FR8;  deskew.py FR2
│   └── evaluate.py, synth.py, layout.py, ptbxl.py, grid_reference.py (stale)
├── python/  score_device_sheets.py (phone 12-lead CSVs vs truth), run_geometry.py
│            (geometry vs registration), run_ticks.py (the nine ticks),
│            make_sheet_port_fixtures.py / make_grid_port_fixtures.py (golden vectors),
│            score_12lead.py (whole-sheet scorer: FR4 + every lead),
│            score_kit_photo.py (desktop scorer + simulate self-test),
│            make_kit_target.py (print targets), make_kit_assets.py (app
│            truth), ingest_device_exports.py, rerun_results.py +
│            compare_results.py (before/after), run_*.py (experiments)
├── results/  committed CSV + provenance JSON: deskew/ gate/ grid/ photo/
│             kitphoto/ (score.csv real photos, selftest.csv simulated)
│             device/ (phone runs) ml/ (training data manifests, label
│             ceiling, training histories, photo_compare_<run>.csv)
├── ml/       the learned tracer (TensorFlow, runs in WSL2): make_strips.py
│             (training bands + labels -> data/ml_strips/; --all-rows ->
│             data/ml_rows/), strips.py (canonical frame, augmentation,
│             targets, --ceiling/--selftest), train_tracer.py (model A/B,
│             training, predict_tiled), eval_photos.py (rhythm strip vs
│             classical), eval_sheets.py (all 12 leads vs classical),
│             eval_val_clean.py (scan-image check), eval_phone_path.py (THE
│             desktop replay of the phone's path), run_timemap.py (time maps),
│             make_device_fixture.py, plot_traces.py, plot_sheet.py,
│             export_tflite.py, setup_wsl.sh, wsl_run.sh
├── docs/     progress_log.md (THE engineering log since 2026-10-02),
│             ml_plan.md (the ML plan), photo_protocol.md (§8: print and
│             photograph the kit sheets), STATUS.md (2026-09-23 snapshot, stale)
├── out/      gitignored, regenerable: kit_targets/ (printable PDFs), kitphoto/
└── data/     gitignored: ptb-xl/1.0.3/, kit_photos/, device_exports/, a30_pics/,
              ml_strips/ (2,000 train + 300 val bands; copy in WSL ~/data/)
```

Data is at `data/ptb-xl/1.0.3/` and must never be committed.
It must also not live inside OneDrive (sync locks and on-demand
placeholders break bulk reads).

## Domain facts that constrain the code

These are easy to get wrong and expensive to discover late. Treat them as
invariants.

**PTB-XL contains waveforms, not images.** 21,799 clinical 12-lead
records, 10 s each, at 100 Hz and 500 Hz. Any image-based method must
render printouts from these signals (or source images elsewhere).
Currently we render our own; ECG-Image-Kit is the intended replacement.

**Diagnosis is multi-label, not multi-class.** A record can be both MI
and STTC simultaneously. Use sigmoid outputs and binary cross-entropy —
never softmax. Five superclasses: NORM, MI, STTC, CD, HYP. Report macro
AUROC (PTB-XL benchmark convention) and/or macro F-measure (PhysioNet
Challenge 2024 convention).

**Never split randomly.** ~21,800 records come from ~18,900 patients, so
a random split leaks patients across the train/test boundary and inflates
results. Use the `strat_fold` column: folds 1–8 train, 9 validation,
10 test.

**A 3x4 printout contains only 2.5 s per lead**, and the four columns are
*consecutive* time windows — so leads in different columns are not
simultaneous. Only the rhythm strip (lead II) spans the full 10 s. Total
printed content is ~33% of the recorded lead-seconds. A classifier
trained on 12 × 10 s concurrent waveforms therefore cannot consume
digitiser output from such a printout without accounting for this.

**Calibration cannot come from image DPI.** For camera-captured images
the correspondence between a physical inch and image DPI does not hold
(Reyna et al. state this explicitly). Scale must be recovered from image
content — the ECG grid is a physical ruler: 1 mm minor divisions, 5 mm
major, conventionally 25 mm/s and 10 mm/mV, though 50 mm/s and 5 mm/mV
occur.

**Metrics fail differently and must be reported together.** SNR is the
field default (the Challenge used a *shifted* SNR tolerant of small
translations). RMSE is in mV and is amplitude-sensitive, so it depends on
correct gain calibration. PCC is scale- and offset-invariant, so a
correctly-shaped but mis-calibrated trace scores near 1.0. High PCC with
high RMSE localises the fault to calibration; low PCC means the tracer
lost the waveform.

## Known limitations of the current code

Do not treat these as bugs to fix silently — they are documented
scope boundaries.

1. **`digitise()` handles a single continuous strip only.** It will trace
   straight across cell boundaries on a 3x4 printout and produce
   nonsense. Lead segmentation is the next stage and does not exist yet.

   **No longer silent, as of the FR8 gate.** `quality.traces_per_column`
   rejects a 3x4 sheet 12/12 on the observable that a column of a stacked
   printout carries four ink bands where a strip carries one (measured 4.0
   against 1.0). What that catch is worth: the accepted output had PCC
   0.273 and *every calibration statistic correct* — pitch to 0.00%,
   anisotropy 0.0000, implied 26.6 mm/s, coverage 1.000. Calibration was
   right; the digitisation was not, which is why the gate had to live
   downstream of it. `digitise()` itself is unchanged and still cannot
   detect this; the gate is a separate call.

2. **`binarise_trace()` keys on the red channel**, which works because
   `synth.render_printout` draws the grid in red. Real ECG paper varies —
   faded thermal prints, blue-grey grids, black-and-white photocopies.
   This stage is expected to need replacement for real images.

   **Measured now, and the news is mixed -- see limitation 15.** On a
   photographed pink-gridded printout the red-channel logic separates trace
   from paper and grid very well: 255-R of ~205 against ~79 and ~85, a
   120-level margin. What fails is the Otsu threshold, which is
   discontinuous in the crop -- widening a strip ROI by 44 pixel rows moves
   the ink fraction from 12% to 4% -- and which on one frame never finds the
   correct basin at all. The channel choice survives for FR5; the
   thresholding does not. **Done 2026-09-24:** Otsu is replaced by
   `digitise.midpoint_threshold` (halfway between the paper median and the
   99.9th-percentile ink level); `method="otsu"` keeps the original as the
   control.

3. **Rendered images are synthetic.** Signals are real PTB-XL; images are
   not. ECG-Image-Kit should replace `render_printout`/`render_12lead`.

4. **At 100 Hz rendered at 25 mm/s there are fewer samples than pixels**,
   so the trace is interpolated between samples. This slightly flatters
   the digitiser relative to real printouts.

5. **`synth.synthetic_ecg` is a sum of Gaussians**, not a physiological
   model. It is a test fixture only, used when PTB-XL is unavailable.

6. **The required autocorrelation threshold differs eightfold between
   image sources, and no fixed one is right for both.** ecg-image-kit
   draws the 1 mm rules about seven times lighter than the 5 mm ones, so
   their autocorrelation peaks stand at 0.11–0.19 against 0.72–0.95 for
   the majors; `synth` renders put both families at 0.78–0.83. Measured
   2026-09-17 across four channel responses and both sources — see
   `results/grid/`. `peak_fraction` was lowered from 0.20 to 0.05 as a
   result, which fixes the fixtures (39.26 → 7.90 px/mm against a true
   7.874, and `check_calibration` now accepts them) and changes nothing on
   the synthetic renders. This is a *parameter* change; the selection rule
   itself is deliberately unchanged, and the case for replacing it with a
   harmonic test rather than a height threshold is made in CHANGES.txt
   §11.5 and not acted on. **Acted on 2026-09-24:** `digitise._resolve_harmonic`
   (ported as `GridEstimator.resolveHarmonic`) tests sub-multiples of the
   spacing and walks a 2x/5x lock down; `resolve_harmonics=False` keeps the
   old rule as the control.

   Two things the earlier reading of this got wrong, both now measured.
   The green channel does not help — it is the *worst* of blue, redness,
   green and luma, because ecg-image-kit's minor-rule ink is (219,223,254)
   in BGR and green sees less of it than blue does. And `GridEstimator.kt`
   never recovered these fixtures either: run unmodified it returns NaN on
   all three, because its `0.25 * max(acf[3:60])` cut lands at 0.178–0.238
   — the same place as the Python's 0.20. The two ports agreed all along;
   only their failure modes differed.

   **The Kotlin port was divergent and is now reconciled** (commit
   `9d4acab`). All three changes landed and are visible in
   `GridEstimator.kt`: the redness response instead of green (`:100`, with
   the measurement that retired the green-channel argument), an absolute
   `PEAK_THRESHOLD = 0.05` instead of `0.25 * windowMax` (`:63`), and
   `maxLag = minOf((MAX_PX_PER_MM * 6).toInt(), acf.size - 2)` instead of
   `min(60, n-2)` (`:229`) — at `MAX_PX_PER_MM = 80.0` that is 480 lags
   against the 59.06 px the 5 mm rule needs at 300 dpi. The device results
   quoted in the status section above could not exist otherwise: run
   unmodified the old port returned NaN on all three fixtures.

   **A third image class makes the point sharper, not just different.** On
   real photographs redness is the worst of the four for 1 mm content on 3
   of 3 frames, by 6 to 15 times -- but it gives by far the most *regular*
   comb, because it sees only the 5 mm family, and the responses that do
   carry the 1 mm rules produce combs too irregular for the median-spacing
   rule to use. So on a photograph no response makes the selection rule
   work: one answers the wrong question cleanly, the others answer the right
   one noisily. Limitation 15 has the numbers. That is a stronger version of
   the conclusion this entry already draws, and it moves the harmonic test
   in CHANGES.txt §11.5 from an argument to a requirement.

7. **Deskew must run before grid estimation.** Deskew recovers a rotated
   strip to within ~1% of the unrotated PCC; rotation is not a tolerable
   input condition. Note that the *symptom* changed on 2026-09-17: at
   `peak_fraction=0.20` a 1 degree rotation made grid estimation fail
   outright in 12/12 records, so `digitise` refused. At 0.05 it succeeds
   and returns a waveform at PCC 0.49–0.77, and `check_calibration`
   accepts 10 of the 12 because the pitch really is right (the other 2 are
   caught on anisotropy). The accidental safety net is gone; a rotated
   frame now fails on PCC downstream rather than on calibration.

8. **Deskew's fine pass runs at 900 px, not full resolution, on purpose.**
   Full resolution cost 1.7–1.9 s at 8.41 MP and was superlinear in
   pixels. It bought 0.004 degrees of angle recovery where ~0.1 is what
   the waveform can detect — 1 degree of residual rotation costs 0.25 of
   PCC, 0.04 costs nothing measurable. At 900 px the stage is 74–89 ms
   *regardless of input size*, worst-case error 0.041 degrees, and
   end-to-end PCC moves by 0.00003. See `results/deskew/finescale.csv`,
   `finerobust.csv`, `finee2e.csv` and `levers.csv`, and CHANGES.txt §11.1
   to §11.3. Of the other levers tried, only narrowing the fine angular
   range helped; a coarser step, early termination and restricting the
   projection to a subregion all cost more accuracy than resolution does
   for the same time.

9. **On-device timings must separate pipeline from scaffolding.** The
   verification loop injects a rotation and renders previews so a human can
   score the estimator; a deployed capture does neither — a photograph of a
   crooked page arrives crooked at no cost. The app reports `PIPELINE`
   (to Mat + estimate skew + correcting warp) and `HARNESS` (injecting warp
   + previews) separately for this reason, and only `PIPELINE` belongs in
   the results chapter. The first observed total of ~3500 ms on the 300 dpi
   fixture is roughly double the figure that should be quoted.

10. **The desktop-to-device scaling factor is not yet known.** CHANGES.txt
    §11.3 projected 8–12× from clock and IPC ratios; the first real
    comparison (§12.4) suggests 2–4× for this kind of work, i.e. that the
    original band was threefold pessimistic. Both are inferences. Neither is
    a measurement of the deskew stage, which §12.7's benchmark now makes
    directly observable.

11. **`python/ecgproto/grid_reference.py` is stale.** It is the
    line-for-line Python twin of `GridEstimator.kt` and still carries the
    green channel and the `0.25 * windowMax` rule that limitation 6
    retired. The Kotlin has moved; the twin has not. Bring it across before
    trusting it.

12. **The FR8 gate has three measured blind spots, and they are not
    tuning problems.** `results/gate/`, 2026-09-20, 171 rows. The gate is
    anchored only on quantities known from outside the image, because the
    pipeline's internal statistics were measured ranking the wrong answer
    higher (`confidence` sat in 0.45–0.64 across clean, rotated, blurred
    and outright failed estimates; `spacing_cv` scored 0.412 on a pitch
    correct to -0.0% against 0.184 on one wrong by +399%). False rejects
    on known-good input: **0 of 36**. What it misses:

    - **A 0.5 degree rotation is flagged, never rejected, and the only
      thing that notices it is `spacing_cv`** — the internal statistic
      just ruled out. No externally anchored check saw it at all. PCC
      0.857 against 0.978 clean, so the damage is real.
    - **With the recording duration unknown, a 5x pitch error at PCC
      0.005 is only flagged.** That is the normal condition for a
      photograph of a stranger's ECG: nobody can tell the app how long
      the strip was meant to be, so the paper-speed anchor skips itself.
      Rejecting it would need the amplitude floor raised to 0.20 mV,
      which real low-amplitude leads occupy — not externally justifiable.
    - **Two checks cannot be built at all yet.** Implied gain is blocked
      because mm/mV is *assumed*, never recovered, so an implied gain is
      that assumption handed back; it needs the 1 mV calibration-pulse
      detector. Per-region duration is blocked on FR4. Both report
      `available=False` rather than passing, so the blind spots appear in
      the CSV instead of reading as clean bills of health.

    One false-flag source, also measured: the heart-rate check flags 8.3%
    of known-good input, all on record 93 — and `implied_bpm` returns the
    same 230.8 bpm on the *ground-truth* waveform, so the flag is about
    the signal's morphology, not the digitisation. It is kept only because
    it is the sole time-axis anchor that survives when duration is
    unknown. The gate is ported to Kotlin (`QualityGate.kt`, port-tested).
    A fourth blind spot, found on real photographs: the gate cannot catch a
    duration-anchor error, because the anchor assumes 10 s at 25 mm/s and
    the paper-speed check then passes by construction; it ACCEPTed traces
    with the anchor 10-14% out and one with the calibration pulse traced as
    signal.

13. **Three of the six bundled Android fixtures are 3x4 sheets.** They are
    known-good for FR3 (grid estimation) and are an instance of
    limitation 1 for everything downstream: fed to the tracer they produce
    a confident wrong waveform. `QualityGate.tracesPerColumn` is ported
    alongside the tracer for exactly this reason and rejects them. The
    other three are single strips with committed ground truth
    (`strip_ii_*dpi.png` + `strip_ii_truth.csv`, from
    `python/make_strip_fixtures.py`) and are the valid input for FR5–FR7.

14. **Half a percent of pitch error costs a third of PCC, and nothing can
    see it.** `results/gate/pitchsens.csv`, 2026-09-20, 256 rows. This is
    the most consequential number in the repository and it was found by
    accident.

    A pitch error is a time-base error: `px_per_s = px_per_mm * 25`, so
    +0.5% stretches a 10 s strip by 50 ms, five samples at 100 Hz, which
    is about the width of a QRS. The last beats are then compared against
    the wrong beats.

        pitch error    PCC    gain   gate acted
           0.000%     0.960   0.867      6%
          +0.100%     0.939   0.847      6%
          +0.250%     0.795   0.717      9%
          +0.500%     0.564   0.507      9%
          +1.000%     0.371   0.333      3%

    Three consequences, all uncomfortable.

    - **FR3's "within 1%" is too loose for FR6/FR7 to deliver.** The
      device's +0.61 / +0.32 / +0.08% map onto roughly PCC 0.5 / 0.75 /
      0.95. What a usable 10 s reconstruction needs is about **0.1%**.
    - **The gate is blind to it.** 3–9% acted across every row including
      +1%, which is the same background rate as the clean rows. It cannot
      do better: at +0.5% the implied paper speed is 25.1 mm/s against a
      nominal 25. No externally anchored check separates those.
    - **Limitation 12's 0/36 and 0.980 were measured at integer px/mm**
      (8, 12, 16), where the printed rules land on exact pixel columns and
      the estimator is right to 0.000% by construction. At the realistic
      dpi-derived scales the estimator is out by ~0.5% and PCC is 0.60 to
      0.68. The false-reject rate stands; the 0.980 does not stand as a
      figure for this pipeline on a realistically-scaled printout.

    Mechanism, and it may be an artefact of our renderer: at 7.874 px/mm
    successive 1 mm rules rasterise to integer columns, so the gaps are a
    sequence of 8s and 7s whose **median** — which is what `_pitch_1d`
    takes — is pulled toward the nearest integer. A photograph's grid is
    anti-aliased over several pixels and the bias may differ. One more
    reason the closed loop has to be broken.

    Do not retune the gate's bands to catch this. The fault is
    calibration precision; widening bands would cost false rejects
    without addressing the cause.

15. **First real photographs, and the shipped FR3 recovers nothing from
    any of them.** `results/photo/`, 2026-09-22, 45 rows across four
    experiments from `python/run_photo.py`. Source: six frames of one real
    clinical printout -- 3x4 plus a full-width lead II rhythm strip, pale
    pink grid, a 1 mV calibration pulse printed at the start of each row --
    shot on the target A30s (`SM-A307FN`). The images live in
    `data/a30_pics/` and are not committed; `python/photo_rois.json`
    records the hand-measured crops so the CSVs can be regenerated by
    anyone holding them.

    **Read the CSVs, not this entry, and not the conversation that produced
    it.** Several first-pass figures quoted while this was being explored
    did not survive a standardised ROI -- a "52% perspective spread" that
    was an estimator failure, an Otsu story told as frame-to-frame when the
    crop dominates, a channel ranking measured on one band. That is the
    whole argument for the Conventions rule about committing measurements,
    demonstrated at our own expense.

    **What the page is, settled.** It is a print of **`ecg00001.png` from
    `ecg-image-kit/sample-data/ecg-images/`** -- confirmed by digitising
    that PNG's rhythm strip and scoring it against the photographs
    (`results/photo/source.csv`). Layout, the bold lowercase `aVR`/`aVL`/
    `aVF` labels, the printed 1 mV pulse, the pale pink grid running to the
    paper edge: all match, and nothing else does.

    **There is no waveform ground truth for it and there never can be.**
    That sample-data README says "deidentified ECG images, sourced from the
    public domain", the sibling `ecg-time-series/` folder contains a README
    and nothing else, and no signal file for `ecg00001` exists anywhere in
    the vendored tree. **`ecg00001` is a filename, not a PTB-XL ecg_id**,
    which is why `--ecg-id 1` scored 0.03-0.18 (`00001_hr` is an unrelated
    56-year-old's recording at 63.9 bpm) and why three full-dataset
    matching sweeps found nothing: they were searching PTB-XL for a page
    that was never in it. **`run_photo.py --experiment fidelity` can never
    run on these photographs.** Keep it for a page printed from PTB-XL with
    a provenance sidecar.

    **What replaced it is better for the thesis question.** The clean PNG
    is not a waveform, but it is the same page acquired without a camera,
    so pushing both through the identical pipeline isolates exactly what
    the camera costs. `source.csv`, scored against that original:

        frame      PCC as-is   PCC lag only   PCC best   gain (aligned)
        113331       0.0897       0.3565       0.4072        0.574
        113332       0.1587       0.3082       0.4411        0.501
        113327       0.1319       0.4641       0.4720        0.471

    `PCC best` optimises both the pitch and the lag and is therefore the
    **ceiling** for this pipeline on this page: about 0.45, over a 9.2-9.7 s
    overlap. Amplitude comes back at roughly half (0.47-0.57), worse than
    the centroid's known bias predicts at 20 px/mm, so the photograph costs
    amplitude as well as time base.

    **And it shows that `repeat` overstates accuracy by about two.** Two
    photographs of this page agree at PCC 0.62-0.85 full length; either one
    against the clean original tops out at 0.47. The dominant errors -- the
    hand crop, the page tilt, the centroid's amplitude bias -- are common to
    both photographs and cancel between them. **Repeatability is a lower
    bound on error, not a stand-in for accuracy**, and any deployment
    number quoted from `repeat` alone is roughly twice too flattering. The
    same trap caught the duration anchor: it reproduces across three frames
    to 0.87% and is accurate to +1.9 / -8.1 / +0.35%, so its consistency
    said nothing about its correctness either.

    One concrete calibration trap the reference exposed: the printed 1 mV
    pulse is inked, and counting it as trace inflates the width the
    duration anchor divides by. On the clean original that is 12.644 px/mm
    with the pulse against 11.704 without -- an 8% calibration error from
    one printed rectangle. `reference_waveform` excludes it explicitly.

    **Overlap has to be stated with any agreement figure.** A scale search
    left unconstrained shrinks one signal until only its best-matching
    seconds remain: the same frame scores 0.85 over 4 s and 0.41 over
    9.7 s. `MIN_OVERLAP_S` forces the long answer.

    **Three matching sweeps over PTB-XL preceded this, and every one of
    them was searching for a page that was never in the dataset. Do not
    repeat them.** They are worth recording because they looked plausible
    throughout. Attempt 1 allowed a free lag, so a search could slide one
    signal 7 s past the other and score the 3 s that still overlapped; the
    null over 1500 random records then reached **0.8713**, above the
    pipeline's own frame-to-frame ceiling of 0.847, which is a
    discriminator with no power at all. Attempt 2 fixed the template to the
    central 8 s, which genuinely restores power (null max 0.5781), and
    returned `06825_hr` at 0.7540, 8.4 sd above a mean of 0.2175. Attempt 3
    scored those leaders against evidence the ranking never used -- the
    second photograph, and the leads I and III cells -- and they
    disintegrated: the two photographs of the same sheet rank different
    records first, and the three-lead winner is fourth on the rhythm strip.

    Two lessons outlive the episode. A similarity search with a free
    alignment parameter must be nulled against random data before its
    ranking is believed, and the test is whether chance can beat the
    method's own best case. And a ranking is never an identification when
    two views of the same object disagree -- 8.4 sd above a mean included.

    **FR3 fails on all six through the shipped call**, `shipped_failed` in
    `pitch.csv`, zero accepted peaks on both axes. Two independent causes:

    * **The projection smears.** 0.5 to 1.0 degrees of residual skew over
      ~4300 rows displaces a vertical rule by 45-75 px, 2-3 mm at this
      scale. Limitation 7 already says deskew runs first; this is the cost
      when it has not.
    * **A photograph's projection carries an illumination envelope** that
      `_pitch_1d`'s mean subtraction does not remove, and which flattens
      every comb peak below `DEFAULT_PEAK_FRACTION`. `run_photo.py`
      subtracts a 61 px moving average first (`DETREND_PX`) and the 5 mm
      comb then appears immediately, at cv 0.007-0.012. **That detrend is a
      property of the script, not of `digitise`, and it is the single
      smallest change that would make FR3 work at all on a photograph.**
      **Done 2026-09-24** (`b4567d4`): `digitise.detrend`, a 41-sample moving
      average as in `GridEstimator.kt`. It measurably changed deliberately
      un-deskewed rotated renders (see `results/deskew/endtoend.csv`).

    **`redness` is reliably the worst response for the 1 mm rules, and no
    alternative is reliably the best.** `channels.csv`, 1 mm / 5 mm
    spectral power, three frames:

        response   113331   113332   113327
        redness     0.108    0.169    0.097
        blue        1.471    1.503    1.024
        green       1.764    1.866    0.005
        luma        2.440    0.481    1.432

    Redness loses on 3 of 3, by 6 to 15 times. But green collapses on
    `_113327` and luma on `_113332`, so "use luma" is not the finding.
    Mechanism for redness: `r - 0.5*(g+b)` is a chroma quantity and JPEG
    4:2:0 stores chroma at half resolution and quantises it hard; luma is
    full resolution. This does **not** overturn limitation 6, which
    measured ecg-image-kit's (219,223,254) BGR minor rules where green
    genuinely loses -- a third image class, a third answer.

    **And the trade runs the other way on comb regularity**, which is what
    makes this more than a parameter choice. Redness gives by far the most
    *regular* comb (cv 0.0074-0.0117) precisely because it sees only the
    5 mm family; blue, green and luma carry the 1 mm information and
    produce combs at cv up to 0.59, which the median-peak-spacing rule
    cannot use. **No response makes the shipped selection rule work on a
    photograph**: one gives a clean answer to the wrong question, the others
    give a noisy answer to the right one. That is the case for replacing
    the height threshold with a harmonic test, argued in CHANGES.txt §11.5
    and still not acted on -- now with photographic evidence behind it.
    (Acted on 2026-09-24; see limitation 6.)

    Luma does win the one skew statistic that is internally checkable:
    `axis_disagreement_deg` between the horizontal and vertical estimates
    is 0.088 / 0.278 / 0.256 for luma against 0.949 / 0.816 / 1.000 for
    redness, and redness systematically over-reads the vertical angle
    (1.449, 1.816, 2.000 against horizontals of 0.5, 1.0, 1.0).

    **An external duration anchor beats the grid by 4 to 50 times, and it
    is the most useful thing in this entry.** The rhythm strip of a 3x4
    printout is the whole 10 s recording, so inked width / (px_per_mm * 25)
    must come to 10 s. Inverting that gives the pitch the page really has,
    and `pitch.csv` compares it against what the comb reports:

        frame     duration anchor   paper comb   trace-region comb
        113331        20.644         19.958 (-3.3%)   21.906 (+6.1%)
        113332        20.596         20.939 (+1.6%)   22.922 (+11.3%)
        113327        20.776         20.137 (-3.1%)   11.148 (-46.4%)

    The anchor agrees across three photographs to **0.87%**; the combs are
    out by 1.6-3.3% (paper) and 6-46% (over trace ink). Two consequences.
    First, against limitation 14's 0.1% requirement even the anchor is
    8.7x too coarse, so this does not solve calibration -- but it is the
    best figure this repository has ever had from paper, and it comes from
    outside the image, which is the same doctrine the Conventions section
    already applies to gates. Second, **a comb measured over trace ink is
    not trustworthy**: redness of trace (+21.6) is close to redness of grid
    (+26.2) on these photographs, and `_113327` shows the failure at -46%.

    **The page is tilted, and the pitch gradient down it is ~9.5%.**
    `vertical_gradient_pct`, paper band at the top against the rhythm strip
    at the bottom: +9.76% and +9.47% on the two frames that yield both.
    Calibrating a bottom-of-page strip from a top-of-page grid measurement
    therefore costs ~10% of time base before any estimator error. This is
    what FR1 rectification is for; there is no FR1.

    **Otsu is discontinuous in the crop, which is worse than being
    frame-dependent.** `otsu.csv`, sweeping only how much blank paper the
    rhythm ROI includes, on one frame:

        ROI rows   Otsu thr   Otsu ink%   pinned thr   pinned ink%
          611        106        13.51        126           3.47
          524        109        12.09        136           3.43
          480        125         4.47        140           3.45
          349        125         5.78        152           3.32

    The threshold jumps 109 -> 125 and the ink fraction falls 12.09% ->
    4.47% between two crops 44 rows apart, because the trace mode is small
    beside paper-plus-grid and the between-class variance maximum moves
    between basins. `_113327` jumps at a different crop; `_113332` never
    reaches the correct basin at any crop tested, holding 16.9-18.1% ink
    where the trace is 3.4%. A pinned percentile tracks smoothly across the
    whole sweep. **A threshold that depends on how much blank paper you
    included is not a property of the ink.** The trace-to-paper separation
    on 255-R is about 120 levels wide (trace ~205, paper ~79, grid ~85), so
    this is not a hard problem -- Otsu is simply not looking for it.

    **Deskew-then-binarise is unsafe, and the app escapes it only by
    accident.** `rotate_image` pads white and so does `ImageWarp.kt`; both
    are right, and a black pad would be worse. But white padding drags
    Otsu's threshold to 43, below the paper's own 79, so **96.5% of the
    deskewed frame inks, paper included.** `DigitisePipeline` survives today
    only because it does not run deskew, which its docstring says is for an
    unrelated reason. Any real capture path walks into this.

    **Repeatability, which is the only fidelity figure available without
    the record id.** `repeat.csv`, pairs of photographs of the same paper,
    threshold pinned, coverage 0.975-1.000:

        pair               PCC as-is   PCC aligned   align scale
        113327 / 113331      0.847        0.932         +0.25%
        113331 / 113332      0.749        0.801         +0.50%
        113327 / 113332      0.622        0.806         +1.00%

    `align_scale_pct` is the time-rescale that maximises agreement, and
    comparing it against the pitch ratio the grid reported gives
    `scale_error_vs_grid_pct`: the grid claimed 4.92% between `_113331` and
    `_113332` where the truth is 0.50%, so the estimator is wrong by 4.4
    percentage points. Half the remaining loss is tracing noise and half is
    calibration.

    **All three 24.8 MP frames are usable** (`_113327`, `_113331`,
    `_113332`), and `_113327` was wrongly written off on first reading. The
    three 12 MP frames are not, and not for resolution: ~14.7 px/mm clears
    the 3 px/mm floor comfortably. They fail on blur and oblique lighting,
    `_113130` marginally and `_113036`/`_113053` with no recoverable comb.

    One thing these images unblock: the calibration pulse is printed and
    clearly visible, which is the anchor limitation 12 lists as missing for
    the `implied_gain` check.

    Do not "fix" FR3 by widening `max_pitch_px`. It is the guard that turned
    a fivefold calibration error into an honest refusal, and the faults it
    exposed are an undetrended projection, a response that cannot see the
    1 mm rules, and a selection rule that cannot use them when it can --
    none of which is a range problem.

## Technical decisions already made

- **Kotlin, not C++/NDK.** LiteRT inference and OpenCV already run
  natively under the hood; the JNI boundary costs a crossing, not the
  compute. CameraX, ML Kit and UI are Kotlin-only anyway.
- **LiteRT (formerly TensorFlow Lite)** as the mobile runtime.
- ~~ML Kit Document Scanner for capture-time rectification.~~ **Dropped
  2026-09-25, pending the supervisor's agreement:** it returns only a
  processed image (no corners or transform), so its accuracy cannot be
  measured against the truth; it runs as its own Play-services screen, so it
  cannot be timed as a pipeline stage; and it cannot run on the desktop
  scorer. FR1 is now automatic corners (`page.find_sheet_corners`) plus the
  grid fit (`gridfit`), which makes corner error irrelevant; hand corners
  remain possible in the app's CornerEditor.
- **Emulator for correctness only.** It runs on host x86 CPU with active
  cooling, so latency, memory and thermal numbers from it are invalid for
  a thesis about mobile constraints. All benchmarking must use a physical
  device (one borrowed older Android phone is currently the only target).
  Note the APK is now filtered to `arm64-v8a` alone, so it will not install
  on an x86 emulator image at all.
- **OpenCV does resize and warp; Kotlin does the response and the search.**
  Not a stylistic split. Resize and warp are full-resolution and per-pixel,
  where a SIMD native routine wins. The angle search bins pixels by
  `y - x*tan(a)` and has no OpenCV equivalent — expressing it in primitives
  would mean one warp per angle, which is slower *and* wrong, since every
  warp low-passes the grid detail the objective measures.

## Building the Android app

`gradlew.bat` is broken in this project's Gradle 9.5 wrapper — it sets an
empty CLASSPATH. Build from `android/` with:

```
java -Dorg.gradle.appname=gradlew -jar gradle/wrapper/gradle-wrapper.jar assembleDebug
```

The APK lands at `android/app/build/outputs/apk/debug/app-debug.apk`.
The APK is filtered to `arm64-v8a` alone, so it will not install on an x86
emulator image. Nothing in the Android tree can be verified beyond
compilation and the JVM unit tests without the physical device; say so
explicitly rather than asserting behaviour.

The unit tests — the port-fidelity suite included — need no device:

```
java -Dorg.gradle.appname=gradlew -jar gradle/wrapper/gradle-wrapper.jar testDebugUnitTest
```

112 JVM tests as of 2026-10-08 (UI redesign, progress_log §16-24). After changing `segment.py`, `page.py` or the
digitise functions the photo path uses, regenerate the golden vectors with
`python make_photo_port_fixtures.py` (from `python/`) and port the change to
Kotlin in the same commit; each new rule gets a fixture case checked to
FAIL under the old rule.

## Running the Python tests

From `python/`. There is no pytest dependency and no reason to add one.

```
python -m unittest discover -s tests     # 89 tests
python tests/mutate_timemap.py           # 21 of 21 mutations caught
python tests/mutate_quality.py           # 14 of 15 mutations caught
```

`mutate_quality.py` edits `ecgproto/quality.py` in place and restores it on
every exit path. One mutation is a documented equivalent mutant and is
expected to survive; the script says which.

## Conventions

- Python 3, NumPy/OpenCV/SciPy/pandas/wfdb in `python/`, which stays free of
  any deep-learning dependency. TensorFlow/Keras (2.21) lives only in `ml/`
  and runs in WSL2.
- British spelling in prose and identifiers (`digitise`, `binarise`,
  `vectorise`) to match the thesis document.
- Docstrings explain *why*, not just what — several encode the domain
  invariants above and should not be stripped.
- The classical pipeline is a deliberate **control condition**. Every ML
  component added later must be justified against its numbers. Do not
  delete or "improve away" the classical path.
- **Measurements are committed.** Every experiment writes a CSV plus a
  JSON sidecar carrying the git commit, the parameters and the library
  versions, under `results/`. A number that exists only in a terminal
  scrollback cannot be cited or rechecked later. `out/` is the opposite:
  regenerable artefacts, gitignored.
- **A gate or guard is anchored outside the image, never on the
  pipeline's own self-consistency.** Measured twice now: every internal
  statistic ranked a +399%-wrong grid estimate above a correct one. If an
  anchor does not exist, the check reports itself unavailable rather than
  borrowing one — see limitation 12.
- **Vary one thing at a time, and check that you did.** The first
  interpolation comparison in `run_deskew.py` varied the warp used to
  *simulate* a rotation together with the warp used to *correct* it, and
  produced a confident, wrong conclusion. Simulation parameters and
  parameters under test need separate flags.

- **Measure against the truth, and before trusting a fix, measure it.**
  Twice in this project a plausible fix made things worse and was caught
  only by an end-to-end score against the PTB-XL waveform (an area filter
  replacing the opening; a width filter on the row profile). Replay the
  phone's own path on the desktop (3200 px decode, `open_kernel=2`, the
  run's hand corners from its CSV) before porting anything to Kotlin.
- **Any agreement figure states its span.** `pcc_aligned` or a scale error
  from an alignment covering < 8.5 s is not a measurement; report it as
  not measurable.
- **Commit messages carry no Claude co-author trailer** (the user's
  instruction, 2026-09-25). Never cite a commit hash in a message without
  checking it exists.
- Scores run from uncommitted code record `git_dirty: true` in their
  sidecar: commit the code (and `python/kit_photos.json`) first, then
  regenerate the results.
