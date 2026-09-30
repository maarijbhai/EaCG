# Project status audit — 2026-09-23

> **Superseded.** This is a snapshot from 2026-09-23 and is not maintained.
> The current state, the photo pipeline and the next steps are in
> `CLAUDE.md` ("Current state — read this first"); the ML plan is
> `docs/ml_plan.md`.

Branch `prototype-and-dataset-repair`, HEAD `f064ebe`. Read-only audit; no code changed.

> **Correction, 2026-09-24.** §4 originally said the kit sheets carry no
> printed lead labels, reading `printed_text: False` in the sidecar. That was
> wrong: the flag suppresses only the kit's header band. Every sheet prints
> the lead name under the start of each strip and `25mm/s  10mm/mV` below
> the rhythm strip, and a crop that includes them digitises the text as
> trace -- found when the pipeline scored PCC 0.30 on a clean sheet. See
> `segment.isolate_trace`. The rest of this document is the 2026-09-23
> snapshot and is not updated as the code moves.

**VERIFIED** = run or measured in this audit session (command given).
**READ** = inferred from code, docs or commit contents, not executed.
FR numbers are the thesis report's flat FR1–FR11 (CLAUDE.md), not the nested
FR-x.y of `docs/functional_requirements.tex`.

---

## 1. Stage status

| FR | Python | Kotlin | Renders? | Real photos? | Device? | Key result + evidence |
|---|---|---|---|---|---|---|
| FR1 acquisition / rectification | Manual 4-corner warp only (`python/score_photo.py` `rectify`); hand ROIs in `python/photo_rois.json` stand in for it | Gallery picker (`DigitiseScreen.kt:231`, `PickVisualMedia`) + bundled assets; downsampled to 2560 px (`ImageSource.kt:47-64`). No camera, no rectification | score_photo self-test on *simulated* photos (CHANGES.txt §10.5) — READ | No. score_photo never run on a real print — READ (no output in `results/`) | Picker exists; untested on a photo — READ | No CSV. |
| FR2 deskew | `ecgproto/deskew.py` — VERIFIED runs: 0.500° on `20260921_113331`, 590 ms | `SkewEstimator.kt`, port tests 9/9 pass at 2026-09-20T16:12Z (`build/test-results`) — READ | Yes | Yes (angle only; no ground-truth angle) | Wired to UI with injected-rotation loop; only the ~3500 ms total was observed on device (CHANGES.txt §12.7: "NOT VERIFIED: every number above except the 3500 ms total") — READ | `results/deskew/finescale.csv`, `finerobust.csv` (0.041° worst, 74–89 ms desktop, per CLAUDE.md lim. 8) — READ; `results/photo/channels.csv` (skew per response) |
| FR3 grid | `digitise.estimate_grid_pitch` | `GridEstimator.kt` (one axis only; has a 41-px moving-average detrend, `GridEstimator.kt:66,151` — the Python does not) | Yes | Yes: **fails on all six** (`shipped_failed`, `results/photo/pitch.csv`). VERIFIED this session: NaN on `113331` both deskewed and not | Yes: 5.941/7.899/11.82 px/mm on the 3 kit sheets (+0.61/+0.32/+0.08 %), 205 ms at 8.41 MP — **recorded only in CHANGES.txt §12.4 text, no CSV** — READ | `results/grid/robust.csv`, `threshold.csv`, `results/photo/pitch.csv` |
| FR4 lead segmentation | **Absent** (a rhythm-band finder exists only in a scratchpad, see §3) | Absent | — | — | — | — |
| FR5 trace segmentation | `binarise_trace` (red channel + Otsu) | `TraceExtractor.binarise` (Otsu, `TraceExtractor.kt:104-130`) | Yes | Yes, via hand ROI: Otsu discontinuous in the crop (`results/photo/otsu.csv`) | **Never run on handset** (CHANGES.txt §14.6) — READ | `results/photo/otsu.csv` |
| FR6 vectorisation | `trace_positions` (column centroid) | `TraceExtractor.digitise` | Yes | Yes, hand ROI + external pitch: gain 0.47–0.57 vs clean original (`results/photo/source.csv`) | Never run on handset — READ | `results/photo/source.csv`, `results/gate/pitchsens.csv` |
| FR7 calibration | `pixels_to_mv`, `check_calibration` | in `TraceExtractor.digitise` | Yes | Yes: grid comb 1.6–46 % out; duration anchor consistent to 0.87 % (`pitch.csv`), accurate to +1.9/−8.1/+0.35 % (`source.csv` `anchor_error_pct`) — READ | Never run on handset — READ | `results/gate/pitchsens.csv` (0.5 % pitch → PCC 0.564) |
| FR8 quality gate | `ecgproto/quality.py`; 31/31 tests pass — VERIFIED (`python -B -m unittest discover -s tests`) | `QualityGate.kt`, 15/15 port tests pass (2026-09-20 run) — READ | Yes: 0/36 false rejects | VERIFIED this session: gate returns FLAG (paper-speed skipped) on hand-ROI rhythm strip of `113331` | Never run on handset — READ | `results/gate/knowngood.csv`, `inject.csv`, `pitchsens.csv` |
| FR9 on-device execution | n/a | FR2, FR3, FR5–FR8 compiled into APK (`app-debug.apk`, built 2026-09-20 18:12 local) | — | — | FR2/FR3 ran on A30s; FR5–FR8 not — READ | CHANGES.txt §12.4, §14.6 |
| FR10 export | Absent | Absent ("cannot leave the device", CHANGES.txt §14.6) | — | — | — | — |
| FR11 latency reporting | Per-stage ms in deskew experiments | `TimingStats.kt`, `DigitisePipeline.Timings` (pipeline vs harness split) | Desktop yes | — | Grid 205 ms, deskew ~3500 ms total (harness-inclusive); no CSV — READ | `results/deskew/cost.csv`, `finescale.csv` |

Kotlin unit tests: 48 `@Test` methods (1+5+3+15+9+15), all passing in the last recorded run
(`android/app/build/test-results/testDebugUnitTest/*.xml`, timestamp 2026-09-20T16:12Z) — READ.
Not re-run this session. The last commit touching `android/app/src` (`8503434`) is dated
2026-09-22 21:00Z, after that run, so whether the committed code passes is **not verified**.

---

## 2. End-to-end capability

### Python — VERIFIED this session on `data/a30_pics/20260921_113331.jpg`

| Step | Automatic? | Result |
|---|---|---|
| Load + EXIF orientation (`run_photo.load_oriented`) | yes | 4312 × 5760 |
| FR2 `deskew.deskew(img, response_fn=dig.redness)` | yes | 0.500°, 590 ms |
| FR3 `estimate_grid_pitch(deskewed)` | yes | **NaN / NaN, 0 peaks — stops here** |
| `digitise(deskewed)` | — | `ValueError: grid pitch estimation failed; cannot calibrate` |

The automatic path stops at FR3. It goes through only with **both** manual inputs:
- a hand crop of the rhythm strip (`photo_rois.json` → 611 × 5161 px), and
- a supplied pitch (`GridEstimate(20.644, ...)`, the duration-anchor value from `pitch.csv`).

With both, `digitise(strict=False)` returned 5000 samples = 10.00 s, and the gate returned
FLAG. The 10.00 s is circular: 20.644 px/mm was itself derived by assuming a 10 s strip.
There is no automatic page detection, lead segmentation, or rhythm-strip location in
committed code.

### Android — READ (`DigitiseScreen.kt`, `DigitisePipeline.kt`)

Start to finish today:
- **Load.** Pick one of 6 bundled fixtures, or pick an image from the gallery. Gallery images
  are downsampled to ≤ 2560 px.
- **Deskew panel.** Rotation slider → inject → estimate → correct, plus a benchmark button.
  Wired to the UI.
- **Digitise button.** Runs `DigitisePipeline.run` on the **whole loaded image**:
  FR3 (one axis, assumed isotropic) → binarise → trace → calibrate → gate → fidelity.
  - Fidelity is scored only when a truth waveform exists. That is only for the bundled
    `strip_ii_*` fixtures (`TruthWaveform.kt`, `strip_ii_truth.csv`), so a gallery photo
    gets no score.
  - Deskew is **not** part of this path (`DigitisePipeline.kt:20-23`).
- **Results.** Strip row, verdict, waveform plot and timings are shown on screen. Nothing is
  exported.

Exists as code but not in the UI flow: nothing extra was found. Every Kotlin stage class is
reachable from `DigitiseScreen`. Missing entirely: camera, rectification, crop/ROI selection,
FR4, export, and importing a truth file for a user-supplied image.

---

## 3. Uncommitted work

| Item | Where | What |
|---|---|---|
| **Photo proof of concept** (2026-09-21) | scratchpad `C:\Users\maari\AppData\Local\Temp\claude\C--dev-ECG-App\7912a201-8b8c-4192-a572-051cbbd58fb0\scratchpad\poc_pipeline.py` + `poc_*.png/npy` | "photo → deskew → find rhythm strip → digitise". Finds the 4 trace bands by row ink density and takes the bottom one as the rhythm strip — an **automatic rhythm-strip locator not present in any committed file**. Also a manual rectification (`rectified.png`, `rect_M.npy`). `poc_runs.pkl` referenced in the listing is gone. Outputs not in `results/`. |
| PTB-XL identification sweeps | same scratchpad: `identify*.py`, `identify_all.txt`, `identify_fast.txt`, `ident_best.txt` (`00511_hr`), `confirm*.py` | The three matching sweeps. Superseded: the page is `ecg00001.png` (CLAUDE.md lim. 15). Not in `results/`, by decision. |
| Session patch scripts | same scratchpad `patch*.py`, `patch_claude_md*.py` | Edit scripts; their effects are committed (`3b8b412`, `f064ebe`) — READ, not diffed. |
| Deskew before/after CSVs | scratchpad `5031bd35-…\scratchpad\before\*.csv`, `beforeafter.py` | Pre-change deskew results. The "after" is committed under `results/deskew/`; the "before" is not. |
| Gate/tracer probes | scratchpad `56de9ee7-…\scratchpad\probe_*.py`, `changes_13/14.txt` | Probe scripts; the text went into CHANGES.txt §13–14. |
| Demo overlays | scratchpad `e301ff93-…\scratchpad\ovl_poc_*.png` | Figures for `demo_kit_pipeline.py`. |
| **Device measurements** | CHANGES.txt §12.4, §12.7 text only | Grid pitch/latency on A30s and the ~3500 ms deskew total. **No CSV under `results/`**, contrary to the Conventions rule. |
| Kit targets | `out/kit_targets/` (gitignored) | Generated at commit `8039ea9+dirty`, *before* `make_kit_target.py` was committed (`29d9435`). Whether the committed script reproduces them byte-for-byte: **cannot tell**. |
| Single-strip print targets | `out/print_targets/ptbxl9_II_500hz_300dpi`, `ptbxl180_V2_500hz_300dpi` (gitignored) | From `make_print_target.py`; regenerable. |
| `prompt_b5_and_photo.md` | repo root, untracked | Task prompt. Part 1 (B5) is done (`9acf9cb`; `synth.OVERFLOW_MODES`, `synth.py:94`). |
| Photographs | `data/a30_pics/` (gitignored, 6 jpg) | By design; ROIs committed. |

---

## 4. Print-and-score readiness

### Contents of `out/kit_targets/` — VERIFIED (read the JSON sidecars this session)

| Sheet | PTB-XL ecg_id | patient_id | strat_fold | Clean/augmented | Layout | Truth sidecars |
|---|---|---|---|---|---|---|
| `ptbxl9_kit_500hz_300dpi` | 9 | 18792 | 10 | clean (augment/wrinkles/noise/rotate/crop all off) | 3×4 + full-length II rhythm, 13 leads, 1 mV pulse on, lead names and a `25mm/s  10mm/mV` legend **printed** (see correction below), A4 landscape, 277.03 × 189.99 mm, 300 dpi | `_truth.npy` (5000 × 12), `_kit.json` (plotted_pixels), `.json` provenance — all present |
| `ptbxl38_kit_500hz_300dpi` | 38 | 17076 | 10 | clean | same | all present |
| `ptbxl40_kit_500hz_300dpi` | 40 | 19501 | 10 | clean | same | all present |
| `ptbxl57_kit_500hz_300dpi` | 57 | 16063 | 10 | clean | same | all present |

Each PDF has 2 pages (sheet + ruler verification page) — VERIFIED (page count).
The kit sidecar has **no `lead_bounding_box` field**. Leads carry only `lead_name`,
`start_sample`, `end_sample` and `plotted_pixels` — VERIFIED.

### Steps

| # | Step | Status |
|---|---|---|
| 1 | Generate sheets with truth | **exists and works** (4 sheets above) |
| 2 | Print at 100 % scale, colour, A4 landscape | **exists, untested.** Procedure in `docs/photo_protocol.md` §1, but it is written for `make_print_target.py` single strips, not kit sheets |
| 3 | Verify scale with page-2 ruler | **exists, untested** (page present; never measured) |
| 4 | Photograph on the A30s | **exists, untested** for these sheets (`photo_protocol.md` §3–4) |
| 5a | Python: locate the rhythm strip | **missing** in committed code. Options are hand ROI via `photo_rois.json`, or the scratchpad PoC |
| 5b | Python: calibrate | **missing automatically** (FR3 fails on photos, §2). The external comb + duration anchor exist only inside `run_photo.py` |
| 5c | Python: score vs PTB-XL — `run_photo.py --experiment fidelity --ecg-id N` | **exists, untested** (never run on any photo). Constraints, READ: photos must be in `data/a30_pics` (`run_photo.py:121`); ROIs must be in `photo_rois.json`; one `--ecg-id` applies to **all** frames per run, so sheets from different records cannot share a run |
| 5d | Python: score via `score_photo.py` | **broken for kit sheets.** VERIFIED: `KeyError: 'geometry'` on `ptbxl9_kit_…json`. Works only with `make_print_target` manifests (which have `geometry`) |
| 6 | Get the photo onto the phone app | **exists, untested** (gallery picker) |
| 7 | Phone: crop to rhythm strip | **missing** (the app digitises the whole image; a 3×4 sheet is rejected by `tracesPerColumn`, CLAUDE.md lim. 13) |
| 8 | Phone: calibrate on a photo | **exists, untested** (`GridEstimator` has a detrend the Python lacks; never run on a photograph) |
| 9 | Phone: score against PTB-XL truth | **missing** (truth is loaded only for bundled `strip_ii_*` assets) |
| 10 | Phone: get the result off the device | **missing** (FR10) |

---

## 5. Open problems (CLAUDE.md limitations)

| # | One line | Blocks §4? |
|---|---|---|
| 1 | `digitise()` handles one continuous strip only; no FR4 | **Yes** — steps 5a and 7 |
| 2 | Red-channel binarisation; Otsu discontinuous in the crop | **Partly** — Kotlin uses Otsu (step 7+). `run_photo` uses a pinned percentile |
| 3 | Images synthetic | No — §4 is the remedy |
| 4 | 100 Hz fewer samples than pixels flatters digitiser | No (500 Hz truth) |
| 5 | `synthetic_ecg` is a Gaussian sum | No |
| 6 | Grid threshold source-dependent; selection rule can't use 1 mm rules | **Yes** — automatic FR3 (5b, 8) |
| 7 | Deskew must precede grid | Affects 5b, 8 |
| 8 | Deskew fine pass at 900 px by design | No |
| 9 | Device timings must separate pipeline from harness | No (affects reporting only) |
| 10 | Desktop→device scaling unknown | No |
| 11 | `grid_reference.py` stale | No |
| 12 | Gate blind spots (0.5° rotation, unknown duration, no implied-gain/per-region checks) | No — but the gate cannot catch calibration errors |
| 13 | 3 of 6 Android fixtures are 3×4 sheets | Indirectly — same mechanism rejects a photographed kit sheet (step 7) |
| 14 | 0.5 % pitch error costs ~⅓ PCC; gate blind to it | No — but it dominates the result |
| 15 | Real photos: FR3 fails, Otsu unstable, deskew-then-binarise unsafe, ~9.5 % pitch gradient down the page | **Yes** — 5b, 7, 8 |

---

## 6. Inconsistencies

### Kit minor grid colour: (219,223,254) vs (254,223,219)

**Both are correct.** They are the same colour written in opposite channel order.
- The kit records it as RGB `grid_line_color_minor = [253.98, 223.0, 218.99]`, in `ptbxl9_kit.json` (matplotlib RGB × 255, `ecg_plot.py:503`).
- Read with `cv2.imread` (BGR), the most common pixel in `gt_norm_150dpi.png` and in `ptbxl9_kit_500hz_300dpi.png` is `(219, 223, 254)` — VERIFIED.

Each mention, by order:
- `CLAUDE.md:261`, `CLAUDE.md:547`, `CHANGES.txt:1326` and `GridEstimator.kt:106` label it BGR — correct.
- `python/make_kit_target.py:118` and `make_port_fixtures.py:336` use (254,223,219) as RGB — correct. The fixture's comment says "(r, g, b)", and the Kotlin receives ARGB.
- `docs/ml_plan.md:454` writes (254,223,219) **unlabelled**.

What it changes: nothing numerically, if the order is read right. Read the wrong way round it
would be a pale *blue* rule:
- redness `r − ½(g+b)` would be 0 instead of +33;
- the "green sees 32/255, blue 36/255" argument in limitation 6 would invert.

One substantive problem in the same line: `ml_plan.md:454` calls the kit's minor grid
"unrecoverable from a photograph — known, measured". **No photograph of a kit-rendered sheet
exists.** The photographed page is `ecg00001.png`, a scanned image with its own pale-pink grid
(CLAUDE.md lim. 15). The claim is unsupported by any CSV.

### Other contradictions found

1. **CLAUDE.md lim. 12 vs the code.** Lim. 12 ends "The gate is Python only; nothing is ported to Kotlin." But `QualityGate.kt` exists with 15 port tests, and lim. 13 and the layout section both describe it. Lim. 12 is stale.
2. **The detrend.** CLAUDE.md lim. 15 says the detrend "is a property of the script, not of `digitise`" — true for Python (`_pitch_1d` only subtracts the mean, `digitise.py:356`). But `GridEstimator.kt` already applies a 41-px moving-average detrend (`:66`, `:151`). So the Kotlin and Python FR3 are not equivalent on photographs. Neither CLAUDE.md nor lim. 6's "reconciled" wording says so. (CHANGES.txt:1328 does mention "the moving-average detrend the Kotlin applies".)
3. **Result counts.** CLAUDE.md says "18 CSV/JSON pairs" under `results/`; there are 23 (deskew 11, gate 3, grid 4, photo 5) — VERIFIED `ls`. It also says `results/photo/` is "45 rows over four experiments"; it is 49 rows over five CSVs (channels 12, otsu 21, pitch 9, repeat 3, source 4) — VERIFIED `wc -l`.
4. **Test count.** CLAUDE.md says "39 host-side unit tests pass". The last run shows 48 (39 plus the 9 `SkewEstimatorPortTest`) — READ from test XML.
5. **`ml_plan.md` §3 Phase 2** proposes "IoU against `lead_bounding_box`". The kit sidecars carry no such field (§4).
6. **`results/grid/kotlin.csv`** holds the desktop Python twin's output (`kotlin_green_movavg_0.25` columns, NaN on kit fixtures), not device measurements. The device grid numbers exist only in CHANGES.txt §12.4.
7. **`docs/photo_protocol.md` §0** generates targets with `make_print_target.py` and scores with `score_photo.py`. Neither the procedure nor the scorer covers `make_kit_target.py` sheets (score_photo `KeyError`, §4 step 5d).
8. **Status paragraph dates.** CLAUDE.md says the tracer was ported "2026-09-20", but commit `8503434` is dated 2026-09-22. The test XML is 2026-09-20, so the work predates the commit. Not a contradiction in substance.
