# The machine learning element: plan, model choice, and decision rules

Status: rewritten 2026-09-29, replacing the 2026-09-22 plan (in git
history). Written against the real-photograph measurements in
`results/device/photo_scores.csv` and `results/kitphoto/`. Every number
marked *measured* has a CSV or a commit behind it; every number marked
*expected* is a prediction with its reasoning, to be overwritten by a
measurement.

---

## Progress, 2026-10-01 (read this first)

**Two days into the two-week plan (deadline ~2026-10-14), and ahead of it on
the model.** Data, training and a first real-photo result are done; nothing
has touched the phone yet.

| | Status | Where |
|---|---|---|
| Training data: 2,000 fold 1–8 + 300 fold 9 strips, exact labels | **done** | `ml/make_strips.py`, `results/ml/strips_*` |
| Augmentation with labels carried exactly (100% on ink) | **done**, camera stage added after b0 | `ml/strips.py`, `results/ml/strips_selftest.*` |
| Label ceiling; frame resized to 48 mm, head output ×2 | **done** (§2.2) | `results/ml/label_ceiling.*` |
| WSL2 + TF 2.21 + RTX 3050 | **done** | `ml/setup_wsl.sh`, `ml/wsl_run.sh` |
| Model B trained (b0, then b1 with camera augmentation) | **done** | `results/ml/train_b0.*`, `train_b1.*` |
| Real-photo comparison, matched corners and calibration | **done on PTB-XL 40 only** | `ml/eval_photos.py`, `results/ml/photo_compare_*` |
| Photos of PTB-XL 9, 38, 57 | **not yet** (user, 2026-10-02) | `data/kit_photos/` |
| Model A (ablation) | not started | `train_tracer.py --head mask` |
| INT8 LiteRT, Kotlin port, app switch | not started | §5 |
| Latency and memory on the A30s | not started; memory not instrumented | rules 2, 3 |

**Measured on six real A30s photographs of PTB-XL 40** (matched corners,
warp and calibration; aligned PCC only where the alignment spans ≥ 8.5 s):

| photo | classical PCCa / R | b0 PCCa / R | **b1 PCCa / R** |
|---|---|---|---|
| 0928_01 | 0.851 / 0.728 | 0.646 / 0.950 | 0.832 / 0.948 |
| 0929_01 | 0.873 / 0.595 | 0.638 / 0.528 | 0.922 / 0.935 |
| 0929_02 | — (7.3 s) | — | 0.840 / 0.913 |
| 0929_03 | — (< 8.5 s) | — | 0.921 / 0.913 |
| 0929_04 | — (8.3 s) | — | 0.932 / 0.918 |
| 0929_05 | 0.890 / 0.675 | 0.896 / 0.892 | 0.915 / 0.890 |

Against decision rule 1 (§4), on this one record:
- **R-peak ratio ≥ 0.90: met on 5 of 6** (0.89 on the sixth); classical
  0.59–0.73.
- **No loss of coverage: met.** b1 traces 94–100% of columns; classical
  53–97%, too little on three photos to be scored at all.
- **Aligned PCC ≥ classical + 0.08: not met.** −0.02, +0.05, +0.03 on the
  three photos where both can be scored.

What b0 → b1 showed: the render-to-photo gap was real (b0 dropped whole beats
on real photos at a simulated PCC of 0.997), and closing it took a camera
model in the augmentation (exposure, gamma, desaturation, blur, ISP
sharpening), not a different network.

**Two caveats limit what can be claimed.** It is one record. And the
PTB-XL 40 photographs' appearance informed the camera augmentation, so they
are no longer an untouched test set. **The clean test is PTB-XL 9, 38 and 57,
photographed after b1 was frozen**, scored once, with nothing changed.

---

## 0. What changed since the last plan, and why the order changes

The 2026-09-22 plan argued that **calibration dominates** and that a learned
tracer should therefore come last. That was right on the evidence then: on
the `ecg00001` photographs the recovered scale was 1–3% out, and a 0.5%
pitch error alone costs about a third of PCC (`results/gate/pitchsens.csv`).

The evidence has moved. On a real A30s photograph of a printed PTB-XL
record (PTB-XL 40, phone run 5, measured):

| | value | source |
|---|---|---|
| time-scale error against the truth waveform | **−0.5%** | duration anchor, pulse cut |
| start lag | **0 samples** | pulse detected and removed |
| aligned PCC, phone, as-placed corners | **0.74** over 9.8 s | `photo_scores.csv` |
| aligned PCC, desktop replay, accurate corners | **0.83–0.86** over 9.7 s | `kitphoto/score.csv` |
| R-peak height, digitised / true | **0.72** | decomposition of run 5 |
| T-wave height, digitised / true | **0.97** | same |

Calibration is no longer the binding constraint on this sheet. What is left
is almost entirely **trace extraction**:

1. **R peaks under-read** (0.72 of true height while T waves read 0.97).
   The classical tracer takes the ink centroid of each column; on a steep
   QRS stroke that is the stroke's midpoint, not its tip.
2. **Grid rules traced as signal.** At the phone's working resolution the
   trace is 2–3 px thick. A 3×3 opening erases it; a 2×2 opening keeps it
   but also keeps the thin grid rules, which join the trace where QRS peaks
   touch them. Colour and threshold sweeps found **no pixel-level rule** that
   separates the two on this print. This is a segmentation problem, and it
   is exactly what a learned model is for.
3. **Broken strokes.** Thin strokes break at the phone's resolution, and
   short pieces were dropped as text (patched classically in `db2520a`, but
   the break itself remains).

So the new order is: **learned trace extraction first**. Corners (FR1) and
lead segmentation (FR4) follow, each only if its classical version measurably
fails.

Two corrections to the old plan, stated because they change the model:

- **Segmenting the ink does not fix the R-peak bias.** The old plan proposed a
  U-Net ink mask followed by a column-wise expectation, claiming the
  expectation over a thin mask has no midpoint bias. It does: a correctly
  segmented steep upstroke *is* a tall vertical run of ink in that column,
  and its expectation is its midpoint. The model has to be trained to report
  **where the waveform is at each column**, not where the ink is (§3).
- **ML Kit Document Scanner is dropped** as the FR1 route (pending the
  supervisor's agreement). It returns only a processed image, with no corners
  or transform, so its accuracy cannot be measured against the truth. It runs
  as its own screen inside Google Play services, so it cannot be timed as a
  pipeline stage. And it cannot run on the desktop scorer.

---

## 1. Where things stand

| Stage | Classical status | Measured |
|---|---|---|
| FR1 page corners | hand-placed with magnifier; edge refinement fails without the 1 mm grid | 2–9 mm placement error, runs 4–5 |
| FR2 deskew | works; strip now levelled locally | residual drift 0.90 → 0.29 mm (replay) |
| FR3 grid | 5× lock on prints without 1 mm rules; anchored variant −4% | runs 3–5 |
| FR4 lead segmentation | rhythm strip only | — |
| FR5/FR6 trace extraction | **the bottleneck** | R peaks 0.72, grid contamination |
| FR7 calibration | duration anchor −0.5% with pulse cut | run 5 |
| FR8 gate | works; blind to anchor error by construction | — |
| FR11 latency | pipeline 1.5 s on the A30s | run 5 |
| Memory | **not instrumented** | — |

---

## 2. The model: recommendation

**A small fully convolutional encoder–decoder (a U-Net built from
depthwise-separable blocks, about 0.3–1 M parameters) with a column-wise
row-distribution head.** For every image column it outputs a probability
distribution over rows. The waveform value at that column is the
distribution's expectation (a soft-argmax), plus a per-column "trace present"
probability for gaps and strip ends.

It runs on one rhythm-strip crop, resampled to a canonical scale, and plugs
into the existing pipeline in place of `binarise_trace` + `trace_positions`.
Calibration, the gate and export stay unchanged.

### 2.1 Why this formulation

| Option | What it predicts | Fixes R-peak bias? | Fixes grid contamination? | Verdict |
|---|---|---|---|---|
| Classical (control) | threshold + column centroid | no | no | the baseline, kept |
| **A. U-Net ink mask + centroid** | which pixels are trace | **no** (a correct mask still holds the whole upstroke) | yes | the ablation: separates the two effects |
| **B. U-Net + column-wise row head** | the waveform's row at each column | **yes**: trained on the sample positions, it learns the tip | yes | **recommended** |
| C. Direct 1-D waveform regression from the image | a vector of mV values | yes | yes | not recommended: opaque failures, and it absorbs calibration into the network, which breaks the existing gate and anchor |
| D. Large segmentation nets (DeepLab-class, nnU-Net at full size) | ink mask | no | yes | too heavy for the A30s budget, and still option A's bias |

Option B keeps what matters for the thesis:
- **The output is interpretable.** A row per column, turned into mV by the
  existing calibration. A failure shows up as a visible wrong row, not an
  inexplicable voltage.
- **Calibration stays independent of the model**, so the duration anchor, the
  gate and every committed calibration measurement remain valid.
- **It has exact supervision for free.** `plotted_pixels` gives the pixel
  position of every one of the 5,000 samples, which is precisely a row per
  column.
- **It is a known shape of problem.** Predicting one position per column for
  a thin curve matches the row-anchor formulation used for fast lane-line
  detection. U-Net-style segmentation is also the common approach in
  ECG-image digitisation work (for example, PhysioNet Challenge 2024
  entries). Check those citations before the write-up.

**Run A and B both.** A costs little extra, as it is B without the head. The
difference between them is the thesis's cleanest ablation: how much of the
gain is *seeing which ink is trace*, and how much is *reading the waveform
rather than the ink*.

### 2.2 Input and output

**Revised 2026-09-30 on measurement** (`results/ml/label_ceiling.*`, all
2,000 training strips): the window is 48 mm, not 32, and the head outputs two
columns per input column, not one. Both numbers below are what a *perfect*
model could score, so they bound decision rule 1 from above.

| | 32 mm window | 48 mm window |
|---|---|---|
| strips clipped, baseline centred exactly | 6.7% | 1.1% |
| strips clipped, centre off by up to 4 mm | 20.8% | 2.0% |

(A strip counts as clipped if any sample leaves the window; within a clipped
strip it is a few columns at the tallest R wave, and those columns are
excluded from the loss via `in_window`, not trained toward the edge.)

Lead II R waves reach 19.6 mm above baseline at p99.

| Head output | R-peak ratio of a perfect model, median | strips < 0.90 |
|---|---|---|
| 1 column per input column (8 px/mm) | 0.953 | 7.5% |
| **2 columns per input column (16 px/mm)** | **0.989** | **0.45%** |
| perfect ink mask + column centroid (model A) | 0.80 | 86% |

One output per 8 px/mm column averages ~2.5 samples at the QRS apex and caps
the ratio too close to the 0.90 rule. The last row is the plan's §0
correction, now measured: a *perfect* segmentation still reads R peaks at 0.80,
so model A cannot pass rule 1 however well it is trained.

- **Canonical scale.** After calibration (the duration anchor), resample the
  levelled strip crop to a fixed **8 px/mm**: a 10 s strip is then 2,000 px
  wide. Crop or pad to **384 px tall (48 mm, ±24 mm)** centred on the strip's
  baseline. The model always sees the same geometry, whatever the photo's
  resolution, and output rows convert to mV exactly. 8 px/mm is above the
  ~3 px/mm floor where the 1 mm grid resolves and below the phone's
  ~11 px/mm, so no detail is invented. The mapping is
  `ml/strips.to_canonical` (INTER_AREA resize, integer crop, padding with the
  image's median colour); it is the function that gets Kotlin golden vectors.
- **Input:** RGB, 384 × 2048 (padded), uint8, normalised inside the model.
  Keep colour. The grid is red and the trace neutral, and the network can use
  that where a threshold could not.
- **Output:** per *output* column (4,096 of them, 16 px/mm), a 384-way
  softmax over rows, plus a presence logit. Row = Σ p·row. Mask the output
  where presence < 0.5. The existing gap interpolation and the gate take over
  from there.
- **Training runs on 512-column crops** (64 mm): the full frame does not fit
  the RTX 3050's 4 GB even at batch 2. The network is fully convolutional and
  runs on the full width at inference.
- **Fully convolutional**, so the 3×4 block's 2.5 s leads (500 px wide at
  8 px/mm) go through the same network unchanged once FR4 exists.

### 2.3 Architecture sketch

- Encoder: 4 stages, stride 2 each, channels 16 → 32 → 64 → 96,
  depthwise-separable 3×3 blocks with batch norm and ReLU6. MobileNetV2-style
  inverted residuals are the obvious off-the-shelf choice.
- Decoder: U-Net skip connections back to full resolution, 16 channels.
- Heads: 1×1 conv → 2 channels, reshaped to a (384 × 2W) logits map (output
  column 2w + c). A softmax **down each column** gives the row distribution.
  A separate column-pooled branch gives presence.
- Built: `ml/train_tracer.py`, **389,580 parameters** (two dilated blocks at
  1/16 added for vertical context). Compute at 384 × 2048 is roughly
  1–1.5 GFLOP (*expected*).

### 2.4 Training

- **Targets:** for each column, the rows of the truth samples falling in it,
  from `plotted_pixels` mapped into the canonical frame. At 8 px/mm and 500 Hz
  that is ~2.5 samples per column. Target distribution: a Gaussian (σ ≈ 1 px)
  around each sample's row, normalised. Presence = 1 where the column holds a
  sample.
- **Loss:** per-column cross-entropy (KL) against the target distribution,
  + λ·L1 between the soft-argmax row and the mean target row, + BCE on
  presence. The L1 term is what directly penalises the R-peak under-read.
- **Framework:** Keras / TensorFlow. The runtime decision is already LiteRT,
  and Keras → LiteRT INT8 conversion is the best-trodden path. PyTorch would
  add a conversion step (e.g. via Google's AI Edge Torch) that is itself a
  thesis risk. No deep-learning framework is committed to the repo yet; add it
  under a separate `ml/` directory so the classical `python/` stays
  dependency-light.
- **Compute:** 2,000 strips × on-the-fly augmentation trains in hours on one
  consumer GPU or a free cloud notebook (*expected*).

---

## 3. Data

### 3.1 Splits: never random

Folds from PTB-XL's `strat_fold`: **1–8 train, 9 validation, 10 test**. The
image inherits its record's fold, and the fold goes in the filename.
~21,800 records come from ~18,900 patients, so a random split leaks patients
and inflates every number. The printed test sheets (PTB-XL 9, 38, 40, 57) are
all fold 10 and must never appear in training.

### 3.2 Generation

`make_kit_target.py` already renders a record with exact per-sample truth.
Scale it to **2,000 fold-1–8 sheets** (and ~300 fold-9). Do not generate
20,000 before 2,000 is shown to be insufficient.

**Keep labels exact.** Some of the kit's augmentations (wrinkles, creases,
rotation, crop) move pixels. Whether `plotted_pixels` follows them has not
been checked. So:
- Render **clean** sheets with truth. Then apply **our own** geometric and
  photometric augmentation, whose transforms are known, to image and labels
  together.
- Use kit augmentations that do not move pixels (noise, colour temperature)
  freely.
- Before trusting any pixel-moving kit augmentation, re-run the check that
  all 20,000 of 20,000 `plotted_pixels` land on ink.

### 3.3 Augmentation: aim it at the failures measured on the real photo

Randomise every condition the real photograph showed to break the classical
path:

| Measured failure | Augmentation |
|---|---|
| Printer dropped the 1 mm grid | minor-grid colour from near-white (dropped) through `MINOR_GRID_RGB` to dark; also the kit's other grid palettes |
| Grid rules as dark as the trace (dim light) | illumination gradients strong enough to push rules to trace darkness; local shadows |
| Trace 2–3 px thick, strokes breaking | render at the phone's ~11 px/mm and downsample with bilinear filtering (Android's) before resampling to 8 px/mm; vary trace grey 0–0.2 and line width |
| JPEG colour bleed | JPEG 60–95 with 4:2:0 chroma subsampling |
| Imperfect corners | residual perspective and local tilt up to ±0.5°, small non-uniform stretch (±2%) |
| Text near the strip | lead labels and the `25mm/s 10mm/mV` legend kept in the crop |
| Pulse at the strip start | pulse kept in some crops, labelled presence = 0 |

`score_kit_photo.simulate` already implements perspective, blur,
illumination, noise and JPEG with the ground truth carried through. Extend
it rather than writing a second simulator.

### 3.4 Evaluation sets

1. **Validation:** fold-9 renders + simulated photos, for model selection only.
2. **Simulated-photo test:** fold-10 renders through `simulate`, all
   conditions. It is large, so tight error bars.
3. **Real-photo test: the headline.** Printed fold-10 sheets photographed on
   the A30s. Currently **n = 1 photograph of 1 sheet**, which is an anecdote.
   **Minimum before any claim: 4 sheets × 3–5 shots (12–20 photos).
   Target: 10 sheets.** This needs a printer and an afternoon, not a GPU,
   and it is the cheapest improvement to the whole thesis.

Report all three. The render → photo gap is a result, not a nuisance.

---

## 4. Decision rules: declared before training

Adopt the learned tracer only if **all three** hold on the **real-photo
test set**, against the classical control at the **same corners and the same
calibration**:

1. **Accuracy.**
   - R-peak height ratio **≥ 0.90** (classical: 0.72).
   - Aligned PCC **≥ classical + 0.08**, over ≥ 8.5 s (classical run 5:
     0.74; replay with accurate corners: 0.83–0.86).
   - No loss of coverage or new gaps.
2. **Latency.** Added `PIPELINE` time on the A30s **≤ 600 ms** for the rhythm
   strip, warm, scaffolding excluded. The total stays **≤ 5 s** per sheet.
3. **Memory.** Peak additional native + heap **≤ 150 MB**. Memory must be
   instrumented first; it is not today.

If the model fails a rule, it stays out and the failure is reported, with its
numbers. **Never remove or "improve away" the classical path.** It is the
control, and every ML claim is relative to it.

---

## 5. On-device

- **Runtime:** LiteRT, CPU (XNNPACK). Treat NNAPI/GPU as a bonus reported
  separately. The APK is `arm64-v8a` only, and emulator numbers are invalid.
- **Quantisation:** INT8 post-training quantisation with ~200 representative
  strips, **including real-photo-like ones**. The risk is sub-pixel precision
  in the soft-argmax. Measure INT8 against FP32 on the real-photo set. If
  INT8 costs > 0.02 PCC or > 0.03 R-peak ratio, try FP16 and report the
  size/latency trade.
- **Why per-strip, never per-sheet:** a whole sheet at 8 px/mm is ~2,200 ×
  1,500. The model runs only on the strip crop (256 × 2048), and later on
  12 lead crops of 256 × 512, about 3× the rhythm-strip cost in total
  (*expected*).
- **Port fidelity:** the model file is shared by both sides; the risk is in
  the arithmetic around it. Canonical resampling, the column → time map and
  the soft-argmax get golden vectors asserted between Python and Kotlin, the
  same pattern as `PhotoPortFixturesTest`.

---

## 6. Phases and timeline

**Revised 2026-10-01 to the two-week plan agreed with the user** (deadline
~2026-10-14). The original five-week outline assumed the photo session and
memory instrumentation came first; in practice the data and model went first
(days 1–2) and the device work is the critical path.

| Days | Dates | Work | Status |
|---|---|---|---|
| 1–2 | 09-30 – 10-01 | Data, augmentation, label ceiling, WSL/TF, model B b0 and b1, first real-photo comparison | **done** |
| 3 | 10-02 | Photos of PTB-XL 9, 38, 57 (3–5 shots each); register corners; classical + b1 scored **once** | user photographs |
| 3–4 | 10-02 – 10-03 | Model A (`--head mask`, ~2 h) and its real-photo score: the A-vs-B ablation | |
| 4–5 | 10-03 – 10-04 | INT8 LiteRT export at the full 384 × 2048 frame; INT8 vs FP32 on the real photos (§5) | |
| 5–8 | 10-04 – 10-07 | Kotlin: `to_canonical` port with golden vectors, LiteRT in the app behind a switch, decode, gap fill | **critical path** |
| 8–9 | 10-07 – 10-08 | Memory instrumentation; latency and memory on the A30s (rules 2, 3) | |
| 9–10 | 10-08 – 10-09 | Phone runs of all sheets, both tracers, same corners | |
| 11–14 | 10-10 – 10-14 | Write-up; small fixes only (8.5 s rule in `score_kit_photo`, merge to `main`) | |

**On track?** Yes for the model, which is a week ahead of the original
outline. The risk has moved to the device: nothing has run on the A30s, the
model runs at 384 × 2048 against a 600 ms budget that is only *expected* to
fit, and memory is not yet instrumented. If the Kotlin integration slips past
day 8, cut model A's device run (keep its desktop score) before cutting the
phone measurement of B, which is the thesis's actual question.

Practical constraint learned: Claude Code's background-job reaper stops long
runs when the laptop is low on memory (b0 at epoch 19, b1's first attempt at
epoch 4). Run training from a WSL terminal by hand.

**Classification** stays optional and last. If it is done, window PTB-XL
exactly as the printout does (2.5 s per lead at the printed offsets, 10 s for
lead II), and use a multi-label sigmoid/BCE head over the five superclasses
(never softmax). Report the degradation curve: true waveforms →
classically digitised → ML-digitised → photographed. A classifier trained on
12 × 10 s concurrent leads cannot consume a 3×4 printout's content, and
comparing the two would measure the wrong thing.

---

## 7. After the tracer, only if measurement says so

- **FR1 corners.** Hand placement costs PCC 0.83–0.86 → 0.63–0.74 on the
  same photo (measured). First try a classical page-quad detector (largest
  quadrilateral, then refine to the grid-line tips). If that fails on the
  real set, a **corner regressor** is the next ML component: 8 outputs from a
  256 × 192 downscaled photo, ~0.5 MB INT8, trained on the same renders with
  known perspective.
- **FR4 lead segmentation.** Classical row-density bands first, as the
  control. A small detector or U-Net over a downscaled sheet only if the
  classical version fails on the real set. Supervision: the kit's per-lead
  `plotted_pixels` extents. There is no `lead_bounding_box` field.

---

## 8. Expected results (*predictions; overwrite with measurements*)

Measured 2026-10-01 on six real photographs of PTB-XL 40 (desktop, matched
corners, `results/ml/photo_compare_b1.csv`): classical 0.85–0.89 / 0.59–0.73
where scorable (3 of 6); **model B (b1) 0.83–0.93 / 0.89–0.95 on all 6.**
Model A and INT8 are not measured yet. The predictions below stand as
written, for comparison.

| Condition | Aligned PCC | R-peak ratio | Basis |
|---|---|---|---|
| Classical, phone, as-placed corners | 0.74 | 0.72 | measured, run 5 |
| Classical, accurate corners | 0.83–0.86 | ~0.75 | measured replay; ratio expected |
| Model A (ink mask + centroid) | 0.85–0.88 | 0.75–0.80 | removes grid contamination and gaps, keeps the midpoint bias |
| **Model B (row head)** | **0.88–0.93** | **0.90–0.97** | also reads the tip; bounded by corners and residual warp |
| Model B, INT8 on the A30s | B − 0.00 to 0.02 | B − 0.00 to 0.03 | quantisation risk |
| Added latency, rhythm strip | 150–500 ms | | ~0.5–1 GFLOP on an A73-class CPU |

The prediction worth stating up front, because it is falsifiable: **A buys
most of the PCC gain, and B buys most of the amplitude gain.** If B does not
beat A on R-peak ratio, the row-head formulation is not doing what §2.1
claims.

The likely thesis contribution, updated:

> On a consumer phone, once the time base is anchored externally (the
> printed recording's known duration, with the calibration pulse removed),
> paper-ECG digitisation is limited by trace extraction and page geometry,
> not by calibration. A compact learned tracer that reads the waveform's
> position per column, rather than segmenting ink, recovers the QRS
> amplitude the classical centroid loses, at a measured cost in latency and
> memory on the target device.

And the methodological findings already measured, which stand regardless of
the model:
- The printer is part of acquisition (an office printer dropped the grid
  colour the classical digitiser relied on).
- Self-consistent checks confirm themselves (the gate cannot catch an
  anchor error, and accepted a trace with the pulse in it).
- Repeatability overstates accuracy by about two (earlier `ecg00001` work).

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Real-photo test set too small | **high** | Photo session in week 1; nothing is claimed on n = 1 |
| Render → photo domain gap | high | §3.3 augmentation aimed at measured failures; report both numbers |
| Labels misaligned under augmentation | high | Apply our own known transforms; re-verify labels land on ink |
| INT8 breaks sub-pixel precision | medium | FP16 fallback; measure on real photos |
| Model B no better than A | medium | That is a reportable result; A alone may still pass the rules |
| Corners dominate the remaining error | medium | Matched-corner comparisons; FR1 work in §7 |
| Memory over budget | medium | Instrument first; per-strip inference only |
| Time | high | Classical + model A is a complete thesis; B is the upside, classification optional |
