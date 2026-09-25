# The machine learning element: plan, decision rules, and expected results

Status: written 2026-09-22, against the measurements in `results/` as of that
date. Every number quoted as *measured* has a CSV behind it and is cited.
Every number quoted as *expected* is a prediction with its reasoning shown,
and should be overwritten by a measurement as soon as one exists.

---

## 0. The one thing that decides everything else

The thesis contribution is a **deployment characterisation**, not a novel
architecture. That is not modesty, it is a constraint with teeth: it means
**no model earns a place in this project by being good in the abstract.** It
earns a place by beating the classical control condition on a committed
measurement, at a stated cost in latency and memory on the Galaxy A30s.

So the ML plan is not "which architecture". It is:

1. Find where the classical path measurably fails.
2. Rank those failures by how much they cost the end-to-end result.
3. Attack them in that order, and only with ML where ML is the right tool.
4. Gate each addition on a pre-declared threshold.

Steps 1 and 2 are already done, and they produce an answer that is not the
obvious one. Read section 2 before deciding anything.

---

## 1. What is already built, and what state it is in

| Stage | Status | Classical performance |
|---|---|---|
| FR1 acquisition / rectification | **absent** | hand-cropped ROIs stand in |
| FR2 deskew | works | 0.041 deg worst-case error, 74–89 ms |
| FR3 grid estimation | **fails on photographs** | NaN on 6 of 6 frames |
| FR4 lead segmentation | **absent** | — |
| FR5 trace segmentation | works, fragile | Otsu discontinuous in the crop |
| FR6 vectorisation | works, biased | gain 0.47–0.57 on photographs |
| FR7 calibration | works, imprecise | pitch accurate to 1–3%, needs 0.1% |
| FR8 quality gate | works | 0/36 false rejects, 3 blind spots |
| Classification | **absent** | — |

Ported to Kotlin and running on-device: FR2, FR3, FR5–FR8. Not yet run on
the handset for FR5–FR8.

---

## 2. Where the classical path actually fails, ranked by cost

This is the section that should drive the whole ML decision, and it
overturns the intuitive ordering.

### 2.1 The dominant error is the time base, not the tracer

`results/gate/pitchsens.csv` established the sensitivity: a **+0.5%** error
in recovered px/mm drops reconstruction PCC from 0.96 to **0.564**, and
+1.0% to 0.371, because `px_per_s = px_per_mm * 25` makes a pitch error a
time-base error that accumulates over the strip. A usable 10 s
reconstruction needs pitch to about **0.1%**.

What we actually achieve on a photograph (`results/photo/pitch.csv`,
`source.csv`):

| method | reproducibility | accuracy |
|---|---|---|
| grid autocorrelation, paper band | — | −3.3% / +1.6% / −3.1% |
| grid autocorrelation, over trace ink | — | +6.1% / +11.3% / **−46.4%** |
| external 10 s duration anchor | 0.87% | +1.9% / −8.1% / +0.35% |

Best case is roughly **1–3%**, which is **10–30× too coarse**. And the
page itself is tilted: the pitch measured at the top of the sheet differs
from the bottom by **+9.76%** and **+9.47%** on the two frames that yield
both.

### 2.2 Therefore: improving the tracer alone will not move the result

This is the single most important prediction in this document, and it
follows arithmetically from 2.1.

The ceiling measured against the clean digital original
(`results/photo/source.csv`) is PCC **0.407 / 0.441 / 0.472** — and that is
*after* optimising pitch and lag per frame. A learned tracer that
reconstructs the trace perfectly, fed a scale that is 2% wrong, still
produces a waveform whose last beats are compared against the wrong beats.

**Consequence for the plan: calibration and rectification come before any
learned tracer.** Build the tracer second and its benefit is measurable;
build it first and it is invisible, and the thesis records a null result
that is an artefact of ordering.

### 2.3 What the classical tracer genuinely loses

Amplitude. `gain_ratio_best` is **0.574 / 0.501 / 0.471** — the recovered
signal is about half the true amplitude. The cause is known and structural,
not a bug: `trace_positions` takes the ink centroid of each column, and on
a steep QRS upstroke one column holds a tall vertical run of ink whose
centroid sits at the run's midpoint. Measured on synthetic renders at gain
0.60 at 4 px/mm rising to 0.96 at 16 px/mm.

This is the clearest thing a learned tracer beats, and `plotted_pixels`
gives exact supervision for it. But see 2.2: it is worth doing **second**.

### 2.4 FR4 does not exist, and it blocks 11 of 12 leads

Only the rhythm strip is currently digitisable. The 3×4 block carries the
other 11 leads and the tracer, handed a whole column, returns the average
of four stacked traces — measured at PCC 0.273 with every calibration
statistic clean. `QualityGate.tracesPerColumn` correctly refuses it.

### 2.5 FR3 fails for two mundane reasons, and one is a one-line fix

`estimate_grid_pitch` returns NaN on all six photographs. Causes:

- A photograph's projection carries an illumination envelope that the mean
  subtraction in `_pitch_1d` does not remove, flattening every comb peak
  below `DEFAULT_PEAK_FRACTION`. `run_photo.py` subtracts a 61 px moving
  average and the 5 mm comb appears immediately at cv 0.007–0.012.
- No channel response makes the median-peak-spacing rule work. `redness`
  gives the most *regular* comb (cv 0.0074–0.0117) because it sees only the
  5 mm family; luma/green/blue carry the 1 mm rules but at cv up to 0.59.
  1 mm/5 mm spectral power: redness **0.097–0.169**, blue 1.02–1.50, green
  0.005–1.87, luma 0.48–2.44.

Neither needs ML. **Do these before reaching for a model** — it is the
cheapest large gain available, and a thesis that adds a neural network
where a high-pass filter was missing will be asked about it in the viva.

---

## 3. The plan, in dependency order

### Phase 0 — classical fixes, no ML (days)

Not optional and not ML, but listed because everything downstream is
measured against it and because skipping it corrupts every later comparison.

1. Detrend the projection in `_pitch_1d`. Verify against the committed
   synthetic measurements that nothing there changes.
2. Replace the peak-height selection rule with a harmonic test (the case is
   already argued in CHANGES.txt §11.5 and is now a requirement, not a
   preference).
3. Replace Otsu in `binarise_trace`. The trace-to-paper separation on
   `255 − R` is ~120 levels wide (trace ≈ 205, paper ≈ 79, grid ≈ 85); Otsu
   is not looking for it and its threshold moves discontinuously with the
   crop — 12.09% ink to 4.47% across a 44-row change.
4. Strip the kit's 1 px black page frame at load; it binarises as trace.

**Re-run `run_photo.py` after each one, separately.** One change at a time,
and check that you did — the repository has been burned by this before.

### Phase 1 — rectification (FR1). The highest-value ML-adjacent work.

The ~9.5% vertical pitch gradient cannot be fixed by any estimator; it is
geometry. Two routes:

- **ML Kit Document Scanner** — already the decided approach, ships as a
  Play Services module, no training, no model to quantise, no thesis risk.
- **A learned corner regressor** — four (x, y) outputs from a downscaled
  image, trained on kit renders with the kit's own perspective
  augmentation. ~1–2 MB INT8, runs on one small input.

**Recommendation: use ML Kit first.** It is free, it is the documented
decision, and it lets you measure how much rectification is worth before
spending weeks on a model. If ML Kit's rectification proves insufficient
(it is tuned for documents, not for sub-percent metric accuracy), the
corner regressor becomes justified *with a measurement behind it* — which
is exactly the standard this project holds itself to.

### Phase 2 — lead segmentation (FR4). The strongest pure-ML case.

You now have exact supervision, validated: `make_kit_target.py` writes the
kit sidecar, and **20,000 of 20,000 `plotted_pixels` land on ink** in the
final sheet once the recorded offset is applied. Per lead you get
`lead_name`, `start_sample`, `end_sample` and every plotted sample's pixel.

Recommended approach, in increasing cost:

1. **Classical first, as the control.** Row-wise ink density already finds
   four trace bands reliably on a rectified sheet. Column boundaries follow
   from the 4-column convention plus the printed lead labels. Measure it.
   A decent chance this is sufficient and no model is needed — and if so,
   that is a *result*, not a disappointment.
2. **Small U-Net or a lightweight detector** if the classical version fails
   on real photographs. Input a downscaled sheet (e.g. 512×384), output 13
   region masks or 13 boxes. This is small: ~1–3 MB INT8.

Metric: IoU against `lead_bounding_box`, plus the practical one — does the
tracer produce one waveform per lead rather than a splice.

### Phase 3 — learned trace extraction (FR5/FR6).

Only now, because of 2.2.

**Architecture:** a small encoder–decoder (U-Net family) doing binary
trace-pixel segmentation on a single lead region, followed by a
column-wise expectation over the predicted mask to get a sub-pixel row per
column. Do **not** regress the waveform directly from pixels — segmentation
plus a deterministic reduction keeps the output interpretable, keeps the
failure modes visible, and lets you reuse the existing calibration and gate
code unchanged.

**Why this beats the centroid:** the centroid averages all ink in a column,
including grid remnants and the vertical run of a QRS upstroke. A
segmentation model learns which ink is trace, and the expectation over a
*thin* predicted mask does not have the midpoint bias.

**Supervision:** rasterise `plotted_pixels` into a mask. You have per-sample
truth, so this is cheap and exact.

**Size budget:** on the A30s, target **≤ 3 MB INT8** and one forward pass
per lead region at ≤ 256×1024. Twelve regions then cost twelve small passes
rather than one huge one — see section 6.

### Phase 4 — classification, and a methodological trap

CLAUDE.md scopes this as *optional*. Keep it optional, and if you do it,
get the framing right, because there is a trap here that would invalidate
the result.

**The trap.** A 3×4 printout carries only **2.5 s per lead**, and the four
columns are *consecutive* time windows — so leads in different columns are
not simultaneous. Only the rhythm strip spans 10 s. Total printed content
is ~33% of the recorded lead-seconds. **A classifier trained on 12 × 10 s
concurrent waveforms cannot consume digitiser output from such a printout**,
and reporting its AUROC on digitised input would be comparing two different
input distributions.

Three honest options:

| option | what it is | verdict |
|---|---|---|
| **A. Rhythm-strip only** | classify from 10 s of lead II | Cleanest. Single-lead classification is an established task with published baselines. Under-powered for lead-specific findings (V-lead ST changes), and say so. |
| **B. Match the printout** | window PTB-XL exactly as the printout does — 2.5 s per lead at the right offsets, 10 s for lead II — and train on *that* | **Recommended.** Methodologically correct, cheap (it is a windowing function, not a new model), and it makes the digitiser's output and the classifier's input the same distribution by construction. |
| **C. Ignore it** | train on 12 × 10 s, test on digitised | Do not. It is the comparison that looks best and means least. |

**If you do B**, the classifier itself is unremarkable and that is fine: a
1-D CNN or small ResNet over the windowed leads, **sigmoid outputs and
binary cross-entropy — never softmax**, because PTB-XL is multi-label (a
record can be MI and STTC at once). Five superclasses: NORM, MI, STTC, CD,
HYP. Report **macro AUROC** (PTB-XL benchmark convention) and optionally
macro F-measure (PhysioNet Challenge 2024 convention).

**The interesting experiment is not the classifier's accuracy.** It is the
*degradation curve*: macro AUROC on true waveforms, then on
classically-digitised, then on ML-digitised, then on photographed-and-
digitised. That curve is a deployment characterisation result and nobody
else in this project's reference set has it for a phone.

---

## 4. Data strategy

**Generation.** `make_kit_target.py` already does this. Scale it up:

- Draw records by `strat_fold`: **folds 1–8 train, 9 validation, 10 test.**
  **Never split randomly** — ~21,800 records come from ~18,900 patients and
  a random split leaks patients across the boundary and inflates results.
  The *image* inherits the *record's* fold. This is easy to get wrong when
  images are generated into flat directories; put the fold in the filename.
- Use the kit's own augmentation for training data: `--wrinkles`,
  `--hw_text`, creases, `--rotate`, `--noise`, `--crop`, `--temperature`.
  That is what the kit is for, and it is the cheapest domain randomisation
  available. Note the handwriting path needs TensorFlow and spaCy, which
  the clean-render path does not.
- Quantity: **2,000–5,000 sheets** is ample for the tracer and FR4. Do not
  generate 20,000 before checking that 2,000 is insufficient.

**The domain gap, stated plainly.** Training data is *rendered*; test data
is *printed and photographed*. That gap is the thesis's subject, not a
nuisance to be hidden. So:

- Train on kit renders (with augmentation).
- Validate on held-out kit renders (fold 9).
- **Report the headline on real photographs of printed fold-10 sheets.**
- Report both numbers. The gap between them *is* a result.

**Sample size for the real-photograph test set.** Currently n=3 usable
frames of one page. That is an anecdote. Target **≥ 10 sheets × 2–3 shots
= 20–30 frames**, spanning lighting and angle. This is the cheapest
improvement available to the whole thesis and it needs a printer, not a GPU.

---

## 5. Decision rules — declare these before training anything

Pre-registering the thresholds is what stops a marginal model being
argued into the thesis after the fact.

**Adopt a learned component only if all three hold:**

1. **Accuracy.** It beats the classical control by a pre-stated margin on
   the *same committed measurement*, on the *real photograph* test set, not
   on renders. Suggested margins:
   - FR5/FR6 tracer: **gain ratio ≥ 0.85** (against 0.47–0.57 measured) and
     **PCC improvement ≥ 0.10** at matched calibration.
   - FR4 segmentation: **≥ 95% of leads** produce a single-trace region, and
     `tracesPerColumn` ≈ 1.0 per region.
2. **Latency.** Added `PIPELINE` time on the A30s is **≤ 2× the current
   classical stage cost** it replaces, and the total stays under a
   defensible interactive budget. State the budget now: **≤ 5 s** for a
   complete single-sheet digitisation, measured on the handset, warm-up
   discarded, scaffolding excluded (limitation 9).
3. **Memory.** Peak additional native + heap is **≤ 150 MB**, and the app
   does not OOM on the A30s at full capture resolution. Memory is currently
   the *uninstrumented half* of the deployment cost — instrument it before
   Phase 3, not after.

**If a learned component fails any gate, it stays out and the failure is
reported.** "We tried a U-Net for trace extraction; it improved gain ratio
from 0.50 to 0.88 but cost 4.2 s per sheet on the target device, exceeding
the declared budget" is a *better thesis result* than a quiet adoption,
because it is the deployment characterisation the project promised.

**Never delete or improve away the classical path.** It is the control
condition. Every ML claim in the report is a claim relative to it.

---

## 6. On-device reality, and the sizing arithmetic

Target: **Galaxy A30s (SM-A307FN)**, a 2019 mid-range device. Runtime:
**LiteRT**. APK is filtered to `arm64-v8a`, so the emulator cannot run it —
and emulator numbers are invalid anyway because it runs on host x86 with
active cooling.

**Assume CPU inference.** Do not plan around NNAPI or a GPU delegate on this
chip; treat any acceleration that works as a bonus and report it separately.

**The arithmetic that shapes the architecture.** A rectified sheet at
~20 px/mm is roughly 5,500 × 4,000 = **22 MP**. A U-Net run over all of it,
tiled at 512×512, is ~86 tiles. At even 150 ms per tile that is **13 s** —
far outside budget. Therefore:

- **Never run a dense model over the whole sheet.** Segment leads first
  (FR4, on a heavily downscaled image), then run the tracer **per lead
  region** at the minimum resolution that preserves the trace.
- A single lead region is ~1,400 × 250 px. At 256×1024 input that is one
  modest pass; twelve of them is tractable.
- Reference point you already have: Kotlin grid estimation is **205 ms at
  8.41 MP** on this device, and OpenCV resize/warp are fast. The classical
  chain is cheap; the model is what threatens the budget.

**Quantisation.** INT8 post-training quantisation with a representative
dataset. Check the accuracy cost explicitly — a tracer whose sub-pixel
precision collapses under INT8 is a real risk given that 0.1% of scale
matters. If INT8 costs too much, try FP16 and report the size/latency
trade. **Measure quantised accuracy on the real-photograph set**, not on
renders.

**Port fidelity.** The project already has the right pattern: golden
vectors asserted to 1e-12 between Python and Kotlin, plus a mutation check
that the suite would fail if the port diverged. Do the same for any
pre/post-processing around the model. The model itself is one artefact
shared by both sides, so the risk sits entirely in the surrounding
arithmetic — which is exactly where `SkewEstimatorPortTest` found bugs
before.

---

## 7. Expected results

Predictions, with reasoning. Overwrite each with a measurement.

### 7.1 Reconstruction fidelity

| condition | PCC | gain ratio | basis |
|---|---|---|---|
| classical, clean kit render | 0.90–0.96 | 0.85–0.95 | synthetic strips measured 0.96 at integer px/mm; renders are the friendly case |
| classical, photographed (today) | **0.41–0.47** | **0.47–0.57** | measured, `source.csv` |
| classical + Phase 0 + rectification | 0.60–0.75 | 0.50–0.60 | removing a ~9.5% pitch gradient and a 1–3% scale error; pitchsens says +0.5% → 0.564, so getting to ~0.2% should reach ~0.75 |
| + learned tracer (Phase 3) | 0.75–0.88 | 0.85–0.93 | tracer fixes amplitude, not time base; PCC gain comes mostly via better trace continuity |
| ceiling if calibration were exact | 0.90+ | 0.90+ | the render case, minus photograph noise |

**The honest expectation is that ML buys most of the gain ratio and
relatively little of the PCC**, because PCC is dominated by calibration.
That prediction is falsifiable and worth stating up front.

### 7.2 Lead segmentation

Expect **classical row-density segmentation to work on rectified sheets**,
IoU ~0.9, and to fail on unrectified or poorly lit ones. Expect a small
U-Net to reach IoU 0.95+ and to be more robust to lighting. Expect the
honest conclusion to be that **rectification matters more than the
segmentation method** — the same shape as 2.2.

### 7.3 Classification

- True 10 s × 12-lead waveforms, 5 superclasses, macro AUROC **0.90–0.93**.
  This is the published PTB-XL benchmark range; do not expect to beat it and
  do not try — it is a reference point, not a contribution.
- Option B windowing (2.5 s per lead, 10 s lead II): **0.85–0.90**. The drop
  is the cost of the printout's ~33% content, and *quantifying that drop is
  itself a result worth reporting.*
- From classically-digitised photographs: **0.70–0.82**, wide error bars,
  and strongly dependent on calibration quality.
- From ML-digitised photographs: **0.78–0.86**.

Report the **degradation curve**, not the endpoint.

### 7.4 Deployment cost

| stage | expected on A30s | basis |
|---|---|---|
| FR1 rectification (ML Kit) | 200–600 ms | Play Services, device-optimised |
| FR2 deskew | 74–89 ms | measured |
| FR3 grid | ~205 ms at 8.41 MP | measured |
| FR4 learned segmentation | 150–500 ms | one small pass on a downscaled sheet |
| FR5/FR6 learned tracer | 100–300 ms × 12 regions = **1.2–3.6 s** | the binding term |
| FR7/FR8 | < 100 ms | pure arithmetic |
| **total** | **2–5 s** | at the edge of the declared budget |

Expect the learned tracer to be **the reason the budget is tight**, and
expect a real trade-off to report: per-lead inference count versus latency.
A defensible fallback is the learned tracer on the rhythm strip only, with
the classical tracer on the 3×4 block, and the cost of that asymmetry
measured.

Memory: expect peak native + heap of **150–400 MB** at full resolution,
dominated by the decoded bitmap and OpenCV Mats rather than the model. The
A30s ships with 3–4 GB shared with the OS, so this should fit but must be
instrumented, not assumed.

### 7.5 The result most likely to be the thesis's actual contribution

Not a model. This:

> On consumer mobile hardware, paper-ECG digitisation is limited by
> **metric calibration**, not by trace extraction. A learned tracer
> recovers amplitude that the classical centroid loses (gain 0.50 → ~0.90)
> but leaves reconstruction correlation largely unchanged, because a 1–3%
> error in recovered grid pitch is a time-base error that dominates the
> result. Sub-percent scale recovery from a hand-held photograph is the
> open problem.

Plus the methodological finding already measured, which generalises beyond
ECG:

> Repeatability overstates accuracy by roughly a factor of two. Two
> photographs of the same page agreed at PCC 0.62–0.85 while either one
> against the source page reached only 0.47, because the dominant errors
> were common to both photographs and cancelled between them.

---

## 8. Risks, and what to do about each

| risk | severity | mitigation |
|---|---|---|
| **Sub-percent calibration proves unreachable** | high — it is the headline | Report it as the finding. Add the 1 mV calibration-pulse detector (the pulse is printed and visible) to anchor gain externally, and use the duration anchor where duration is known. |
| INT8 destroys sub-pixel tracer precision | medium | Test FP16; report the trade. Measure on photographs. |
| Latency budget blown by 12 per-lead passes | medium | Rhythm-strip-only learned tracer as the documented fallback. |
| Domain gap too large (renders → photographs) | medium | Kit augmentation; report both numbers; the gap is a result. |
| Kit's pale minor grid, RGB (254,223,219) = BGR (219,223,254), unrecoverable from a photograph | **not measured on a kit print** (corrected 2026-09-24). The photographs behind this claim are of `ecg00001.png`, a different page with its own grid. On *simulated* photographs of the kit sheets the 1 mm rules survive in the redness autocorrelation (comb contrast 0.07-0.57), but the peak-height rule still locks onto 2x or 5x -- a selection failure, not missing information. | Resolve harmonic locks by testing sub-multiples (`digitise._resolve_harmonic`), and keep the 10 s duration anchor as the external check. |
| Scope creep into classification | medium | It is optional. Phase 4 only if Phases 0–3 are measured and written up. |
| Time | high | Phases 0 and 1 plus more photographs would make a complete, defensible thesis with no neural network at all. Treat Phase 3 as the upside, not the core. |

---

## 9. What to do in the next two weeks

1. Print the four sheets in `out/kit_targets/`, verify with a ruler,
   photograph each 2–3 times. **This unblocks the first real
   PCC-against-truth in the project** and costs an afternoon.
2. Phase 0, one change at a time, re-running `run_photo.py` after each.
3. Wire ML Kit Document Scanner and measure what rectification is worth.
4. Generate ~2,000 augmented kit sheets with fold-aware naming.
5. Classical FR4 by row density; measure it before writing a model.
6. Only then: decide whether Phase 3 is affordable in the time left.

If step 6 says no, the thesis is still complete. That is the point of
putting the classical control condition first.
