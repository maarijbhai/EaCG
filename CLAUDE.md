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

**Status: early.** The Python prototype exists and runs: rendering,
deskew, grid estimation, classical digitisation and the reconstruction
metrics, with committed measurements under `results/`. Lead segmentation
does not exist, and nothing has been measured on a physical device. The
Android app loads images and has one ported stage (`GridEstimator.kt`),
which is currently divergent from the Python — see limitation 6.

## Repository layout

```
ECG-App/
├── android/          Kotlin/Compose app; image loading only so far
│   └── app/src/main/assets/reference/   3 printouts of known px/mm
├── python/           the prototype (this is what exists)
│   ├── ecgproto/
│   │   ├── ptbxl.py      PTB-XL loader
│   │   ├── synth.py      signal synthesis + single-strip rendering
│   │   ├── layout.py     clinical 3x4 12-lead printout rendering
│   │   ├── deskew.py     stage 2: recover and remove in-plane rotation
│   │   ├── digitise.py   classical (non-ML) digitisation pipeline
│   │   └── evaluate.py   SNR / shifted SNR / RMSE / PCC
│   ├── run_baseline.py   experiment on synthetic signals
│   ├── run_ptbxl.py      experiment on real PTB-XL records
│   ├── run_deskew.py     deskew measurement suite -> results/deskew/
│   ├── run_gridchannel.py  grid channel/threshold suite -> results/grid/
│   ├── make_print_target.py  print a lead at true physical scale
│   ├── score_photo.py    score a photograph of that print
│   └── verify_ptbxl.py   dataset download integrity check
├── results/          committed measurements — CSV + provenance JSON
├── out/              gitignored — generated artefacts, reproducible
└── data/             gitignored — PTB-XL lives here, NOT in the repo
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

2. **`binarise_trace()` keys on the red channel**, which works because
   `synth.render_printout` draws the grid in red. Real ECG paper varies —
   faded thermal prints, blue-grey grids, black-and-white photocopies.
   This stage is expected to need replacement for real images.

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
   §11.5 and not acted on.

   Two things the earlier reading of this got wrong, both now measured.
   The green channel does not help — it is the *worst* of blue, redness,
   green and luma, because ecg-image-kit's minor-rule ink is (219,223,254)
   in BGR and green sees less of it than blue does. And `GridEstimator.kt`
   never recovered these fixtures either: run unmodified it returns NaN on
   all three, because its `0.25 * max(acf[3:60])` cut lands at 0.178–0.238
   — the same place as the Python's 0.20. The two ports agreed all along;
   only their failure modes differed.

   **The Kotlin port is currently divergent** and must change to match:
   redness not green, an absolute 0.05 threshold not `0.25 * windowMax`,
   and a lag window wider than its present `min(60, n-2)` (at 300 dpi the
   5 mm rule sits at 59.06 px, right at the edge).

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

## Technical decisions already made

- **Kotlin, not C++/NDK.** LiteRT inference and OpenCV already run
  natively under the hood; the JNI boundary costs a crossing, not the
  compute. CameraX, ML Kit and UI are Kotlin-only anyway.
- **LiteRT (formerly TensorFlow Lite)** as the mobile runtime.
- **ML Kit Document Scanner** for capture-time rectification in the
  first version (perspective correction, edge detection, cropping).
- **Emulator for correctness only.** It runs on host x86 CPU with active
  cooling, so latency, memory and thermal numbers from it are invalid for
  a thesis about mobile constraints. All benchmarking must use a physical
  device (one borrowed older Android phone is currently the only target).

## Conventions

- Python 3, NumPy/OpenCV/SciPy/pandas/wfdb. No deep learning framework
  committed to yet.
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
- **Vary one thing at a time, and check that you did.** The first
  interpolation comparison in `run_deskew.py` varied the warp used to
  *simulate* a rotation together with the warp used to *correct* it, and
  produced a confident, wrong conclusion. Simulation parameters and
  parameters under test need separate flags.

