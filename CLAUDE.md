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

**Status: early.** The Python prototype exists and runs. The Android app
has not been started.

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

6. **`estimate_grid_pitch` reports the 5 mm rule, not the 1 mm rule, on
   the ecg-image-kit fixtures.** Measured 2026-09-16: it returns 39.26
   px/mm against a true 7.874, a 5x error, on all three reference
   printouts. The minor rules *are* resolved — autocorrelation peaks exist
   at lags 8, 16, 24 — but they stand at 0.11–0.19 against 0.87 for the
   5 mm peaks, so `peak_fraction=0.20` filters most of them out and the
   median spacing collapses onto the majors. This never appeared on
   `synth` renders, whose minor and major rules have similar contrast, and
   it is the strongest evidence so far that synthetic images flatter the
   pipeline. Nothing inside the autocorrelation can detect it (both
   `spacing_cv` and a major/minor height ratio were tried and fail);
   `digitise.check_calibration`'s paper-speed test catches it, and needs
   `expected_duration_s` to do so. Fixing the selection itself is
   deliberately left open — see `results/deskew/guards.csv`.

7. **Deskew must run before grid estimation.** At 1 and 3 degrees,
   digitisation of a rendered strip fails outright in 12/12 records
   (`estimate_grid_pitch` returns NaN). Deskew recovers it to within ~1%
   of the unrotated PCC. Rotation is not a tolerable input condition.

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

---

# TASK: Audit and report

Please read every file under `python/` and produce a written report.
Do not modify any code. The report will be handed to a collaborator who
does not have access to this repository, so it must stand alone.

Cover the following:

**1. Module inventory.** For each file: its purpose, its public
functions/classes with signatures, and what it depends on. Note anything
that is dead code or unused.

**2. Data flow.** Trace the path from a PTB-XL record on disk through to
a metric value. Name the exact functions called in order and the shape
and units of the data at each hand-off.

**3. Correctness review.** Identify actual bugs, as distinct from the
documented limitations listed above. Pay particular attention to:
   - unit handling (ADC counts vs millivolts vs pixels)
   - the sign convention in the pixel-to-millivolt conversion
     (image y grows downward)
   - `estimate_grid_pitch` — it previously locked onto a harmonic of the
     true grid frequency; assess whether the autocorrelation version is
     robust, and under what conditions it would fail
   - baseline estimation in `digitise()` (currently the median trace
     position — is that sound for a lead with large deflections?)
   - resampling in `evaluate.align_length` — does comparing signals of
     different lengths by resampling introduce bias?

**4. Missing pieces.** What must exist before this can (a) read a real
3x4 printout, and (b) run on Android. Be specific about which functions
would need to change.

**5. Test coverage.** There are currently no tests. Propose a minimal set
that would catch regressions in the numeric path — name the specific
properties worth asserting.

**6. Anything that looks wrong to you** that is not covered above.

Output as markdown with headed sections. Include short code excerpts
where they clarify a point, but do not reproduce whole files. Be direct
about problems; a list of everything that is fine is not useful.
