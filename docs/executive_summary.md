# EaCG — Executive summary for the report

*Written 2026-10-05. Plain-language explanations with analogies, then the
numbers. Every number comes from a committed results file; the source is
given so it can be cited. The detailed, dated history is
`docs/progress_log.md`.*

---

## 1. The project in one paragraph

Hospitals and clinics still print ECGs on paper. A paper ECG is a picture: you
cannot measure it, store it as data, or feed it to software. **EaCG is an
Android app that photographs a printed 12-lead ECG and turns it back into the
numbers it was drawn from** — twelve waveforms in millivolts, 500 samples a
second — **entirely on the phone**, with no server. The thesis question is not
"can a new neural network do this?" (others have shown it can, on powerful
computers) but **"does it still work when it has to run on an ordinary,
older phone, and what does that cost in accuracy, time and memory?"**

> **Analogy.** Think of a paper ECG as a song printed as sheet music. The app
> photographs the sheet music and plays the song back. The question is
> whether a cheap phone can do it well, not just a studio computer.

The target phone is a **Samsung Galaxy A30s** (2019, mid-range).

---

## 2. Background you need to explain the work

### 2.1 What is printed on an ECG sheet

- **Twelve leads** are twelve views of the same heartbeat from different
  angles (I, II, III, aVR, aVL, aVF, V1–V6).
- The standard printout is a **3×4 layout plus a rhythm strip**: three rows of
  four leads, each lead shown for only **2.5 seconds**, then one long strip of
  lead II for the full **10 seconds** at the bottom.
  > **Analogy:** a TV split into twelve small channels, each shown for a
  > quarter of the programme, plus one big channel shown the whole time.
- **The grid is a ruler.** Small squares are 1 mm, big squares 5 mm. By
  convention the paper moves at **25 mm per second** (so one big square is
  0.2 s) and the pen moves **10 mm per millivolt**.
  > **Analogy:** the grid is graph paper with a known scale, like a map's
  > "1 cm = 1 km". Knowing the scale is what lets you turn a drawing back into
  > real distances.
- **Calibration pulse:** a small rectangle at the start of each row, exactly
  1 mV tall. Its falling edge marks **time zero**.
  > **Analogy:** a starting gun — everything is timed from it.
- **Separator ticks:** a short thick vertical bar wherever one lead hands
  over to the next, at **2.5, 5.0 and 7.5 s** in each of the three rows, so
  **nine ticks** per sheet.
  > **Analogy:** the kilometre markers on a road. If you know where marker
  > 2.5 is, you know exactly how far along you are.

### 2.2 Where the "true answer" comes from

**PTB-XL** is a public dataset of 21,799 real 10-second 12-lead ECG
recordings. They are *numbers*, not pictures. We print chosen recordings with
**ECG-Image-Kit** (a tool that draws realistic ECG printouts), photograph the
print, digitise the photo, and compare the result with the original numbers.

> **Analogy:** a dictation test. We know exactly what was read out, so we can
> mark every word the student wrote down.

Four test records have been printed and photographed: **PTB-XL 9, 38, 40 and
57**. All are from the **test fold** (fold 10), which the model never saw in
training.

### 2.3 How results are scored (and why there are several numbers)

| measure | what it asks | analogy |
|---|---|---|
| **PCC** (correlation, 0–1) | Is the *shape* right? Ignores size and vertical position. | Two signatures: same shape even if one is bigger. |
| **Aligned PCC** | Is the shape right *if* we forgive a small timing slip? | Marking a singer's notes even if they came in a beat late. |
| **SNR** (dB) | How big is the error compared with the signal, sample by sample? Shape *and* timing. | How loud is the static compared with the music. |
| **Shifted SNR** | The same, forgiving a constant slip of up to ±50 ms. Standard in the PhysioNet Challenge 2024. | Volume of static once the late start is forgiven. |
| **R-peak ratio** | How tall are the tallest spikes compared with the truth (1.0 = perfect). | Did the drawing keep the mountain tops, or flatten them? |

**Reading them together:** high aligned PCC but lower plain PCC/SNR means
"shape right, timing slightly off". Low aligned PCC means the waveform itself
was lost. Rough guide for SNR: 0 dB = error as big as signal; 10 dB = error a
tenth of the power; 20 dB = a hundredth.

---

## 3. How the app works (the pipeline)

The app does eight steps. Each one has a Python version (the reference, on
the laptop) and a Kotlin version (on the phone) that is tested to give the
same answer.

1. **Photo in.** The phone decodes the photo to 3,200 pixels on its long side.
2. **Find the sheet's corners** automatically.
3. **Straighten the page** using the corners (remove the camera angle).
4. **Grid fit:** straighten it *properly* using every grid line (§4.2).
5. **Locate:** find the four rows and the calibration pulses (time zero).
6. **Time scale:** find the nine ticks; they set the time for each lead.
7. **Trace:** read where the line is in every column — with the **learned
   tracer** (the neural network) or the **classical tracer** (the control).
8. **Cut into leads** using the ticks, convert pixels to seconds and
   millivolts, save and show the 12 leads.

> **Analogy for the whole thing:** scanning a crumpled, photographed
> spreadsheet. First flatten the paper (steps 2–4), find the column headers
> (5–6), read every cell (7), and put the numbers in the right columns (8).

A pipeline diagram (engineering drawing) was published earlier:
https://claude.ai/artifact/FxMTo4UfoKaPh1YnuMmqNk

---

## 4. What was built, in order — and why each step was needed

### 4.1 Phase 1 (September): the classical method, on the phone

**What:** a hand-written, rule-based method for the lead II rhythm strip:
straighten, find the strip, separate ink from paper ("binarise"), remove text,
and take the middle of the ink in each column.

> **Analogy for binarising:** turning a colour photo into a black-and-white
> stencil — ink becomes black, paper and grid become white.
>
> **Analogy for the classical tracer:** in each thin vertical slice of the
> picture, find the centre of the black ink. Works for gentle slopes, but at a
> sharp spike the ink is spread up the whole spike, so its "centre" sits
> halfway down. **It flattens the mountain tops.**

**Result:** on a real photo on the phone, PTB-XL 40 rhythm strip: shape PCC
0.83 after alignment, but R-peaks only ~0.70 of their true height, 2.0 s per
run (`results/device/photo_scores.csv`). This flattened-peak problem is
fundamental to the method and became the reason for the learned tracer. The
classical method is kept on purpose as the **control** the ML is judged
against.

### 4.2 Phase 2 (late Sept – early Oct): the learned tracer

**What:** a small neural network (a "U-Net", **389,580 parameters**) that
looks at a strip of the ECG image and, for every column, says *which row the
line is in* (with sub-pixel precision) and *whether there is a line at all*.

> **Analogy:** instead of averaging ink, the network has learned to *read*
> the line, like a person who has seen thousands of ECGs and can point to
> exactly where the tip of each spike is, even through blur and shadow.

**How it was trained** (`results/ml/train_b3.*`, `strips_rows_*.csv`):
- **2,000 PTB-XL records** (folds 1–8) rendered with ECG-Image-Kit →
  **8,000 row images** with the exact pixel of every sample known; **300
  records (1,200 rows)** from fold 9 for validation. Seven grid colour styles
  (red, pink, brown, blue, green, random, black-and-white).
- **Augmentation:** each training image is randomly made to look like a phone
  photo — tilt, stretch, blur, exposure, JPEG compression, sharpening — or
  like a flatbed scan.
  > **Analogy:** a driving instructor practising in rain, fog and at night,
  > so the real test is not the first time the student sees bad weather.
- Patients never overlap between training and testing (PTB-XL's official
  folds), so the scores are honest.
- Four versions were trained (b0 → b3), each fixing a measured weakness;
  **b3** is the one used everywhere and is now **frozen**.

**Result on the validation set:** average error **0.33–0.44 pixels** (about
0.04–0.06 mm) and R-peaks at **0.97** of true height.

**On the phone:** the model is converted to **LiteRT FP16** (0.82 MB) —
the phone's neural-network format, numbers stored at half precision.
> **Analogy:** saving a photo as a slightly smaller file that looks
> identical. Measured: the phone's answers match the laptop's to 0.0004 px.

It runs the page in overlapping pieces ("tiles", 512 × 1,024 pixels each),
because the full row is too big to process at once.
> **Analogy:** reading a long scroll through a window that you slide along.

### 4.3 The pivot (2 October): the whole sheet, all 12 leads

With supervisor agreement, the goal became **all 12 leads plus the rhythm
strip, on the phone, with the learned tracer**. Standard 3×4 layout only.
This needed new steps: find the four rows, find time zero for each row, use
the nine ticks to cut the rows into leads. Lead identity comes from position
(row 1 is always I, aVR, V1, V4, and so on).

### 4.4 Separator ticks set each lead's timing

Originally one time scale was used for the whole page. Because a photo is
never perfectly flat, later leads drifted tens of milliseconds early. Now
**each tick pins the time at its lead boundary**, so every lead gets its own
local time scale; a missing tick borrows the position of the same tick in the
other rows.
> **Analogy:** resetting your stopwatch at every kilometre marker rather than
> trusting one watch for the whole marathon.

### 4.5 Automatic corners, and a "truth" to check them

The app finds the sheet's corners itself (no tapping). To *measure* how good
any geometry is, the photo is precisely overlaid on the original printed
image ("registration") — used **only for testing**, never in the app.
> **Analogy:** laying a transparent copy of the original over the photo to
> see how far off things are.

### 4.6 The grid fit — the biggest single improvement

**Problem:** straightening the page from just four corners spreads any
corner error across the whole page, and a page that is slightly stretched is
a clock that runs slightly fast or slow.

**Fix:** use **every 5 mm grid line** (about 55 × 37 of them) and fit the
page so that all of them are perfectly straight and evenly spaced.
> **Analogy:** instead of pinning a rubber sheet at its four corners, you
> pull it until *every* square on the graph paper is square. Four pins can
> be off; hundreds of squares cannot all be wrong together.

**Measured** (`results/kitphoto/geometry.csv`): page distortion from
**0.27–1.75 mm** down to **0.035–0.094 mm**; the corners stopped mattering
(automatic corners + grid fit = perfect corners + grid fit, to 0.002 mm).
Median lead PCC (PTB-XL 40, 72 leads): **0.865 → 0.966**; SNR **5.8 →
11.4 dB**. Now on the phone (1.8–2.8 s per sheet).

### 4.7 A failure on the phone, found and fixed (the "3×3 eraser")

A new photo of PTB-XL 9 came back **completely wrong** on the phone (every
lead lost; it measured the scale as 6.52 px/mm instead of ~9.9).

**Cause:** while finding the rows, the black-and-white stencil was cleaned
with a 3×3-pixel "eraser" (a morphological *opening*) to remove specks. The
ink line in this photo was only 2–3 pixels thick, so the eraser **rubbed out
most of the line itself**. The strip then looked like it ended
three-quarters of the way across the page, so every later step was scaled
wrongly.
> **Analogy:** cleaning a pencil drawing with an eraser too big for the
> thinnest lines — the specks go, but so do half the lines.

**Fix:** a 2×2 eraser, as the tracing step already used. Same photo:
**0.04 → 0.84** (and **0.91 on the phone** with the grid fit). A test was
added that fails under the old rule. *Trade-off, also measured:* the smaller
eraser leaves some grid-line fragments, which occasionally hide a
calibration pulse on darker rows (§6).

### 4.8 The one-pixel timing fix (5 October)

**Problem:** timing was still off by a few milliseconds everywhere.

**Investigation:** measuring each half-lead's timing and working backwards
showed **every time marker (pulse edge and ticks) was read about 4 ms late**,
even on the clean printed image with no camera at all.

**Cause:** a 2×2 eraser has no centre pixel, so the image library shifts the
cleaned stencil **one pixel to the right**. The markers were read from the
shifted stencil; the neural network reads the original image. The two
disagreed by one pixel ≈ 4 ms.
> **Analogy:** a rubber stamp whose handle is off-centre — every stamp lands
> one step to the right of where you aimed.

**Fix:** subtract that pixel. On the clean prints, plain SNR rose from
**~11 dB to ~22 dB** (timing essentially exact). On all 9 photos (108 leads),
grid path: PCC **0.943 → 0.952**, leads ≥ 0.9 **76 → 92**, SNR **9.7 → 10.3
dB** (`results/ml/phone_path_b3_512x1024_fp16_grid.csv`).

### 4.9 The Records tab

Every scan is saved privately on the phone (photo, straightened page,
traced lines, 12-lead CSV) and can be reopened: the 12 leads, the **page with
the digitised trace drawn over it** (pinch to zoom), and the original photo,
with export and delete. Health data stays in app-private storage.

---

## 5. Results now

### 5.1 Headline table

Median of each record's 12 leads, learned tracer, grid fit on, automatic
corners:

| record | on the phone (A30s)* | phone's method replayed on laptop (latest) | shape if timing forgiven (aligned) | SNR / shifted SNR | R-peak height |
|---|---|---|---|---|---|
| PTB-XL 9 | 0.910 | 0.915 | 0.99 | ~8 / ~15 dB | 0.96 |
| PTB-XL 38 | 0.921 | 0.955 | 0.99 | ~8–11 / ~13–14 dB | 0.91 |
| PTB-XL 57 | 0.908 | 0.927 | 0.99 | ~8 / ~13 dB | 0.87 |
| PTB-XL 40 (6 photos) | — | 0.933–0.975 | 0.99 | ~9–13 / ~15–18 dB | 0.93 |

\*Phone runs were made before the one-pixel fix (§4.8); re-running them is
the next step. Sources: `results/device/sheet_scores.csv`,
`results/ml/phone_path_b3_512x1024_fp16_grid.csv`.

**Classical tracer on the same photos (the control):** PCC 0.46–0.87,
R-peaks 0.15–0.83 — far behind.

### 5.2 What the numbers mean in plain words

- **The network reads the waveform almost perfectly on every record** — 0.99
  once small timing slips are forgiven — including records it never saw.
  That is the generalisation evidence.
- **The phone gives the same answers as the laptop** (within ~0.01 PCC on
  the same method), so phone results are real results, not approximations.
- **What remains is timing, not reading:** plain SNR is about 5–6 dB lower
  than shifted SNR — the leads are the right shape but a few milliseconds
  early or late.
- **Spike heights are preserved** (0.87–0.96), where the classical method
  flattened them to ~0.70 or worse.

### 5.3 On-device cost (A30s)

| stage | time |
|---|---|
| straighten + grid fit | ~2–3 s |
| find rows, pulses, ticks | ~0.3 s |
| neural network (12 tiles) | ~19–20 s |
| cut into leads + score | ~7 s (slow code, fixable) |
| **total per sheet** | **~30 s** |

Model file 0.82 MB. The accuracy-over-speed decision (2026-10-04) made speed
a reported figure, not a target. **Memory has not been measured yet.**

---

## 6. What did not work (keep these in the report)

Negative results show the method was tested, not guessed.

| idea | why it seemed good | what happened |
|---|---|---|
| Pin each row's end to exactly 10 s | each row is exactly 10 s long | the "end of ink" is unreliable; 14 leads got worse, 5 better, 24 of 78 couldn't be measured. Rejected. |
| Use only the grid scale for time | the grid is a perfect ruler | revealed time zero sits early; worse overall. Rejected. |
| Take time zero from the ticks | ticks are more reliable than the pulse | mixed: 30 leads worse, 38 better. Rejected. |
| Pick the pulse as "the first ink of the right length" | skip grid fragments | grid fragments can be the right length too. Rejected. |
| Measure timing error via the registration | direct and tracer-free | not precise enough at millisecond level (lens distortion). Kept as a negative result. |

Tick and pulse quirks found along the way:
- **The ninth tick:** usually the top-right one (before V4) is missed —
  that part of the sheet prints lighter (printer fading). Harmless: it
  borrows the position from the other rows (~6 ms).
- **Automatic top corners** sit 5–7 mm off on the new photos; the grid fit
  makes this irrelevant to the result.

---

## 7. Limitations (state these honestly)

1. **Four records**, and records 9, 38 and 57 have **one photo each**, all in
   good light. More records and harder conditions (shadows, angles, glare)
   are needed before any general claim.
2. **Standard 3×4 + rhythm layout only**; other layouts are out of scope.
3. **Our own prints** (ECG-Image-Kit, office printer), not real hospital
   printouts.
4. **Timing is the weak link:** calibration pulses are sometimes missed on
   darker rows, and photos add a few ms of marker error that is not yet
   explained.
5. **Speed ~30 s per sheet**, memory not yet measured.
6. **Not a medical device.** No clinical validation; results are for
   research.

---

## 8. What is next (to the deadline, ~end of October)

1. Re-run PTB-XL 9, 38, 57 on the phone with the latest app (one-pixel fix).
2. Photos in harder conditions (side light, shadow, tilt).
3. Quality warnings on the phone when the result is likely wrong (too few
   ticks, no pulse, grid fit refused).
4. Speed up "cut + score" (~7 s → < 1 s); measure memory on the A30s.
5. Robust pulse detection (the remaining timing error).
6. Code freeze ~17 October, regenerate every result from one version, then
   write-up.

---

## 9. Suggested report structure (mapping to the above)

- **Introduction / problem:** §1, §2.1.
- **Background:** ECG basics (§2.1), PTB-XL and ECG-Image-Kit (§2.2),
  metrics (§2.3), related digitisation work.
- **Method:** pipeline (§3), classical tracer (§4.1), learned tracer and
  training (§4.2), whole-sheet segmentation and ticks (§4.3–4.4), geometry and
  grid fit (§4.5–4.6), on-device implementation and testing against the
  Python reference ("golden vectors": answer keys checking the phone's code
  gives the laptop's answers).
- **Results:** §5 tables; the effect of each fix (§4.6–4.8 before/after);
  classical vs learned; phone vs laptop; time per stage.
- **Discussion:** what failed (§6), why timing dominates, limitations (§7).
- **Conclusion and future work:** §8.

Evidence trail for any number: `docs/progress_log.md` gives the commit and
results file behind it.
