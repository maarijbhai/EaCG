# Photograph protocol

What a human must do to turn a print target into a scoreable photograph.

This is the first measurement in the project that says anything about field
performance. Everything before it came from a closed loop: the image was drawn
by the same code, with the same constants, that later decoded it. Grid pitch
came back at +0.00% error because the renderer drew rules at exactly that
pitch. That is a tautology, not a result. This procedure breaks the loop.

Follow it exactly. Most of the steps below exist because a specific thing goes
wrong without them, and each one says which.

---

## 0. Before you start

You need: a colour printer, plain matte white A4 paper, a ruler with
millimetre markings, a phone camera, a flat surface, and even indirect light.

Generate the target:

```
python python/make_print_target.py --ecg-id 9 --lead II
```

That writes, into `out/print_targets/<name>/`:

| file | what it is |
|---|---|
| `<name>.pdf` | the page to print, at true physical scale |
| `<name>_truth.npy` | the source waveform in mV — the ground truth |
| `<name>_manifest.json` | identity, calibration, geometry, git commit |

Keep all three together. The `.npy` is what the photograph is scored against;
without it the photograph is unmeasurable.

---

## 1. Print

**Scale must be 100%. This is the single thing most likely to ruin the run.**

In the print dialog:

- Set **Scale: 100%** or **Actual size**.
- Turn **OFF** "Fit to page", "Shrink oversized pages", "Fit to printable area".
  These are **on by default** in Adobe Reader, Chrome's PDF viewer, Preview and
  most Windows print paths, and they silently scale the page by a few percent
  to clear the printer's unprintable margin.
- Paper: **A4**, matching the page size in the PDF.
- Colour: **colour, not greyscale, not "black and white"**, and not "draft" or
  "toner save".
- Orientation: **landscape**.

Why colour is not optional: `digitise.binarise_trace` separates grid from trace
by red-channel arithmetic — white paper and the red grid both keep a saturated
red channel, the near-black trace does not. Printed in greyscale, the grid
becomes neutral and dark and binarisation returns **the grid** as trace. The
audit measured this: on a neutral grid, pitch estimation returns NaN on one
axis and the pipeline raises.

Paper: plain matte. **Not glossy or photo paper** — it produces specular
highlights under any directional light, and §3 explains why that matters more
than it sounds.

---

## 2. Verify the print — do not skip this

Measure the printed reference bars with a real ruler:

| bar | must measure | tolerance |
|---|---|---|
| horizontal, labelled `100.0 mm horizontal` | 100 mm | ±0.5 mm |
| vertical, labelled `50 mm vertical` | 50 mm | ±0.5 mm |

Measure between the **outer edges of the end caps**.

If either is wrong, the printer scaled the page. **Reprint — do not photograph
it, and do not try to correct for it afterwards.** The whole experiment rests
on the grid being a physical ruler of known pitch; a scaled print is a ruler of
unknown length, and nothing in the image reveals the factor.

If the two bars are wrong by *different* amounts, the printer scaled the axes
independently. Amplitude and time would then be wrong by different factors.
Reprint.

Record both measured values in your notes. If you later reprint on a different
printer, they are how you know the two runs are comparable.

---

## 3. Set up the shot

**Surface.** Flat, matte, non-white (a wooden desk is ideal — it makes the page
edge easy to see). Tape or weight the corners so the paper lies genuinely flat.
A curled page is not a perspective distortion and the four-corner rectification
cannot undo it.

**Lighting. This is the step that surprised us, so read it.**

Use **even, diffuse, indirect light** — daylight from a window with the page
out of the direct beam, or two lamps at 45° from opposite sides, or an
overcast-daylight ceiling light. Avoid:

- a single desk lamp to one side,
- direct sunlight,
- the camera flash,
- your own shadow, or the phone's shadow, falling across the page.

In simulation, an uneven illumination gradient across the page — on its own,
harmless — **combined** with ordinary lens blur and JPEG compression caused
`estimate_grid_pitch` to lock onto the 5 mm major rules instead of the 1 mm
minor ones on the vertical axis. The result was silent and it looked excellent:

```
PCC 0.994          the shape was recovered perfectly
gain 0.238         every amplitude was wrong by a factor of ~5
```

Nothing raised. `digitise` accepts the pitch because it is finite and positive.
A report that quoted PCC alone would have called this a good reconstruction.
Even lighting is the cheapest defence available.

(That figure comes from a simulated photograph, not a real one — see §7.)

---

## 4. Take the photographs

For each shot:

- **Highest resolution** the camera offers. Turn off any "storage saver" mode.
- **HDR off**, **flash off**, **portrait/beauty modes off**. HDR composites
  multiple exposures and can produce local contrast changes that behave like
  the illumination gradient in §3.
- Hold the phone **parallel to the page** and **centred over the strip**,
  unless the shot is deliberately oblique.
- Frame so the **gridded strip and all four crop marks are inside the frame**,
  filling as much of it as possible. The rulers below may be cut off — they
  have done their job by now.
- **Tap to focus on the grid**, not on the trace, and wait for focus to settle.
- Take the shot without moving. Check it is sharp at full zoom before moving
  on; a blurred grid cannot be recovered later.

### The minimum set — 9 photographs

The point is to separate what the camera costs from what the *angle* costs from
what the *light* costs. Shoot all nine in one session, on the same print.

| # | condition | what it measures |
|---|---|---|
| 1–3 | square on, even light, identical setup, three times | run-to-run variance — the noise floor of every other number here |
| 4 | rotated ~5° in the page plane | mild skew; renders showed 1° already costs PCC 0.98 → 0.73 |
| 5 | rotated ~15° in the page plane | skew the four-corner warp must undo |
| 6 | tilted ~15° out of plane | mild perspective |
| 7 | tilted ~30° out of plane | perspective at the edge of usability |
| 8 | square on, single directional lamp | the §3 failure, deliberately provoked |
| 9 | square on, dim indoor light, handheld | motion blur and sensor noise together |

Shots 1–3 come first and matter most: without a repeat measurement there is no
way to tell a real effect from capture noise.

**Transfer the files at full resolution.** Do not send them through WhatsApp,
Messenger, Slack or iMessage — all of them recompress, and several downscale.
Use a cable, AirDrop, or email with "actual size" selected.

---

## 5. Score

```
python python/score_photo.py \
    --photo photo_01.jpg \
    --manifest out/print_targets/<name>/<name>_manifest.json \
    --pick --save-corners --diagnostics out/photo_runs/01
```

`--pick` opens a window; click the four strip corners in the order
**top-left, top-right, bottom-right, bottom-left**. Click the **inner vertex of
each printed crop mark** — that vertex sits exactly on the strip corner, which
"the corner of the grid" does not at photograph resolution. Press `u` to undo,
Enter when four are placed.

`--save-corners` writes them into the manifest so the run is reproducible
without clicking again. `--diagnostics` writes the rectified strip, the
binarisation mask, and the recovered trace overlaid on the strip — look at the
overlay before believing any number.

If the corner picker will not open (no GUI), pass them directly:

```
--corners "529,557 3397,690 3382,1475 471,1342"
```

---

## 6. Reading the output

`score_photo.py` reports each stage separately, because a single SNR cannot
tell a calibration failure from a tracing failure. In rough order of what to
check:

1. **`keystone`** near 1.0 means you were square on. Far from 1.0 is fine for
   shots 6–7; it is a mistake in shots 1–3.
2. **`TRUE px/mm`** against the estimate. The script flags a harmonic lock
   explicitly. A 2× or 5× lock is the failure from §3 and means every
   amplitude is wrong by that factor even when PCC looks perfect.
3. **`anisotropy`** — large means the two axes disagree, which is either real
   perspective or a lock on one axis. The script prints it; `digitise` ignores
   it (audit finding B1).
4. **NaN in the output** — the script flags it. NaN propagates into every
   metric and turns a median into `nan` (audit finding B2).
5. **PCC then gain, never PCC alone.** High PCC with gain far from 1.0 is a
   calibration fault, not a tracing one.

Compare against the closed-loop bound: renders at 8–12 px/mm give gain
0.86–0.93 and PCC ~0.97–0.99. **The gap between that and what a photograph
gives is the number this whole exercise exists to produce.**

---

## 7. What has and has not been measured

**Superseded in part (2026-09-24).** Six real photographs now exist, of a
print of ecg-image-kit's `ecg00001.png` -- a page with no waveform ground truth
(CLAUDE.md limitation 15, `results/photo/`). What follows was written before
them and still holds for the single-strip targets: no print of a
`make_print_target.py` page has been photographed. For the full-sheet route
with ground truth, see §8.

**Not measured, when this was written: anything from a real photograph.** No
print had been made and no photograph taken. Every figure quoted in this
document comes from either the closed-loop renders or from *simulated*
photographs — a rendered page put
through a synthetic perspective warp, blur, illumination gradient, sensor noise
and JPEG compression.

A simulation has no ink bleed, no paper texture, no fibre, no thermal fade, no
specular highlight, no rolling shutter, no chromatic aberration, and its corner
coordinates are exact rather than clicked by a human. It is strictly easier
than the real thing. It was built only to prove the tooling runs end to end
before anyone spends time printing.

What the simulation did establish is that the *failure modes are reachable*:
the 5× harmonic lock in §3 is real behaviour of `estimate_grid_pitch` on a
degraded image, and it is silent. Whether a real camera provokes it, and at
what lighting, is exactly what shots 1–9 are for.

---

## 8. Full 3x4 sheets with ground truth — the route to use now

`make_kit_target.py` renders a PTB-XL fold-10 record as an ecg-image-kit
clinical sheet (3x4 plus a full-width lead II rhythm strip, 1 mV pulses, lead
names) at true scale, and keeps the record's waveform and every plotted
pixel beside it. A photograph of that print is the first image in this
project with **both** a real camera **and** a waveform to score against.
Four are generated: PTB-XL 9, 38, 40 and 57, in `out/kit_targets/`.

What differs from §0–§7, and why:

- **No crop marks.** The sheet's own grid runs to its edge; the four corners
  are where the pink grid meets white paper. Leave **at least 1 cm of white
  paper visible on every side** in the photograph: `page.refine_corners`
  finds each edge as the place where the grid rules crossing it stop, and it
  needs bare paper beyond the edge to see that.
- **Rough clicks are fine -- within limits.** The refinement moves each click
  onto the edge to within half a pixel on simulated photographs, from up to
  10 px away, but it only searches ±1.5% of each edge's length either side
  of the line you give it: about 30-40 px on a phone-sized decode, less
  than half a 5 mm square. The first real phone attempt (2026-09-28) was
  tapped 92 px out, outside that window, and every number after it was
  wrong; the app now uses draggable corners with a magnifier (§8.4).
- **The 1 mm rules are redrawn darker.** ecg-image-kit's red palette draws
  them in RGB (254,223,219), which an office printer drops entirely: the
  first photographed sheet showed only the 5 mm rules, and the grid
  estimator -- correctly, for that paper -- read a 5 mm pitch.
  `make_kit_target.py` now redraws them in `MINOR_GRID_RGB` (242,160,160),
  as clinical paper prints them; geometry and truth are unchanged. **Sheets
  printed before 2026-09-28 have no usable 1 mm grid: reprint them.**
- **Scoring is automatic past the corners.** The rhythm strip is located, the
  printed text and calibration pulse are cut away, and three calibrations
  are scored side by side (`grid`, `comb`, and the 10 s `duration` anchor).

### 8.1 Print and verify

1. Open each `out/kit_targets/<name>/<name>.pdf`. **Two pages: print both,
   one job**, colour, A4, landscape, **Actual size / 100%** — the rules of §1
   all apply.
2. On page 2, measure the **100 mm bar** with a ruler (±0.5 mm). If it is
   wrong, the job was scaled: reprint, do not photograph. Write the measured
   value on the back of page 1.
3. On page 1, check by eye that **the 1 mm rules are visible** inside every
   5 mm square. If they are not, the printer dropped them (see above) and
   the sheet cannot test grid calibration.

### 8.2 Photograph — on the Galaxy A30s

Same set-up as §3 and §4 (flat, matte surface, even diffuse light, HDR and
flash off, focus on the grid). One sheet at a time, on a surface that
contrasts with white paper -- not on the stack of other printouts -- with
**white paper visible beyond the grid on all four sides**. The first attempt
ran the grid out of frame at the bottom-left, and the refinement cannot find
an edge it cannot see beyond. Per sheet:

| # | shot | why |
|---|---|---|
| 1–3 | square on, whole sheet plus the white border, three times | the repeatability floor for this sheet |
| 4 | tilted ~15° out of plane | mild perspective, what a hand-held capture really gets |
| 5 | (optional) single side lamp | the lighting failure of §3 |

Four sheets x 3–5 shots is 12–20 photographs, about fifteen minutes.

### 8.3 Score on the desktop

1. Copy the originals (cable, not a messenger app) into `data/kit_photos/`.
2. Register each once, clicking the four sheet corners
   (top-left, top-right, bottom-right, bottom-left):

   ```
   cd python
   python score_kit_photo.py --add 20261001_101500.jpg --target ptbxl9_kit_500hz_300dpi --pick
   ```

   The corners go into `python/kit_photos.json`, which is committed; the
   photographs are not.
3. Score everything registered:

   ```
   python score_kit_photo.py
   ```

   writes `results/kitphoto/score.csv` and a sidecar, plus overlays in
   `out/kitphoto/<photo>/` — look at `known_overlay.png` (blue: where the kit
   plotted the trace; red: where the tracer found it) before believing a
   number.

### 8.4 Score the same photographs on the phone

The photographs are already in the phone's gallery. In the app, **Photograph
of a printed sheet**: pick the photograph and choose its PTB-XL number.
Four blue circles appear; **drag each onto its corner of the printed grid**
(where the pink grid meets white paper). While you drag, a magnifier drawn
from the full-resolution photograph shows exactly where the corner sits --
line its crosshair up with the corner, within half a 5 mm square. A red
warning appears if a corner is at the edge of the photograph: retake it.
Then **Score photograph**; the refined sheet outline is drawn in green, and
an edge-fit warning appears if the fit is worse than 3 px. Then **Export
CSV**. The CSV lands in `Downloads/EaCG/` with the corners, the per-stage
timings and each calibration's waveform and scores in its header; copy it
to `data/device_exports/` and run `python ingest_device_exports.py`. The
phone decodes to 3200 px on the long side (about 10 px/mm on this sheet), so
its numbers are for that resolution and the desktop's are for the full
frame — compare like with like.

### 8.5 Reading the rows

- `mode=known` hands the tracer the true scale and crop: it measures FR5–FR6
  on real ink. `trace_rmse_mm` is the tracer's position error in millimetres
  against the kit's own plotted pixels — no calibration involved at all.
- `mode=pipeline` is everything automatic past the corners. Read
  `pitch_error_pct` beside `pcc_asis`: limitation 14 says a 0.5% scale error
  alone costs a third of PCC.
- `corner_shift_px_max` large (tens of px) means a click was far off or a
  corner is not where you think; `edge_fit_rmse_px_max` above ~2 px means an
  edge is not straight — a curled page.

