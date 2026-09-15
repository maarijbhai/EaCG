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

**Not measured: anything from a real photograph.** No print has been made and
no photograph taken. Every figure quoted in this document comes from either the
closed-loop renders or from *simulated* photographs — a rendered page put
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
