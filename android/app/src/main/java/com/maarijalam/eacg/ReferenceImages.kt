package com.maarijalam.eacg

/**
 * What is actually printed on the page, which decides which stages may run.
 *
 * Not cosmetic. [SHEET_3X4] is valid input for FR3 and invalid for FR5 to
 * FR7: the tracer takes the ink centroid of a whole column, so a column
 * crossing four stacked leads returns their average — a waveform belonging
 * to no lead. Measured on the desktop at PCC 0.273, with pitch correct to
 * 0.00% and every calibration statistic clean, which is exactly why
 * [QualityGate.tracesPerColumn] exists and why it must run before any
 * digitised curve is drawn on a screen.
 */
enum class PrintoutLayout { SHEET_3X4, SINGLE_STRIP }

/**
 * Bundled printouts whose true grid scale is known exactly, rather than
 * estimated.
 *
 * Why these exist: the file picker can open any image, but nothing in the app
 * knows what the *correct* answer is for a photo off the camera roll. That
 * makes "does the Kotlin grid estimator agree with the Python one?" an
 * unanswerable question — a disagreement could be either implementation. A
 * fixed input with a known answer turns it into a decidable one.
 *
 * Where pxPerMm comes from — three independent agreeing sources:
 *
 *  1. By construction. These were rendered by ecg-image-kit from PTB-XL
 *     waveforms at a commanded raster resolution (`-r`). The renderer lays the
 *     grid out in physical units — `x_grid_inch = 5/25.4`, i.e. a 5 mm major
 *     box — and rasterises at `resolution` dots per inch. So the scale is
 *     dpi / 25.4, exactly, and is a property of the render command rather than
 *     anything recovered from the image.
 *  2. The generator's own sidecar JSON (`--store_config 2`) records
 *     `x_grid`/`y_grid` = pixels per 5 mm major box: 29.528, 39.370, 59.055
 *     for 150/200/300 dpi. Divided by 5 these reproduce pxPerMm.
 *  3. Measured back out of the pixels. Locating the major grid lines and
 *     regressing position against index over 45 lines gives 5.90572, 7.87420
 *     and 11.81146 px/mm — agreeing with (1) to within 0.004%, with sub-pixel
 *     fit residual.
 *
 * Note this is legitimate *only* because these are rendered, not photographed.
 * CLAUDE.md's rule that calibration cannot come from DPI holds for camera
 * capture, where there is no fixed correspondence between an image pixel and a
 * physical millimetre. Here the image never existed as paper: the renderer was
 * told how many pixels to put in an inch and did exactly that. That is why
 * these are reference fixtures and not a calibration strategy.
 *
 * The scale therefore holds only as long as the bitmap reaches the pipeline at
 * its authored pixel dimensions. AssetSource decodes unscaled and unsampled
 * for this reason. Downsampling an asset silently divides the true answer by
 * the sample factor.
 */
data class ReferenceImage(
    /** Path under `assets/`. */
    val assetPath: String,
    /** Short label for the button face. */
    val label: String,
    /** PTB-XL ecg_id this was rendered from. */
    val record: String,
    /** PTB-XL diagnostic superclass, for waveform variety across the set. */
    val superclass: String,
    /** Raster resolution the printout was rendered at, in dots per inch. */
    val renderDpi: Int,
    val widthPx: Int,
    val heightPx: Int,
    /** What is on the page. See [PrintoutLayout]. */
    val layout: PrintoutLayout = PrintoutLayout.SHEET_3X4,
    /**
     * Asset holding the source waveform this was rendered from, one float per
     * line in millivolts, or null when there is none.
     *
     * Present only for the single strips. It is what lets the handset score
     * its own reconstruction — the same move the rotation slider makes for
     * deskew, and for the same reason: a fidelity number carried over from
     * the desktop cannot detect a porting bug, which is the whole risk the
     * Kotlin side carries.
     */
    val truthAsset: String? = null,
    /** Sampling rate of [truthAsset], Hz. */
    val truthFs: Double = 100.0,
    /** Duration of the printed recording, seconds. The paper-speed anchor. */
    val durationS: Double? = null
) {
    /** Exact by construction. See the class docstring. */
    val pxPerMm: Double get() = renderDpi / 25.4

    /** Pixels per 5 mm major box — what the sidecar JSON records directly. */
    val pxPerMajorBox: Double get() = pxPerMm * 5.0

    /**
     * Pixels per sample-second along the time axis, at the conventional
     * 25 mm/s paper speed. Useful once the tracer needs to convert x to time.
     */
    val pxPerSecond: Double get() = pxPerMm * 25.0
}

/**
 * Three scales spanning the plausible range, so the set actually discriminates.
 *
 * Deliberately *not* three images at one scale: a grid estimator that ignored
 * its input and returned a hardcoded 7.874 would pass that set completely. The
 * 150/200/300 dpi spread means only an estimator that genuinely reads the image
 * can match all three. They also carry three different diagnostic superclasses,
 * so the traces differ in morphology and amplitude, not just in scale.
 */
val REFERENCE_IMAGES = listOf(
    ReferenceImage(
        assetPath = "reference/gt_norm_150dpi.png",
        label = "150 dpi",
        record = "00009_hr",
        superclass = "NORM",
        renderDpi = 150,
        widthPx = 1650,
        heightPx = 1275
    ),
    ReferenceImage(
        assetPath = "reference/gt_mi_200dpi.png",
        label = "200 dpi",
        record = "00063_hr",
        superclass = "MI",
        renderDpi = 200,
        widthPx = 2200,
        heightPx = 1700
    ),
    ReferenceImage(
        assetPath = "reference/gt_sttc_300dpi.png",
        label = "300 dpi",
        record = "00116_hr",
        superclass = "STTC",
        renderDpi = 300,
        widthPx = 3300,
        heightPx = 2550
    )
)

/**
 * The single strips: one lead, one continuous trace, at the same three
 * scales. Generated by `python/make_strip_fixtures.py`.
 *
 * These are the fixtures FR5 to FR7 are *allowed* to run on, and the set
 * exists because without it the tracer's only on-device inputs would be the
 * three sheets above — which it would digitise into a plausible-looking
 * splice of four leads and report nothing wrong about. The sheets remain the
 * right fixtures for FR3, which measures the grid and does not care how many
 * leads share the page.
 *
 * All three carry the SAME waveform at three render scales, deliberately: the
 * reconstruction error then varies only with pitch, so a difference between
 * them is a resolution effect rather than a difference in the signal. The
 * desktop measured the classical tracer under-reading amplitude by gain ratio
 * 0.60 at 4 px/mm against 0.96 at 16, so that dependence is the thing to see.
 *
 * `durationS` is 10 s, and supplying it is what lets the gate's paper-speed
 * check run at all. A photograph of someone else's ECG comes with nobody to
 * state it, and the gate then reports that check unavailable rather than
 * faking it — the cost of which the desktop measured: a 5x pitch error at
 * PCC 0.005 can then only be flagged, never rejected.
 */
val STRIP_IMAGES = listOf(
    ReferenceImage(
        assetPath = "reference/strip_ii_150dpi.png",
        label = "II 150",
        record = "9",
        superclass = "NORM",
        renderDpi = 150,
        widthPx = 1535,
        heightPx = 413,
        layout = PrintoutLayout.SINGLE_STRIP,
        truthAsset = "reference/strip_ii_truth.csv",
        truthFs = 100.0,
        durationS = 10.0
    ),
    ReferenceImage(
        assetPath = "reference/strip_ii_200dpi.png",
        label = "II 200",
        record = "9",
        superclass = "NORM",
        renderDpi = 200,
        widthPx = 2047,
        heightPx = 551,
        layout = PrintoutLayout.SINGLE_STRIP,
        truthAsset = "reference/strip_ii_truth.csv",
        truthFs = 100.0,
        durationS = 10.0
    ),
    ReferenceImage(
        assetPath = "reference/strip_ii_300dpi.png",
        label = "II 300",
        record = "9",
        superclass = "NORM",
        renderDpi = 300,
        widthPx = 3071,
        heightPx = 827,
        layout = PrintoutLayout.SINGLE_STRIP,
        truthAsset = "reference/strip_ii_truth.csv",
        truthFs = 100.0,
        durationS = 10.0
    )
)
