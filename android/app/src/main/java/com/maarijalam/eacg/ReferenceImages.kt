package com.maarijalam.eacg

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
    val heightPx: Int
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
