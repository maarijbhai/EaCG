package com.maarijalam.eacg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

/**
 * Where an image comes from. The pipeline never knows which one it got.
 * Today: assets and the file picker. Later: CameraSource, unchanged elsewhere.
 */
interface ImageSource {
    suspend fun acquire(): Bitmap
}

/**
 * Bundled test images. Fixed inputs, useful for regression checks.
 *
 * Decoded deliberately unscaled: a [ReferenceImage]'s known px/mm is a property
 * of its authored pixel dimensions, so any density scaling or subsampling here
 * would silently invalidate the ground truth the whole fixture set exists to
 * provide. Assets are not density-scaled the way `res/drawable` is, but
 * `inScaled = false` states the requirement rather than relying on it.
 *
 * The cost is honest: the 300 dpi fixture is 3300x2550, which is ~34 MB as
 * ARGB_8888. That is fine on the bench and is the reason the picker path
 * downsamples and this one must not.
 */
class AssetSource(
    private val context: Context,
    private val path: String
) : ImageSource {
    override suspend fun acquire(): Bitmap = withContext(Dispatchers.IO) {
        val opts = BitmapFactory.Options().apply {
            inScaled = false
            inSampleSize = 1
        }
        context.assets.open(path).use {
            requireNotNull(BitmapFactory.decodeStream(it, null, opts))
        }
    }
}

/** An image the user picked from storage. */
class UriSource(
    private val context: Context,
    private val uri: Uri,
    private val maxDimension: Int = 2560
) : ImageSource {
    override suspend fun acquire(): Bitmap = withContext(Dispatchers.IO) {
        decodeDownsampled(context, uri, maxDimension)
    }
}

/**
 * Decode with downsampling, to exactly [maxDim] on the long side, upright.
 *
 * A 25 MP photo as ARGB_8888 is roughly 100 MB, against a per-app heap of
 * about 200 MB on a 3 GB device. Two of those live at once and the app dies.
 *
 * TWO FAULTS FIXED HERE (2026-09-24), both of which would have gone straight
 * into the device results:
 *
 *  - **The size was not what this comment claimed.** `inSampleSize` only
 *    halves, so the old loop took the first power of two that fitted: a
 *    24.8 MP A30s frame (5760 px) came out at 1440 px, not 2560 -- about
 *    4.7 px/mm on a printed A4 sheet, where the desktop sweep put 8 px/mm as
 *    the floor for a usable trace. (The old comment's "2560 px gives ~12
 *    px/mm on A4" was also wrong: 2560 / 297 mm is 8.6.) Now the decoder
 *    halves only while the result stays at or above [maxDim], and an exact
 *    bilinear scale takes it the rest of the way.
 *  - **EXIF orientation was ignored.** A phone stores most photographs
 *    sideways with a rotation tag; `BitmapFactory` does not apply it, so a
 *    landscape sheet arrived portrait. `run_photo.load_oriented` applies it
 *    on the desktop; this now does the same, so a photograph is the same
 *    picture on both.
 */
private fun decodeDownsampled(context: Context, uri: Uri, maxDim: Int): Bitmap {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri).use {
        BitmapFactory.decodeStream(it, null, bounds)
    }

    var sample = 1
    while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxDim) {
        sample *= 2
    }

    val opts = BitmapFactory.Options().apply { inSampleSize = sample }
    var bmp = context.contentResolver.openInputStream(uri).use {
        requireNotNull(BitmapFactory.decodeStream(it, null, opts))
    }

    val longest = maxOf(bmp.width, bmp.height)
    if (longest > maxDim) {
        val scale = maxDim.toDouble() / longest
        val scaled = Bitmap.createScaledBitmap(
            bmp, (bmp.width * scale).roundToInt(), (bmp.height * scale).roundToInt(), true
        )
        if (scaled !== bmp) bmp.recycle()
        bmp = scaled
    }

    val orientation = context.contentResolver.openInputStream(uri).use {
        if (it == null) ExifInterface.ORIENTATION_NORMAL
        else ExifInterface(it).getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL
        )
    }
    val degrees = when (orientation) {
        ExifInterface.ORIENTATION_ROTATE_90 -> 90f
        ExifInterface.ORIENTATION_ROTATE_180 -> 180f
        ExifInterface.ORIENTATION_ROTATE_270 -> 270f
        else -> 0f
    }
    if (degrees != 0f) {
        val m = Matrix().apply { postRotate(degrees) }
        val rotated = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        if (rotated !== bmp) bmp.recycle()
        bmp = rotated
    }
    return bmp
}
