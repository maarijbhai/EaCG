package com.maarijalam.eacg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

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
 * Decode with downsampling.
 *
 * A 25 MP photo as ARGB_8888 is roughly 100 MB, against a per-app heap of
 * about 200 MB on a 3 GB device. Two of those live at once and the app dies.
 * 2560 px across gives ~12 px/mm on an A4 page, comfortably above the 8 px/mm
 * the Python sweep showed was workable.
 */
private fun decodeDownsampled(context: Context, uri: Uri, maxDim: Int): Bitmap {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri).use {
        BitmapFactory.decodeStream(it, null, bounds)
    }

    var sample = 1
    while (bounds.outWidth / sample > maxDim || bounds.outHeight / sample > maxDim) {
        sample *= 2
    }

    val opts = BitmapFactory.Options().apply { inSampleSize = sample }
    return context.contentResolver.openInputStream(uri).use {
        requireNotNull(BitmapFactory.decodeStream(it, null, opts))
    }
}