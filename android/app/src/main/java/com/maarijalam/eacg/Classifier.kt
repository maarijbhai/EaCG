package com.maarijalam.eacg

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * The diagnosis classifier on the phone: LiteRT FP16, CPU, one sheet at a
 * time. A research classifier trained on PTB-XL (folds 1-8) to the five
 * diagnostic superclasses from what a 3x4 printout shows
 * (`ml/train_classifier.py`, view "printed"; exported by
 * `ml/export_classifier.py`, which also chose the thresholds on fold 9).
 * Input "ecg" (1, 1000, 13) from [ClfInput]; output "prob" (1, 5), the
 * sigmoid already in the graph. 1.7 MB.
 *
 * It was trained on true signals, not digitised ones; how far a digitised
 * sheet moves its output is what `ml/clf_degradation.py` measures.
 */
class Classifier(context: Context, threads: Int = 2) : AutoCloseable {

    class Meta(val classes: List<String>, val thresholds: DoubleArray, val run: String,
               val valMacroAuroc: Double, val testMacroAuroc: Double) {
        val label: String get() = MODEL_ASSET.substringAfterLast('/') + " ($run)"
    }

    val meta: Meta = loadMeta(context)
    private val interp: Interpreter
    private val inName: String
    private val input = ByteBuffer.allocateDirect(ClfInput.N * ClfInput.CHANNELS * 4).order(ByteOrder.nativeOrder())
    private val output = ByteBuffer.allocateDirect(meta.classes.size * 4).order(ByteOrder.nativeOrder())

    init {
        val buf = context.assets.openFd(MODEL_ASSET).use { fd ->
            FileInputStream(fd.fileDescriptor).use {
                it.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
        interp = Interpreter(buf, Interpreter.Options().setNumThreads(threads))
        val key = interp.signatureKeys[0]
        inName = interp.getSignatureInputs(key).first()
    }

    /** Probabilities for [ClfInput]'s 1,000 x 13 input. */
    fun predict(x: FloatArray): DoubleArray {
        require(x.size == ClfInput.N * ClfInput.CHANNELS)
        input.rewind(); input.asFloatBuffer().put(x)
        output.rewind()
        interp.runSignature(mapOf(inName to input), mapOf("prob" to output))
        val p = FloatArray(meta.classes.size)
        (output.rewind() as ByteBuffer).asFloatBuffer().get(p)
        return DoubleArray(p.size) { p[it].toDouble() }
    }

    fun diagnose(leads: List<SheetPipeline.LeadResult>): Diagnosis =
        Diagnosis(meta.classes, predict(ClfInput.fromLeads(leads)), meta.thresholds, meta.label)

    override fun close() = interp.close()

    companion object {
        const val MODEL_ASSET = "ml/clf_superclass_fp16.tflite"
        const val META_ASSET = "ml/clf_superclass.json"
        const val KIT_LABELS_ASSET = "ml/clf_kit_labels.txt"

        fun loadMeta(context: Context): Meta {
            val j = JSONObject(context.assets.open(META_ASSET).bufferedReader().use { it.readText() })
            val cls = j.getJSONArray("classes").let { a -> List(a.length()) { a.getString(it) } }
            val thr = j.getJSONObject("thresholds")
            return Meta(cls, DoubleArray(cls.size) { thr.getDouble(cls[it]) }, j.getString("run"),
                j.optDouble("val_macro_auroc"), j.optDouble("test_macro_auroc"))
        }

        /** PTB-XL ecg_id -> (superclasses, codes) of the printed test sheets. */
        fun kitLabels(context: Context): Map<Int, Pair<String, String>> =
            context.assets.open(KIT_LABELS_ASSET).bufferedReader().useLines { lines ->
                lines.filter { !it.startsWith("#") && it.isNotBlank() }.mapNotNull { line ->
                    val kv = line.split(' ').mapNotNull { p -> p.split('=', limit = 2).takeIf { it.size == 2 } }
                        .associate { it[0] to it[1] }
                    val id = kv["ecg_id"]?.toIntOrNull() ?: return@mapNotNull null
                    id to ((kv["superclasses"] ?: "") to (kv["codes"] ?: ""))
                }.toMap()
            }
    }
}
