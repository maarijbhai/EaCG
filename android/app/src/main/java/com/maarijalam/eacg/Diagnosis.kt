package com.maarijalam.eacg

import java.util.Locale

/**
 * What the diagnosis classifier said about one sheet: a probability per
 * PTB-XL superclass and the threshold at which the app calls it "flagged".
 * Multi-label: a record can be flagged for several classes at once (sigmoid
 * outputs, never softmax). The thresholds were chosen on the validation fold
 * (`ml/export_classifier.py`). Pure JVM code (DiagnosisTest).
 *
 * In the CSV header (and so in every saved record's leads.csv):
 *
 *     # diagnosis: NORM=0.9123 MI=0.0120 STTC=0.0300 CD=0.0101 HYP=0.0410
 *     # diagnosis_thresholds: NORM=0.47 MI=0.35 STTC=0.40 CD=0.05 HYP=0.48
 *     # diagnosis_model: clf_superclass_fp16.tflite (printed_resnet_aug_s2)
 */
class Diagnosis(
    val classes: List<String>,
    val probs: DoubleArray,
    val thresholds: DoubleArray,
    val model: String,
) {
    init { require(classes.size == probs.size && probs.size == thresholds.size) }

    fun flagged(k: Int): Boolean = probs[k] >= thresholds[k]
    val flags: List<String> get() = classes.filterIndexed { k, _ -> flagged(k) }
    /** Flagged classes other than NORM. */
    val abnormal: List<String> get() = flags.filter { it != "NORM" }

    fun probLine(): String = line(probs, "%.4f")
    fun thresholdLine(): String = line(thresholds, "%.2f")
    private fun line(v: DoubleArray, fmt: String) =
        classes.indices.joinToString(" ") { "${classes[it]}=${String.format(Locale.US, fmt, v[it])}" }

    companion object {
        val NAMES = mapOf(
            "NORM" to "Normal ECG",
            "MI" to "Myocardial infarction",
            "STTC" to "ST/T change",
            "CD" to "Conduction disturbance",
            "HYP" to "Hypertrophy",
        )

        private fun pairs(s: String): List<Pair<String, Double>> =
            s.trim().split(' ').filter { '=' in it }.mapNotNull { p ->
                val (k, v) = p.split('=', limit = 2)
                v.toDoubleOrNull()?.let { k to it }
            }

        /** From the CSV's three header values; null if absent or inconsistent. */
        fun parse(probs: String?, thresholds: String?, model: String?): Diagnosis? {
            val p = pairs(probs ?: return null)
            val t = pairs(thresholds ?: return null).toMap()
            if (p.isEmpty() || p.any { it.first !in t }) return null
            return Diagnosis(p.map { it.first }, p.map { it.second }.toDoubleArray(),
                p.map { t.getValue(it.first) }.toDoubleArray(), model.orEmpty())
        }

        /** From a whole SheetPipeline.csv text (a saved record's leads.csv). */
        fun fromCsv(text: String): Diagnosis? {
            val h = HashMap<String, String>()
            for (line in text.lineSequence()) {
                if (!line.startsWith("#")) break
                val kv = line.removePrefix("#").trim()
                val i = kv.indexOf(':')
                if (i > 0) h[kv.substring(0, i).trim()] = kv.substring(i + 1).trim()
            }
            return parse(h["diagnosis"], h["diagnosis_thresholds"], h["diagnosis_model"])
        }
    }
}
