package com.maarijalam.eacg

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** Diagnosis: flags at the thresholds, and the CSV header round trip. */
class DiagnosisTest {

    private val classes = listOf("NORM", "MI", "STTC", "CD", "HYP")
    private val d = Diagnosis(classes, doubleArrayOf(0.30, 0.35, 0.80, 0.04, 0.10),
        doubleArrayOf(0.47, 0.35, 0.40, 0.05, 0.48), "clf_superclass_fp16.tflite (run)")

    @Test fun flaggedAtOrAboveTheThreshold() {
        assertEquals(listOf("MI", "STTC"), d.flags)          // MI exactly at 0.35 counts
        assertEquals(listOf("MI", "STTC"), d.abnormal)
    }

    @Test fun normalIsNotAbnormal() {
        val n = Diagnosis(classes, doubleArrayOf(0.9, 0.1, 0.1, 0.01, 0.1), d.thresholds, "")
        assertEquals(listOf("NORM"), n.flags)
        assertEquals(emptyList<String>(), n.abnormal)
    }

    @Test fun csvRoundTrip() {
        val csv = "# generated_by: EaCG SheetPipeline\n" +
            "# diagnosis: ${d.probLine()}\n" +
            "# diagnosis_thresholds: ${d.thresholdLine()}\n" +
            "# diagnosis_model: ${d.model}\n" +
            "time_s,I\n0.000,0.1\n"
        val back = Diagnosis.fromCsv(csv)!!
        assertEquals(classes, back.classes)
        assertArrayEquals(d.probs, back.probs, 1e-4)
        assertArrayEquals(d.thresholds, back.thresholds, 1e-9)
        assertEquals(d.model, back.model)
        assertEquals(d.flags, back.flags)
    }

    @Test fun absentOrBrokenIsNull() {
        assertNull(Diagnosis.fromCsv("# generated_by: EaCG\ntime_s,I\n"))
        assertNull(Diagnosis.parse("NORM=0.5 MI=0.2", "NORM=0.5", "m"))   // a threshold missing
        assertNull(Diagnosis.parse("", "NORM=0.5", "m"))
    }
}
