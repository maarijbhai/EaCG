package com.maarijalam.eacg

import kotlin.math.ceil

/**
 * Summary statistics for a repeated timing, and the conventions this project
 * reports them under.
 *
 * **The first run is reported separately and never averaged in.** On a JVM
 * the first pass through any code path pays for class loading, interpretation
 * before JIT compilation, and cold instruction and data caches; on a phone it
 * also pays for the CPU governor still being at an idle clock. Folding that
 * into a mean produces a number that describes neither the cold case nor the
 * warm one. Both are real costs and they answer different questions — the
 * cold figure is what a user experiences pressing a button on a freshly
 * opened app, the warm one is what the algorithm actually costs — so both are
 * kept and labelled.
 *
 * **The 95th percentile is reported alongside the mean** because a mean alone
 * hides exactly the behaviour that matters on a thermally constrained device.
 * A stage that usually takes 80 ms and occasionally takes 400 has a mean near
 * 90 and is a different proposition from one that always takes 90.
 */
data class Stats(
    /** The cold run, excluded from every other figure here. */
    val first: Long,
    val mean: Double,
    val p95: Long,
    val min: Long,
    val max: Long,
    /** Warm runs only. */
    val n: Int
) {
    fun format(): String =
        if (n == 0) "%d / -- / --".format(first)
        else "%d / %.0f / %d".format(first, mean, p95)
}

/**
 * Turn a series of repeated measurements into [Stats].
 *
 * [samples] must be in run order: `samples[0]` is the cold run and is
 * excluded from the mean, percentile and range.
 *
 * The percentile is nearest-rank — the `ceil(0.95 * n)`-th smallest warm
 * sample — rather than an interpolated one. With the small `n` a benchmark
 * on a phone can afford, interpolating between two adjacent samples invents
 * precision the sample size does not support; nearest-rank at least names an
 * observation that actually happened. Note that at n = 10 the 95th
 * percentile *is* the maximum, so the two columns carry the same value and
 * the honest reading is "worst observed".
 */
fun summarise(samples: List<Long>): Stats {
    if (samples.isEmpty()) return Stats(0, 0.0, 0, 0, 0, 0)
    val first = samples[0]
    val warm = samples.drop(1)
    if (warm.isEmpty()) return Stats(first, 0.0, 0, 0, 0, 0)
    val sorted = warm.sorted()
    val rank = (ceil(0.95 * sorted.size).toInt() - 1).coerceIn(0, sorted.size - 1)
    return Stats(
        first = first,
        mean = warm.average(),
        p95 = sorted[rank],
        min = sorted.first(),
        max = sorted.last(),
        n = warm.size
    )
}
