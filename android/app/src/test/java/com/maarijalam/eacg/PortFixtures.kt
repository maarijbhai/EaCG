package com.maarijalam.eacg

/**
 * Reader for `port_fixtures.txt`, the golden vectors the Python emits.
 *
 * Deliberately parses a flat `key v1 v2 v3` text format with `split`, rather
 * than JSON. A JSON parser would be a test dependency, and `org.json` is
 * stubbed to throw in Android host-side unit tests, so it would have to be
 * a real one. The format is also greppable and diffable: when a value moves,
 * `git diff` shows which one, which matters for a file whose whole purpose
 * is to make a numeric change visible.
 */
class PortFixtures private constructor(private val rows: Map<String, List<String>>) {

    companion object {
        private const val RESOURCE = "/port_fixtures.txt"

        fun load(): PortFixtures {
            val stream = PortFixtures::class.java.getResourceAsStream(RESOURCE)
                ?: error(
                    "$RESOURCE not found on the test classpath. It is generated " +
                        "by `python python/make_port_fixtures.py` and committed " +
                        "under android/app/src/test/resources/."
                )
            val rows = stream.bufferedReader().useLines { lines ->
                lines.map { it.trim() }
                    .filter { it.isNotEmpty() && !it.startsWith("#") }
                    .map { line ->
                        val parts = line.split(" ")
                        parts[0] to parts.drop(1)
                    }
                    .toMap()
            }
            return PortFixtures(rows)
        }
    }

    private fun row(key: String): List<String> =
        rows[key] ?: error("missing key '$key' in $RESOURCE; regenerate it")

    fun int(key: String, index: Int = 0): Int = row(key)[index].toInt()
    fun long(key: String, index: Int = 0): Long = row(key)[index].toLong()
    fun double(key: String, index: Int = 0): Double = row(key)[index].toDouble()
    fun doubles(key: String): DoubleArray = row(key).map { it.toDouble() }.toDoubleArray()

    /**
     * The shared input, regenerated here rather than transferred.
     *
     * Must match `make_port_fixtures.lcg_plane` exactly. Masking to 31 bits
     * keeps every intermediate inside a signed 32-bit range, so Python's
     * arbitrary-precision integers and this Long arithmetic cannot diverge
     * through an overflow that only one side has.
     */
    internal fun lcgPlane(): SkewEstimator.Plane {
        val rows = int("lcg.rows")
        val cols = int("lcg.cols")
        val mult = long("lcg.mult")
        val add = long("lcg.add")
        val mask = long("lcg.mask")
        var state = long("lcg.seed") and mask
        val data = DoubleArray(rows * cols)
        for (i in data.indices) {
            state = (state * mult + add) and mask
            data[i] = (state % 256L).toDouble()
        }
        return SkewEstimator.Plane(data, rows, cols)
    }
}
