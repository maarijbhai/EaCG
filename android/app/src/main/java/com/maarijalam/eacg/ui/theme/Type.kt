package com.maarijalam.eacg.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp
import com.maarijalam.eacg.R

/*
 * IBM Plex Sans and Mono, bundled in res/font (OFL 1.1, IBM/plex v6.4.0;
 * licence in assets/licenses/OFL-IBM-Plex.txt) so the app needs no network.
 * Mono is for numbers read as data: times, tile counts, px/mm, r values.
 * Sizes from the mock-up (docs/ui_plan.md §2.2).
 */

val PlexSans = FontFamily(
    Font(R.font.ibm_plex_sans_regular, FontWeight.Normal),
    Font(R.font.ibm_plex_sans_medium, FontWeight.Medium),
    Font(R.font.ibm_plex_sans_semibold, FontWeight.SemiBold),
    Font(R.font.ibm_plex_sans_bold, FontWeight.Bold),
)

val PlexMono = FontFamily(
    Font(R.font.ibm_plex_mono_regular, FontWeight.Normal),
    Font(R.font.ibm_plex_mono_medium, FontWeight.Medium),
)

private fun sans(size: Int, weight: FontWeight, line: Int, spacing: Double = 0.0) = TextStyle(
    fontFamily = PlexSans, fontWeight = weight, fontSize = size.sp,
    lineHeight = line.sp, letterSpacing = spacing.sp)

/**
 * Material roles mapped onto the mock-up's sizes:
 * headlineSmall 24/700 (Home, Records title) · titleLarge 20/600 (inner
 * screen titles) · titleMedium 16/600 (section titles, buttons) ·
 * bodyLarge 15/400 (body) · bodyMedium 14 · bodySmall 13 · labels 12-14.
 */
val Typography = Typography(
    headlineSmall = sans(24, FontWeight.Bold, 30, -0.24),
    titleLarge = sans(20, FontWeight.SemiBold, 26),
    titleMedium = sans(16, FontWeight.SemiBold, 22),
    titleSmall = sans(15, FontWeight.SemiBold, 20),
    bodyLarge = sans(15, FontWeight.Normal, 22),
    bodyMedium = sans(14, FontWeight.Normal, 20),
    bodySmall = sans(13, FontWeight.Normal, 18),
    labelLarge = sans(16, FontWeight.SemiBold, 20),
    labelMedium = sans(13, FontWeight.SemiBold, 16),
    labelSmall = sans(12, FontWeight.Normal, 16),
)

/** Numbers read as data (the mock-up's IBM Plex Mono, 13-14 px). */
val DataText = TextStyle(fontFamily = PlexMono, fontWeight = FontWeight.Normal,
    fontSize = 13.sp, lineHeight = 18.sp)
