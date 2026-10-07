package com.maarijalam.eacg.ui.theme

import androidx.compose.ui.graphics.Color

/*
 * Colour tokens of the approved UI (docs/ui_plan.md §2.1; the mock-up at
 * https://claude.ai/artifact/QSpathWQ3beoFJsQt3sTfh). One maroon accent on a
 * warm light ground, a teal reserved for digitised data, amber and red only
 * for quality verdicts, and the pink ECG paper behind every trace.
 */

// Accent
val Primary = Color(0xFF7A1F2B)        // buttons, selected tab underline and nav, links, "Good"
val PrimaryDark = Color(0xFF5A1520)    // text on tint
val Tint = Color(0xFFF4E1E4)           // selected nav indicator, selected segment
val TintBorder = Color(0xFFE6BFC6)     // border on tinted areas

// Surfaces and lines
val Ground = Color(0xFFF7F3F1)         // screen background
val CardSurface = Color(0xFFFFFFFF)        // cards, lists, bottom nav
val CardBorder = Color(0xFFE7DFDC)         // card borders, tab baseline
val RowDivider = Color(0xFFF0EAE7)        // rows inside cards
val ButtonOutline = Color(0xFFD4C7C3)        // outlined buttons
val InputOutline = Color(0xFFD9CDC9)   // inputs, corner close-up frames
val ProgressTrack = Color(0xFFEBD9DC)  // progress bar track, pending spinner ring

// Text
val Ink = Color(0xFF1F1719)            // primary text
val Text2 = Color(0xFF4A3D41)          // secondary text, icons
val Muted = Color(0xFF5E5054)          // captions, labels (4.5:1 on Ground)

// The corners (photo) screen
val DarkScreen = Color(0xFF1C1215)
val OnDarkScreen = Color(0xFFF3F4F6)
val CornerOverlay = Color(0xFFC02942)  // handles, close-up crosshairs
val CornerLine = Color(0xFFD33A52)     // the detected quad

// Digitised data: overlays, scan line, time markers
val Data = Color(0xFF0F766E)

// Quality verdicts (SheetQuality): "check" ...
val WarnBg = Color(0xFFFDEFD9)
val WarnBorder = Color(0xFFF3C98B)
val WarnText = Color(0xFF6B2E05)
val WarnIcon = Color(0xFF8A3B06)
val WarnIconSmall = Color(0xFFB45309)  // the small triangle in detail rows
val MissingTick = Color(0xFFD97706)    // a tick not found, drawn at its borrowed position
// ... and "unreliable" (same layout as the warning banner)
val BadBg = Color(0xFFFDECEA)
val BadBorder = Color(0xFFF5C2BD)
val BadText = Color(0xFF9A1C13)

// ECG paper behind every trace (1 mm minor, 5 mm major)
val EcgPaper = Color(0xFFFFF8F8)
val EcgMinor = Color(0xFFFBE4E4)
val EcgMajor = Color(0xFFF2C4C4)
val EcgTrace = Color(0xFF1F2328)       // printed or digitised trace on the grid
val TruthTrace = Color(0xFF9CA3AF)     // PTB-XL truth under the digitised trace (prototype mode)
