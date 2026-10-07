package com.maarijalam.eacg.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/**
 * The maroon light scheme of docs/ui_plan.md §2. No dynamic colour (Android
 * 12+ wallpaper colours would replace the maroon) and no dark theme in this
 * phase. Material roles that the screens do not set explicitly are mapped
 * onto the same tokens, so stock components (switches, progress bars, text
 * fields) already look right.
 */
private val EaCGColors = lightColorScheme(
    primary = Primary,
    onPrimary = Color.White,
    primaryContainer = Tint,
    onPrimaryContainer = PrimaryDark,
    secondary = Primary,
    onSecondary = Color.White,
    secondaryContainer = Tint,             // NavigationBar's selected indicator
    onSecondaryContainer = PrimaryDark,
    tertiary = Data,
    onTertiary = Color.White,
    background = Ground,
    onBackground = Ink,
    surface = CardSurface,
    onSurface = Ink,
    surfaceVariant = Ground,
    onSurfaceVariant = Muted,
    surfaceContainer = CardSurface,            // NavigationBar
    surfaceContainerLow = CardSurface,         // cards
    surfaceContainerHigh = CardSurface,
    surfaceContainerHighest = ProgressTrack,
    outline = ButtonOutline,
    outlineVariant = CardBorder,
    error = BadText,
    onError = Color.White,
    errorContainer = BadBg,
    onErrorContainer = BadText,
)

@Composable
fun EaCGTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = EaCGColors, typography = Typography, content = content)
}
