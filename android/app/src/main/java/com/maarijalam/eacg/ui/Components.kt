package com.maarijalam.eacg.ui

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.theme.ButtonOutline
import com.maarijalam.eacg.ui.theme.CardBorder
import com.maarijalam.eacg.ui.theme.CardSurface
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.Text2

/*
 * Shared pieces of the redesign (docs/ui_plan.md §2.3): rounded corners only
 * on buttons (full radius, 52 dp) and cards (16-20 dp); 48 dp touch targets;
 * every icon-only button labelled.
 */

const val DISCLAIMER = "Research prototype. Not a medical device; not for diagnosis."

/** Inner-screen top bar: back (or close), a 20/600 title, optional subtitle and actions. */
@Composable
fun TopBar(
    title: String,
    onBack: (() -> Unit)?,
    modifier: Modifier = Modifier,
    subtitle: String? = null,
    backIcon: ImageVector = EaIcons.Back,
    backLabel: String = "Back",
    contentColor: Color = Text2,
    titleColor: Color = MaterialTheme.colorScheme.onBackground,
    actions: @Composable RowScope.() -> Unit = {},
) {
    Row(modifier.fillMaxWidth().padding(start = 8.dp, end = 4.dp, top = 8.dp, bottom = 4.dp),
        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        if (onBack != null) {
            IconButton(onClick = onBack, modifier = Modifier.size(48.dp)) {
                Icon(backIcon, contentDescription = backLabel, tint = contentColor)
            }
        } else Box(Modifier.size(8.dp))
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.titleLarge, color = titleColor)
            if (subtitle != null) Text(subtitle, style = MaterialTheme.typography.bodySmall, color = Muted)
        }
        actions()
    }
}

/** A white card with the 1 dp border. */
@Composable
fun EaCard(
    modifier: Modifier = Modifier,
    radius: Dp = 16.dp,
    padding: PaddingValues = PaddingValues(0.dp),
    content: @Composable ColumnScope.() -> Unit,
) {
    val shape = RoundedCornerShape(radius)
    Column(modifier.fillMaxWidth().background(CardSurface, shape).border(1.dp, CardBorder, shape)
        .padding(padding), content = content)
}

/** The full-radius 52 dp primary button. */
@Composable
fun PrimaryButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier,
                  enabled: Boolean = true, icon: ImageVector? = null) {
    Button(onClick = onClick, enabled = enabled, modifier = modifier.height(52.dp),
        shape = RoundedCornerShape(26.dp),
        colors = ButtonDefaults.buttonColors(containerColor = Primary, contentColor = Color.White)) {
        if (icon != null) { Icon(icon, null, Modifier.size(20.dp)); Box(Modifier.size(10.dp)) }
        Text(text, style = MaterialTheme.typography.labelLarge)
    }
}

/** The full-radius 52 dp outlined button. */
@Composable
fun SecondaryButton(text: String, onClick: () -> Unit, modifier: Modifier = Modifier,
                    enabled: Boolean = true, icon: ImageVector? = null) {
    OutlinedButton(onClick = onClick, enabled = enabled, modifier = modifier.height(52.dp),
        shape = RoundedCornerShape(26.dp), border = BorderStroke(1.dp, ButtonOutline),
        colors = ButtonDefaults.outlinedButtonColors(containerColor = CardSurface, contentColor = Primary)) {
        if (icon != null) { Icon(icon, null, Modifier.size(20.dp)); Box(Modifier.size(10.dp)) }
        Text(text, style = MaterialTheme.typography.labelLarge)
    }
}

/** A 12/600 upper-case label above a group ("RESEARCH OPTIONS"). */
@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier) {
    Text(text.uppercase(), modifier, style = MaterialTheme.typography.labelMedium.copy(
        letterSpacing = MaterialTheme.typography.labelMedium.fontSize * 0.06f), color = Muted)
}

/** The footer disclaimer line. */
@Composable
fun Disclaimer(modifier: Modifier = Modifier, text: String = DISCLAIMER) {
    Text(text, modifier.fillMaxWidth(), style = MaterialTheme.typography.labelSmall,
        color = Muted, textAlign = TextAlign.Center)
}
