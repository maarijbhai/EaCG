package com.maarijalam.eacg

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.Disclaimer
import com.maarijalam.eacg.ui.EaIcons
import com.maarijalam.eacg.ui.theme.CardBorder
import com.maarijalam.eacg.ui.theme.CardSurface
import com.maarijalam.eacg.ui.theme.EaCGTheme
import com.maarijalam.eacg.ui.theme.Ground
import com.maarijalam.eacg.ui.theme.Muted
import com.maarijalam.eacg.ui.theme.Primary
import com.maarijalam.eacg.ui.theme.Text2
import com.maarijalam.eacg.ui.theme.Tint

/** A full screen shown over the tabs (docs/ui_plan.md §3: Settings, Lab). */
private enum class Overlay { NONE, SETTINGS, LAB }

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val settings = AppSettings.get(this)
        setContent {
            EaCGTheme {
                CompositionLocalProvider(LocalAppSettings provides settings) {
                    AppShell()
                }
            }
        }
    }
}

/**
 * Bottom bar with Scan and Records, and Settings / Lab as full screens on
 * top. The Scan tab stays composed (at zero size) while anything else shows,
 * so a digitisation in flight -- ~20 s with the learned tracer -- is not
 * cancelled and the picked photo and corners survive. The Lab, once opened,
 * is kept the same way, so a running benchmark survives a trip back.
 */
@Composable
private fun AppShell() {
    val settings = LocalAppSettings.current
    var tab by rememberSaveable { mutableIntStateOf(0) }
    var overlay by rememberSaveable { mutableStateOf(Overlay.NONE) }
    var labOpened by rememberSaveable { mutableStateOf(false) }
    var recordToOpen by remember { mutableStateOf<RecordStore.Meta?>(null) }
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    // One controller for the Scan flow, held here so it outlives tab switches.
    val scan = remember { ScanController(context.applicationContext, scope, settings) }
    // Prototype mode off hides the Lab; leave it if it was showing.
    if (!settings.prototypeMode && overlay == Overlay.LAB) overlay = Overlay.SETTINGS

    BackHandler(enabled = overlay != Overlay.NONE) {
        overlay = if (overlay == Overlay.LAB) Overlay.SETTINGS else Overlay.NONE
    }

    Scaffold(
        modifier = Modifier.fillMaxSize(),
        containerColor = Ground,
        bottomBar = { if (overlay == Overlay.NONE) MainNavBar(tab) { tab = it } }
    ) { innerPadding ->
        Box(Modifier.padding(innerPadding).fillMaxSize()) {
            val scanShown = overlay == Overlay.NONE && tab == 0
            Box(if (scanShown) Modifier.fillMaxSize() else Modifier.size(0.dp)) {
                ScanFlow(scan, onOpenSettings = { overlay = Overlay.SETTINGS },
                    onSeeAllRecords = { tab = 1 },
                    onOpenRecord = { recordToOpen = it; tab = 1 })
            }
            if (overlay == Overlay.NONE && tab == 1) RecordsScreen(Modifier.fillMaxSize(),
                openRequest = recordToOpen, onOpenHandled = { recordToOpen = null })
            if (overlay == Overlay.SETTINGS) {
                SettingsScreen(onBack = { overlay = Overlay.NONE },
                    onOpenLab = { labOpened = true; overlay = Overlay.LAB },
                    modifier = Modifier.fillMaxSize().background(Ground))
            }
            if (labOpened && settings.prototypeMode) {
                Box(if (overlay == Overlay.LAB) Modifier.fillMaxSize().background(Ground) else Modifier.size(0.dp)) {
                    LabScreen(onBack = { overlay = Overlay.SETTINGS })
                }
            }
        }
    }
}

/** The mock-up's bottom navigation: maroon icon and label in a tint pill when selected. */
@Composable
private fun MainNavBar(tab: Int, onSelect: (Int) -> Unit) {
    Column {
        HorizontalDivider(color = CardBorder)
        NavigationBar(containerColor = CardSurface, tonalElevation = 0.dp) {
            val colors = NavigationBarItemDefaults.colors(
                selectedIconColor = Primary, selectedTextColor = Primary, indicatorColor = Tint,
                unselectedIconColor = Muted, unselectedTextColor = Muted)
            NavigationBarItem(selected = tab == 0, onClick = { onSelect(0) }, colors = colors,
                icon = { Icon(EaIcons.Camera, contentDescription = null, modifier = Modifier.size(22.dp)) },
                label = { NavLabel("Scan", tab == 0) })
            NavigationBarItem(selected = tab == 1, onClick = { onSelect(1) }, colors = colors,
                icon = { Icon(EaIcons.List, contentDescription = null, modifier = Modifier.size(22.dp)) },
                label = { NavLabel("Records", tab == 1) })
        }
    }
}

@Composable
private fun NavLabel(text: String, selected: Boolean) {
    Text(text, style = MaterialTheme.typography.labelSmall.copy(
        fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium))
}
