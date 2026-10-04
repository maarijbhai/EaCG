package com.maarijalam.eacg

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.maarijalam.eacg.ui.theme.EaCGTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            EaCGTheme {
                var tab by rememberSaveable { mutableIntStateOf(0) }
                Scaffold(modifier = Modifier.fillMaxSize(), bottomBar = {
                    NavigationBar {
                        NavigationBarItem(selected = tab == 0, onClick = { tab = 0 },
                            icon = { Icon(painterResource(android.R.drawable.ic_menu_camera), null) },
                            label = { Text("Scan") })
                        NavigationBarItem(selected = tab == 1, onClick = { tab = 1 },
                            icon = { Icon(painterResource(android.R.drawable.ic_menu_recent_history), null) },
                            label = { Text("Records") })
                    }
                }) { innerPadding ->
                    Box(Modifier.padding(innerPadding).fillMaxSize()) {
                        // The Scan screen stays composed while Records is shown
                        // (at zero size), so a digitisation in flight -- ~20 s
                        // with the learned tracer -- is not cancelled, and the
                        // picked photo and corners survive a look at Records.
                        Box(if (tab == 0) Modifier.fillMaxSize() else Modifier.size(0.dp)) {
                            DigitiseScreen(modifier = Modifier.fillMaxSize())
                        }
                        if (tab == 1) RecordsScreen(Modifier.fillMaxSize())
                    }
                }
            }
        }
    }
}
