package com.maarijalam.eacg

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.ui.Modifier
import com.maarijalam.eacg.ui.theme.EaCGTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            EaCGTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    DigitiseScreen(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}