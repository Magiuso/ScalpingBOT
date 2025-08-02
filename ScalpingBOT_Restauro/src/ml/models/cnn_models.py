"""
🧠 CNN Pattern Recognition Models

Extracted from: C:\\ScalpingBOT\\src\\Analyzer.py (lines 5112-5163)
Migration Date: 2025-08-01
Purpose: CNN-based pattern recognition for market data analysis
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Any


# Global debug mode setting for conditional logging
DEBUG_MODE = os.getenv('SCALPINGBOT_DEBUG', 'false').lower() == 'true'


def conditional_smart_print(message: str, category: str = 'general', severity: str = "info") -> None:
    """🔧 SPAM FIX: Log solo se in debug mode o se severity >= WARNING"""
    # Filtra più aggressivamente i log ripetitivi del training
    skip_categories = ['forward', 'architecture_fixes', 'validation', 'normalization', 'tensor_validation']
    if category in skip_categories and severity not in ['error', 'critical']:
        return
        
    if DEBUG_MODE or severity in ['warning', 'error', 'critical']:
        smart_print(message, category)


def smart_print(message: str, category: str = 'general') -> None:
    """Safe print con rate limiting intelligente"""
    try:
        print(f"[{category}] {message}")
    except Exception:
        # Fallback silenzioso se il print fallisce
        pass


class CNNPatternRecognizer(nn.Module):
    """CNN 1D per riconoscimento pattern grafici"""
    def __init__(self, input_channels: int = 1, sequence_length: int = 100, num_patterns: int = 50):
        super(CNNPatternRecognizer, self).__init__()
        self.parent: Optional[Any] = None  # ✅ Inizializza parent reference
        
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(input_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Second conv block
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Third conv block
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Fourth conv block
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_patterns),
            nn.Sigmoid()
        )
    
    def _log(self, message: str, category: str = "cnn", severity: str = "info"):
        """Helper per logging che funziona con o senza parent"""
        conditional_smart_print(f"[{category}] {message}", category, severity)
        
    def forward(self, x):
        x = self.conv_layers(x)
        return self.classifier(x)