"""
Transformer Models for Advanced Pattern Recognition

This module contains the TransformerPredictor class extracted from the main Analyzer.py
for advanced pattern recognition using transformer architecture.

Extracted from: src/Analyzer.py lines 4437-4479
Migration date: 2025-08-01
Status: Pure extraction - identical logic maintained
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Any

# Debug mode configuration
DEBUG_MODE = os.getenv('SCALPINGBOT_DEBUG', 'false').lower() == 'true'

def conditional_smart_print(message: str, category: str = 'general', severity: str = "info") -> None:
    """Log intelligente con filtri per evitare spam - estratto da Analyzer.py"""
    # Filtra più aggressivamente i log ripetitivi del training
    skip_categories = ['forward', 'architecture_fixes', 'validation', 'normalization', 'tensor_validation']
    if category in skip_categories and severity not in ['error', 'critical']:
        return
        
    if DEBUG_MODE or severity in ['warning', 'error', 'critical']:
        print(f"[{category.upper()}] {message}")


class TransformerPredictor(nn.Module):
    """Transformer per pattern recognition avanzato"""
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 6, output_dim: int = 1):
        super(TransformerPredictor, self).__init__()
        self.parent: Optional[Any] = None  # ✅ Inizializza parent reference
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)
        )
    
    def _log(self, message: str, category: str = "transformer", severity: str = "info"):
        """Helper per logging che funziona con o senza parent"""
        conditional_smart_print(f"[{category}] {message}", category, severity)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input and add positional encoding
        x = self.input_projection(x)
        x += self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        
        # Take last token and project to output
        x = x[-1]  # (batch, d_model)
        return self.output_layer(x)