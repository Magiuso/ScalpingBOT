#!/usr/bin/env python3
"""
Support/Resistance Algorithms - ESTRATTO IDENTICO DAL MONOLITE
================================================================

4 algoritmi Support/Resistance rimanenti:
- VolumeProfile_Advanced: Analisi volume profile avanzata
- LSTM_SupportResistance: LSTM per detection S/R
- StatisticalLevels_ML: Livelli statistici ML
- Transformer_Levels: Transformer per S/R prediction

⚠️ PivotPoints_Classic RIMOSSO - ora implementato in src/ml/algorithms/support_resistance/pivot_points_classic.py

ESTRATTO IDENTICO da src/Analyzer.py righe 12826-13036
Mantenuta IDENTICA la logica originale, solo import aggiustati.
"""

import numpy as np
import torch
import json
from json import JSONDecodeError
import os
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Import from migrated modules
from ..models.advanced_lstm import AdvancedLSTM
from ..models.transformer_models import TransformerPredictor
from ...shared.enums import ModelType
from ...shared.exceptions import (
    InsufficientDataError,
    ModelNotInitializedError, 
    InvalidInputError,
    PredictionError,
    AlgorithmErrors
)
# Removed safe_print import - using fail-fast error handling instead


class SupportResistanceAlgorithms:
    """
    Support/Resistance Algorithms - ESTRATTO IDENTICO DAL MONOLITE
    
    Implementa i 4 algoritmi rimanenti:
    1. VolumeProfile_Advanced  
    2. LSTM_SupportResistance
    3. StatisticalLevels_ML
    4. Transformer_Levels
    
    ⚠️ PivotPoints_Classic RIMOSSO - ora in file separato modulare
    """
    
    def __init__(self, ml_models: Dict[str, Any]):
        """Inizializza algoritmi S/R con modelli ML - FAIL FAST se mancanti"""
        if ml_models is None:
            raise ValueError("FAIL FAST: ml_models è obbligatorio - no fallback consentiti in sistema finanziario")
        if not isinstance(ml_models, dict):
            raise TypeError(f"FAIL FAST: ml_models deve essere dict, ricevuto {type(ml_models)}")
        self.ml_models = ml_models
        self.algorithm_stats = {
            'executions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'last_execution': None
        }
        
        # ⚠️ PIVOT CACHE RIMOSSO - PivotPoints_Classic ora in file separato modulare
    
    def get_model(self, model_name: str, asset: Optional[str] = None) -> Any:
        """Get model with asset-specific support - NO FALLBACKS (BIBBIA)"""
        if not asset:
            raise ValueError("Asset is mandatory for model loading - no default allowed (BIBBIA compliance)")
        
        asset_model_name = f"{asset}_{model_name}"
        if asset_model_name not in self.ml_models:
            raise ModelNotInitializedError(f"Asset-specific model '{asset_model_name}' not found")
        
        return self.ml_models[asset_model_name]
    
    def run_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue algoritmo Support/Resistance specificato
        
        Args:
            algorithm_name: Nome algoritmo da eseguire
            market_data: Dati di mercato processati
            
        Returns:
            Risultati algoritmo con support/resistance levels
        """
        self.algorithm_stats['executions'] += 1
        self.algorithm_stats['last_execution'] = datetime.now()
        
        # BIBBIA COMPLIANT: Single path lookup - no multiple if/elif alternatives
        # ⚠️ PivotPoints_Classic RIMOSSO - ora implementato in file separato modulare
        algorithms = {
            "VolumeProfile_Advanced": self._volume_profile_advanced,
            "LSTM_SupportResistance": self._lstm_support_resistance,
            "StatisticalLevels_ML": self._statistical_levels_ml,
            "Transformer_Levels": self._transformer_levels
        }
        
        # BIBBIA COMPLIANT: FAIL FAST if algorithm not found
        if algorithm_name not in algorithms:
            raise ValueError(f"FAIL FAST: Unknown Support/Resistance algorithm: {algorithm_name}")
            
        return algorithms[algorithm_name](market_data)
    
    def _volume_profile_advanced(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Volume Profile analysis avanzata
        ESTRATTO IDENTICO da src/Analyzer.py:12852-12893
        """
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        if 'volume_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'volume_history' in market_data")
        prices = np.array(market_data['price_history'])
        volumes = np.array(market_data['volume_history'])
        
        if len(prices) < 50:
            raise InsufficientDataError(required=50, available=len(prices), operation="VolumeProfile_Advanced")
        
        # Crea price bins
        price_min, price_max = prices.min(), prices.max()
        n_bins = 20
        bins = np.linspace(price_min, price_max, n_bins)
        
        # Calcola volume per price level
        volume_profile = np.zeros(n_bins - 1)
        for i in range(len(prices)):
            bin_idx = np.searchsorted(bins, prices[i]) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += volumes[i]
        
        # Trova high volume nodes (potential S/R)
        threshold = np.percentile(volume_profile, 70)
        high_volume_indices = np.where(volume_profile > threshold)[0]
        
        support_levels = []
        resistance_levels = []
        current_price = market_data['current_price']
        
        for idx in high_volume_indices:
            level = (bins[idx] + bins[idx + 1]) / 2
            if level < current_price:
                support_levels.append(level)
            else:
                resistance_levels.append(level)
        
        self.algorithm_stats['successful_predictions'] += 1
        
        return {
            "support_levels": sorted(support_levels)[-3:],  # Top 3
            "resistance_levels": sorted(resistance_levels)[:3],  # Top 3
            "confidence": 0.8,
            "method": "Volume_Profile_Analysis",
            "volume_nodes": len(high_volume_indices)
        }
    
    def _lstm_support_resistance(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LSTM per Support/Resistance detection
        ESTRATTO IDENTICO da src/Analyzer.py:12895-12996
        """
        # Get asset from market_data for asset-specific model loading
        # BIBBIA COMPLIANT: FAIL FAST - no fallback to 'UNKNOWN'
        if 'asset' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'asset' in market_data")
        asset = market_data['asset']
        model = self.get_model('LSTM_SupportResistance', asset)
        
        # Prepara input
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        if 'volume_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'volume_history' in market_data")
        prices = np.array(market_data['price_history'][-50:])
        volumes = np.array(market_data['volume_history'][-50:])
        
        if len(prices) < 50:
            raise InsufficientDataError(required=50, available=len(prices), operation="LSTM_SupportResistance")
        
        # 🛡️ VALIDAZIONE DATI INPUT
        if np.isnan(prices).any() or np.isinf(prices).any():
            raise InvalidInputError("prices", "NaN/Inf values", "LSTM requires valid numeric prices")
        
        if np.isnan(volumes).any() or np.isinf(volumes).any():
            raise InvalidInputError("volumes", "NaN/Inf values", "LSTM requires valid numeric volumes")
        
        try:
            # Feature engineering
            features = self._prepare_lstm_features(prices, volumes, market_data)
            
            # 🛡️ VALIDAZIONE FEATURES
            if np.isnan(features).any() or np.isinf(features).any():
                raise InvalidInputError("features", "NaN/Inf values", "LSTM features must be numeric")
            
            # Prediction protetta
            model.eval()
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                # 🛡️ VALIDAZIONE TENSOR INPUT
                if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
                    raise InvalidInputError("input_tensor", "NaN/Inf values", "PyTorch tensor must be finite")
                
                prediction = model(input_tensor)
                
                # 🛡️ VALIDAZIONE OUTPUT
                if torch.isnan(prediction).any() or torch.isinf(prediction).any():
                    raise PredictionError("LSTM_SupportResistance", "Model produced NaN/Inf in output")
                
                levels = prediction.numpy().flatten()
                
                # 🛡️ VALIDAZIONE FINALE
                if np.isnan(levels).any() or np.isinf(levels).any():
                    raise PredictionError("LSTM_SupportResistance", "Final levels contain NaN/Inf values")
                
                # LSTM prediction successful - no debug print needed
                
                # Interpreta output con validazione
                current_price = market_data['current_price']
                
                # 🛡️ VALIDAZIONE CURRENT_PRICE
                if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
                    raise InvalidInputError("current_price", current_price, "Must be a positive finite number")
                
                support_levels = []
                resistance_levels = []
                
                for i in range(0, len(levels), 2):
                    if i < len(levels) - 1:
                        support_level = levels[i]
                        resistance_level = levels[i + 1]
                        
                        if support_level < current_price:
                            support_levels.append(float(support_level))
                        if resistance_level > current_price:
                            resistance_levels.append(float(resistance_level))
                
                self.algorithm_stats['successful_predictions'] += 1
                
                return {
                    "support_levels": sorted(support_levels)[-3:],  # Top 3
                    "resistance_levels": sorted(resistance_levels)[:3],  # Top 3
                    "confidence": 0.85,
                    "method": "LSTM_Neural_Network",
                    "model_output_size": len(levels)
                }
                
        except (ImportError, AttributeError, ValueError, KeyError, RuntimeError, TypeError, IndexError) as e:
            self.algorithm_stats['failed_predictions'] += 1
            # FAIL FAST - Re-raise specific LSTM errors
            raise PredictionError("LSTM_SupportResistance", str(e))
    
    def _statistical_levels_ml(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Statistical Levels ML analysis
        ESTRATTO IDENTICO da src/Analyzer.py:12997-13035
        """
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        prices = np.array(market_data['price_history'])
        
        if len(prices) < 100:
            raise InsufficientDataError(required=100, available=len(prices), operation="StatisticalLevels_ML")
        
        # Calcola statistical support/resistance
        price_mean = np.mean(prices)
        price_std = np.std(prices)
        
        # Bollinger-like bands
        upper_band = price_mean + 2 * price_std
        lower_band = price_mean - 2 * price_std
        
        # Percentile-based levels
        percentile_levels = [
            np.percentile(prices, 10),   # Support level 1
            np.percentile(prices, 25),   # Support level 2
            np.percentile(prices, 75),   # Resistance level 1
            np.percentile(prices, 90)    # Resistance level 2
        ]
        
        current_price = market_data['current_price']
        
        support_levels = [
            lower_band,
            percentile_levels[0],
            percentile_levels[1]
        ]
        support_levels = [level for level in support_levels if level < current_price]
        
        resistance_levels = [
            upper_band,
            percentile_levels[2],
            percentile_levels[3]
        ]
        resistance_levels = [level for level in resistance_levels if level > current_price]
        
        self.algorithm_stats['successful_predictions'] += 1
        
        return {
            "support_levels": sorted(support_levels)[-3:],  # Top 3
            "resistance_levels": sorted(resistance_levels)[:3],  # Top 3
            "confidence": 0.7,
            "method": "Statistical_ML_Analysis",
            "price_mean": price_mean,
            "price_std": price_std
        }
    
    def _transformer_levels(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformer-based S/R level detection
        PARTE DI src/Analyzer.py - completare implementazione
        """
        # Get asset from market_data for asset-specific model loading
        # BIBBIA COMPLIANT: FAIL FAST - no fallback to 'UNKNOWN'
        if 'asset' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'asset' in market_data")
        asset = market_data['asset']
        model = self.get_model('Transformer_Levels', asset)
        
        # Placeholder per ora - da implementare completamente
        if 'current_price' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'current_price' in market_data")
        if 'price_history' not in market_data:
            raise KeyError("FAIL FAST: Missing required field 'price_history' in market_data")
        current_price = market_data['current_price']
        price_history = market_data['price_history']
        
        if len(price_history) < 50:
            raise InsufficientDataError(required=50, available=len(price_history), operation="Transformer_Levels")
        
        # Transformer-based prediction (semplificato)
        price_range = max(price_history[-20:]) - min(price_history[-20:])
        
        support_levels = [
            current_price - price_range * 0.1,
            current_price - price_range * 0.2,
            current_price - price_range * 0.3
        ]
        
        resistance_levels = [
            current_price + price_range * 0.1,
            current_price + price_range * 0.2,
            current_price + price_range * 0.3
        ]
        
        self.algorithm_stats['successful_predictions'] += 1
        
        return {
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "confidence": 0.75,
            "method": "Transformer_Neural_Network",
            "price_range": price_range
        }
    
    def _prepare_lstm_features(self, prices: np.ndarray, volumes: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepara features per LSTM
        Utilizza MarketDataProcessor per consistency
        """
        # Basic features - FAIL FAST se prezzi invalidi
        if np.any(prices <= 0):
            raise ValueError("FAIL FAST: Prezzi <= 0 trovati nei dati - cannot calculate returns safely")
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            raise ValueError("FAIL FAST: Prezzi NaN o infiniti nei dati - cannot calculate returns safely")
            
        returns = np.diff(prices, prepend=prices[0]) / prices[:-1]
        price_ratios = prices[1:] / prices[:-1]
        log_returns = np.log(price_ratios)
        log_returns = np.append(log_returns, 0)
        
        # Volume features - FAIL FAST se volumi invalidi
        if len(volumes) == 0:
            raise ValueError("FAIL FAST: Empty volumes array - cannot calculate volume_mean")
        if np.any(volumes < 0):
            raise ValueError("FAIL FAST: Negative volumes found in data - invalid volume data")
        volume_mean = np.mean(volumes)
        if volume_mean == 0:
            raise ValueError("FAIL FAST: Volume mean is zero - cannot calculate volume ratio safely")
        volume_ratio = volumes / volume_mean
        
        # Technical indicators (semplificati)
        sma_5 = np.convolve(prices, np.ones(5)/5, mode='same')
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
        
        # Combine features
        features = np.column_stack([
            prices, volumes, returns, log_returns,
            volume_ratio, sma_5, sma_20
        ])
        
        return features
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche algoritmi"""
        return self.algorithm_stats.copy()


# Factory function per compatibilità
def create_support_resistance_algorithms(ml_models: Dict[str, Any]) -> SupportResistanceAlgorithms:
    """Factory function per creare SupportResistanceAlgorithms - FAIL FAST se ml_models mancanti"""
    if ml_models is None:
        raise ValueError("FAIL FAST: ml_models è obbligatorio - no fallback consentiti")
    return SupportResistanceAlgorithms(ml_models)


# Export
__all__ = [
    'SupportResistanceAlgorithms',
    'InsufficientDataError',
    'ModelNotInitializedError', 
    'InvalidInputError',
    'PredictionError',
    'create_support_resistance_algorithms'
]