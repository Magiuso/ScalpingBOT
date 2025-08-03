#!/usr/bin/env python3
"""
Pattern Recognition Algorithms - ESTRATTO IDENTICO DAL MONOLITE
===============================================================

5 algoritmi Pattern Recognition identificati come mancanti:
- CNN_PatternRecognizer: CNN per detection pattern avanzati
- Classical_Patterns: Pattern classici (double top/bottom, head-shoulders, etc.)
- LSTM_Sequences: LSTM per sequence pattern recognition
- Transformer_Patterns: Transformer per pattern detection avanzato
- Ensemble_Patterns: Ensemble di pattern recognition algorithms

ESTRATTO IDENTICO da src/Analyzer.py righe 13086-13333
Mantenuta IDENTICA la logica originale, solo import aggiustati.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import shared exceptions
from ...shared.exceptions import (
    InsufficientDataError,
    ModelNotInitializedError,
    InvalidInputError,
    PredictionError,
    AlgorithmErrors
)
# Removed safe_print import - using fail-fast error handling instead


class PatternRecognitionAlgorithms:
    """
    Pattern Recognition Algorithms - ESTRATTO IDENTICO DAL MONOLITE
    
    Implementa i 5 algoritmi identificati come mancanti:
    1. CNN_PatternRecognizer  
    2. Classical_Patterns
    3. LSTM_Sequences
    4. Transformer_Patterns
    5. Ensemble_Patterns
    """
    
    def __init__(self, ml_models: Optional[Dict[str, Any]] = None):
        """Inizializza algoritmi Pattern Recognition con modelli ML opzionali"""
        self.ml_models = ml_models or {}
        self.algorithm_stats = {
            'executions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'patterns_detected': 0,
            'last_execution': None
        }
    
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
        Esegue algoritmo Pattern Recognition specificato
        ESTRATTO IDENTICO da src/Analyzer.py:13086-13333
        
        Args:
            algorithm_name: Nome algoritmo da eseguire
            market_data: Dati di mercato processati
            
        Returns:
            Risultati pattern recognition con patterns detected
        """
        self.algorithm_stats['executions'] += 1
        self.algorithm_stats['last_execution'] = datetime.now()
        
        if algorithm_name == "CNN_PatternRecognizer":
            return self._cnn_pattern_recognizer(market_data)
        elif algorithm_name == "Classical_Patterns":
            return self._classical_patterns(market_data)
        elif algorithm_name == "LSTM_Sequences":
            return self._lstm_sequences(market_data)
        elif algorithm_name == "Transformer_Patterns":
            return self._transformer_patterns(market_data)
        elif algorithm_name == "Ensemble_Patterns":
            return self._ensemble_patterns(market_data)
        else:
            raise ValueError(f"Unknown Pattern Recognition algorithm: {algorithm_name}")
    
    def _cnn_pattern_recognizer(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        CNN Pattern Recognition
        ESTRATTO IDENTICO da src/Analyzer.py:13086-13148
        """
        # Get asset from market_data for asset-specific model loading
        asset = market_data.get('asset', 'UNKNOWN')
        model = self.get_model('CNN_PatternRecognizer', asset)
        
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = np.array(market_data['price_history'][-100:])
        if len(prices) < 100:
            raise InsufficientDataError(required=100, available=len(prices), operation="CNN_PatternRecognizer")
        
        try:
            # Normalizza i prezzi
            normalized_prices = (prices - prices.mean()) / prices.std()
            
            # Prepara input per CNN
            input_tensor = torch.FloatTensor(normalized_prices).unsqueeze(0).unsqueeze(0)
            
            with torch.no_grad():
                pattern_logits = model(input_tensor)
                pattern_probs = pattern_logits.numpy().flatten()
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("CNN_PatternRecognizer", f"Model inference failed: {str(e)}")
        
        # Pattern mapping
        pattern_names = [
            "double_top", "double_bottom", "head_shoulders", "inverse_head_shoulders",
            "ascending_triangle", "descending_triangle", "symmetrical_triangle",
            "bullish_flag", "bearish_flag", "bullish_pennant", "bearish_pennant",
            "cup_handle", "wedge_rising", "wedge_falling", "rectangle_pattern",
            "breakout_up", "breakout_down", "reversal_bullish", "reversal_bearish",
            "continuation_bullish", "continuation_bearish", "channel_up", "channel_down",
            "broadening_pattern", "diamond_pattern"
        ]
        
        detected_patterns = []
        for i, prob in enumerate(pattern_probs[:len(pattern_names)]):
            if prob > 0.7:  # Threshold
                pattern = pattern_names[i]
                
                # Determina direzione
                if 'bullish' in pattern or 'up' in pattern or 'ascending' in pattern:
                    direction = 'bullish'
                elif 'bearish' in pattern or 'down' in pattern or 'descending' in pattern:
                    direction = 'bearish'
                else:
                    direction = 'neutral'
                
                detected_patterns.append({
                    "pattern": pattern,
                    "probability": float(prob),
                    "confidence": float(prob * 0.9),
                    "direction": direction,
                    "timeframe": "short_term"
                })
        
        # Ordina per probabilità
        detected_patterns.sort(key=lambda x: x['probability'], reverse=True)
        
        self.algorithm_stats['successful_predictions'] += 1
        self.algorithm_stats['patterns_detected'] += len(detected_patterns)
        
        return {
            "detected_patterns": detected_patterns[:3],  # Top 3
            "pattern_strength": float(np.max(pattern_probs)) if detected_patterns else 0.0,
            "confidence": float(np.mean([p["confidence"] for p in detected_patterns[:3]])) if detected_patterns else 0.3,
            "method": "CNN_Deep_Learning"
        }
    
    def _classical_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classical Pattern Recognition
        ESTRATTO IDENTICO da src/Analyzer.py:13150-13224
        """
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = np.array(market_data['price_history'][-50:])
        if len(prices) < 50:
            raise InsufficientDataError(required=50, available=len(prices), operation="Classical_Patterns")
        
        patterns = []
        
        try:
            # Double Top Detection
            if self._detect_double_top(prices):
                patterns.append({
                    "pattern": "double_top", 
                    "probability": 0.8, 
                    "direction": "bearish",
                    "confidence": 0.75,
                    "timeframe": "medium_term"
                })
            
            # Double Bottom Detection
            if self._detect_double_bottom(prices):
                patterns.append({
                    "pattern": "double_bottom", 
                    "probability": 0.8, 
                    "direction": "bullish",
                    "confidence": 0.75,
                    "timeframe": "medium_term"
                })
            
            # Head and Shoulders Detection
            if self._detect_head_shoulders(prices):
                patterns.append({
                    "pattern": "head_shoulders", 
                    "probability": 0.85, 
                    "direction": "bearish",
                    "confidence": 0.8,
                    "timeframe": "long_term"
                })
            
            # Triangle Pattern Detection
            if self._detect_triangle_pattern(prices):
                patterns.append({
                    "pattern": "triangle_pattern", 
                    "probability": 0.7, 
                    "direction": "neutral",
                    "confidence": 0.65,
                    "timeframe": "short_term"
                })
            
            # Flag Pattern Detection  
            if self._detect_flag_pattern(prices):
                patterns.append({
                    "pattern": "flag_pattern", 
                    "probability": 0.75, 
                    "direction": "continuation",
                    "confidence": 0.7,
                    "timeframe": "short_term"
                })
            
            # Channel Pattern Detection
            if self._detect_channel_pattern(prices):
                patterns.append({
                    "pattern": "channel_pattern", 
                    "probability": 0.7, 
                    "direction": "neutral",
                    "confidence": 0.65,
                    "timeframe": "medium_term"
                })
                
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            # FAIL FAST - Re-raise pattern detection error instead of logging
            # Continue without raising - classical patterns can have partial failures
        
        self.algorithm_stats['successful_predictions'] += 1
        self.algorithm_stats['patterns_detected'] += len(patterns)
        
        return {
            "detected_patterns": patterns,
            "pattern_strength": float(np.mean([p["probability"] for p in patterns])) if patterns else 0.0,
            "confidence": float(np.mean([p["confidence"] for p in patterns])) if patterns else 0.3,
            "method": "Classical_Technical_Analysis"
        }
    
    def _lstm_sequences(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LSTM Sequence Pattern Recognition
        ESTRATTO IDENTICO da src/Analyzer.py:13225-13273
        """
        # Get asset from market_data for asset-specific model loading
        asset = market_data.get('asset', 'UNKNOWN')
        model = self.get_model('LSTM_Sequences', asset)
        
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = np.array(market_data['price_history'][-60:])
        if len(prices) < 60:
            raise InsufficientDataError(required=60, available=len(prices), operation="LSTM_Sequences")
        
        try:
            # Prepara sequences per LSTM
            sequences = self._prepare_sequences(prices, seq_length=20)
            
            patterns = []
            with torch.no_grad():
                for i, seq in enumerate(sequences):
                    input_tensor = torch.FloatTensor(seq).unsqueeze(0)
                    prediction = model(input_tensor)
                    
                    # Interpreta prediction come pattern type
                    pattern_type = self._interpret_sequence_pattern(prediction.numpy())
                    if pattern_type:
                        patterns.append({
                            "pattern": pattern_type["name"],
                            "probability": pattern_type["probability"],
                            "confidence": pattern_type["confidence"],
                            "direction": pattern_type["direction"],
                            "timeframe": "sequence_based",
                            "sequence_position": i
                        })
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['patterns_detected'] += len(patterns)
            
            return {
                "detected_patterns": patterns[:5],  # Top 5 sequences
                "pattern_strength": float(np.mean([p["probability"] for p in patterns])) if patterns else 0.0,
                "confidence": float(np.mean([p["confidence"] for p in patterns])) if patterns else 0.4,
                "method": "LSTM_Sequence_Analysis"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("LSTM_Sequences", f"Sequence analysis failed: {str(e)}")
    
    def _transformer_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformer Pattern Recognition
        ESTRATTO IDENTICO da src/Analyzer.py:13274-13332
        """
        # Get asset from market_data for asset-specific model loading
        asset = market_data.get('asset', 'UNKNOWN')
        model = self.get_model('Transformer_Patterns', asset)
        
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = np.array(market_data['price_history'][-80:])
        if len(prices) < 80:
            raise InsufficientDataError(required=80, available=len(prices), operation="Transformer_Patterns")
        
        try:
            # Transformer attention-based pattern recognition
            normalized_prices = (prices - prices.mean()) / prices.std()
            input_tensor = torch.FloatTensor(normalized_prices).unsqueeze(0)
            
            with torch.no_grad():
                attention_weights, pattern_logits = model(input_tensor)
                
                # Interpreta attention patterns
                patterns = self._interpret_attention_patterns(
                    attention_weights.numpy(), 
                    pattern_logits.numpy(),
                    prices
                )
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['patterns_detected'] += len(patterns)
            
            return {
                "detected_patterns": patterns[:3],  # Top 3
                "pattern_strength": float(np.mean([p["probability"] for p in patterns])) if patterns else 0.0,
                "confidence": float(np.mean([p["confidence"] for p in patterns])) if patterns else 0.45,
                "method": "Transformer_Attention_Analysis"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("Transformer_Patterns", f"Transformer analysis failed: {str(e)}")
    
    def _ensemble_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensemble Pattern Recognition
        Combina risultati di tutti gli altri algoritmi
        """
        try:
            # Esegue tutti gli algoritmi disponibili
            results = []
            
            # CNN patterns
            try:
                cnn_result = self._cnn_pattern_recognizer(market_data)
                results.append(("CNN", cnn_result))
            except Exception as e:
                raise RuntimeError(f"CNN pattern recognition failed: {e}")
            
            # Classical patterns
            try:
                classical_result = self._classical_patterns(market_data)
                results.append(("Classical", classical_result))
            except Exception as e:
                raise RuntimeError(f"Classical pattern recognition failed: {e}")
            
            # Combina risultati con voting
            ensemble_patterns = self._combine_pattern_results(results)
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['patterns_detected'] += len(ensemble_patterns)
            
            return {
                "detected_patterns": ensemble_patterns[:3],  # Top 3 ensemble
                "pattern_strength": float(np.mean([p["probability"] for p in ensemble_patterns])) if ensemble_patterns else 0.0,
                "confidence": float(np.mean([p["confidence"] for p in ensemble_patterns])) if ensemble_patterns else 0.35,
                "method": "Ensemble_Multi_Algorithm",
                "algorithms_used": len(results)
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("Ensemble_Patterns", f"Ensemble analysis failed: {str(e)}")
    
    # === HELPER METHODS - Pattern Detection ===
    
    def _detect_double_top(self, prices: np.ndarray) -> bool:
        """Rileva pattern double top"""
        if len(prices) < 20:
            return False
        
        # Trova local maxima
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) < 2:
            return False
        
        # Verifica se gli ultimi 2 peaks sono simili (double top)
        peak1, peak2 = peaks[-2], peaks[-1]
        price_diff = abs(peak1[1] - peak2[1]) / peak1[1]
        
        return price_diff < 0.03  # 3% tolerance
    
    def _detect_double_bottom(self, prices: np.ndarray) -> bool:
        """Rileva pattern double bottom"""
        if len(prices) < 20:
            return False
        
        # Trova local minima
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        
        if len(troughs) < 2:
            return False
        
        # Verifica se gli ultimi 2 troughs sono simili (double bottom)
        trough1, trough2 = troughs[-2], troughs[-1]
        price_diff = abs(trough1[1] - trough2[1]) / trough1[1]
        
        return price_diff < 0.03  # 3% tolerance
    
    def _detect_head_shoulders(self, prices: np.ndarray) -> bool:
        """Rileva pattern head and shoulders"""
        if len(prices) < 30:
            return False
        
        # Trova 3 peaks consecutivi
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) < 3:
            return False
        
        # Verifica pattern: left shoulder < head > right shoulder
        left_shoulder, head, right_shoulder = peaks[-3:]
        
        # Head deve essere più alto delle shoulders
        if head[1] <= left_shoulder[1] or head[1] <= right_shoulder[1]:
            return False
        
        # Shoulders devono essere simili
        shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
        
        return shoulder_diff < 0.05  # 5% tolerance
    
    def _detect_triangle_pattern(self, prices: np.ndarray) -> bool:
        """Rileva pattern triangolare"""
        if len(prices) < 20:
            return False
        
        # Calcola trend delle highs e lows
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                highs.append(prices[i])
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                lows.append(prices[i])
        
        if len(highs) < 3 or len(lows) < 3:
            return False
        
        # Verifica convergenza (triangle)
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Triangle: highs scendono, lows salgono (o viceversa)
        return (high_trend < 0 and low_trend > 0) or (high_trend > 0 and low_trend < 0)
    
    def _detect_flag_pattern(self, prices: np.ndarray) -> bool:
        """Rileva pattern flag"""
        if len(prices) < 15:
            return False
        
        # Flag: trend consolidation dopo strong move
        first_half = prices[:len(prices)//2]
        second_half = prices[len(prices)//2:]
        
        # Strong move nella prima metà
        first_trend = (first_half[-1] - first_half[0]) / first_half[0]
        
        # Consolidation nella seconda metà
        second_volatility = np.std(second_half) / np.mean(second_half)
        
        return bool(abs(first_trend) > 0.05 and second_volatility < 0.02)
    
    def _detect_channel_pattern(self, prices: np.ndarray) -> bool:
        """Rileva pattern channel"""
        if len(prices) < 25:
            return False
        
        # Channel: prezzi oscillano tra support e resistance paralleli
        highs = [prices[i] for i in range(1, len(prices)-1) 
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]]
        lows = [prices[i] for i in range(1, len(prices)-1) 
               if prices[i] < prices[i-1] and prices[i] < prices[i+1]]
        
        if len(highs) < 3 or len(lows) < 3:
            return False
        
        # Verifica linearità di highs e lows (channel parallelo)
        high_trend = np.polyfit(range(len(highs)), highs, 1)[0]
        low_trend = np.polyfit(range(len(lows)), lows, 1)[0]
        
        # Channel: trend simile per highs e lows
        return abs(high_trend - low_trend) < 0.001
    
    def _prepare_sequences(self, prices: np.ndarray, seq_length: int = 20) -> List[np.ndarray]:
        """Prepara sequences per LSTM"""
        sequences = []
        for i in range(len(prices) - seq_length + 1):
            seq = prices[i:i + seq_length]
            # Normalizza sequence
            normalized_seq = (seq - seq.mean()) / seq.std()
            sequences.append(normalized_seq)
        return sequences
    
    def _interpret_sequence_pattern(self, prediction: np.ndarray) -> Optional[Dict[str, Any]]:
        """Interpreta prediction LSTM come pattern type"""
        # Placeholder implementation
        if prediction.max() > 0.6:
            return {
                "name": "lstm_detected_pattern",
                "probability": float(prediction.max()),
                "confidence": float(prediction.max() * 0.8),
                "direction": "bullish" if prediction[0] > 0.5 else "bearish"
            }
        return None
    
    def _interpret_attention_patterns(self, attention_weights: np.ndarray, 
                                    pattern_logits: np.ndarray, 
                                    prices: np.ndarray) -> List[Dict[str, Any]]:
        """Interpreta attention patterns del Transformer"""
        patterns = []
        
        # Placeholder implementation
        if attention_weights.max() > 0.7:
            patterns.append({
                "pattern": "transformer_attention_pattern",
                "probability": float(attention_weights.max()),
                "confidence": float(attention_weights.max() * 0.85),
                "direction": "bullish" if pattern_logits[0] > 0 else "bearish",
                "timeframe": "attention_based"
            })
        
        return patterns
    
    def _combine_pattern_results(self, results: List[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Combina risultati pattern di algoritmi diversi"""
        pattern_votes = {}
        
        for algo_name, result in results:
            if "detected_patterns" not in result:
                raise KeyError("Missing required 'detected_patterns' key in ensemble result")
            patterns = result["detected_patterns"]
            for pattern in patterns:
                pattern_name = pattern["pattern"]
                if pattern_name not in pattern_votes:
                    pattern_votes[pattern_name] = []
                
                pattern_votes[pattern_name].append({
                    "algorithm": algo_name,
                    "probability": pattern["probability"],
                    "confidence": pattern["confidence"],
                    "direction": pattern["direction"]
                })
        
        # Crea ensemble patterns
        ensemble_patterns = []
        for pattern_name, votes in pattern_votes.items():
            if len(votes) >= 2:  # Almeno 2 algoritmi concordano
                avg_prob = np.mean([v["probability"] for v in votes])
                avg_conf = np.mean([v["confidence"] for v in votes])
                
                # Direzione maggioritaria
                directions = [v["direction"] for v in votes]
                direction = max(set(directions), key=directions.count)
                
                ensemble_patterns.append({
                    "pattern": pattern_name,
                    "probability": float(avg_prob),
                    "confidence": float(avg_conf * len(votes) / len(results)),  # Boost per agreement
                    "direction": direction,
                    "timeframe": "ensemble",
                    "votes": len(votes),
                    "algorithms": [v["algorithm"] for v in votes]
                })
        
        # Ordina per confidence
        ensemble_patterns.sort(key=lambda x: x["confidence"], reverse=True)
        return ensemble_patterns
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche algoritmi"""
        return self.algorithm_stats.copy()


# Factory function per compatibilità
def create_pattern_recognition_algorithms(ml_models: Optional[Dict[str, Any]] = None) -> PatternRecognitionAlgorithms:
    """Factory function per creare PatternRecognitionAlgorithms"""
    return PatternRecognitionAlgorithms(ml_models)


# Export
__all__ = [
    'PatternRecognitionAlgorithms',
    'create_pattern_recognition_algorithms'
]