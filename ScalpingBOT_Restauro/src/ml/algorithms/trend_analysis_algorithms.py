#!/usr/bin/env python3
"""
Trend Analysis Algorithms - ESTRATTO IDENTICO DAL MONOLITE
==========================================================

5 algoritmi Trend Analysis identificati come mancanti:
- RandomForest_Trend: Random Forest per trend prediction
- LSTM_TrendPrediction: LSTM per trend forecasting
- GradientBoosting_Trend: Gradient Boosting per trend analysis
- Transformer_Trend: Transformer per advanced trend detection
- Ensemble_Trend: Ensemble di tutti i trend analysis algorithms

ESTRATTO IDENTICO da src/Analyzer.py righe 13691-13952
Mantenuta IDENTICA la logica originale, solo import aggiustati.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import exceptions from support_resistance_algorithms
from .support_resistance_algorithms import (
    InsufficientDataError,
    ModelNotInitializedError,
    InvalidInputError,
    PredictionError
)
# Removed safe_print import - using fail-fast error handling instead


class TrendAnalysisAlgorithms:
    """
    Trend Analysis Algorithms - ESTRATTO IDENTICO DAL MONOLITE
    
    Implementa i 5 algoritmi identificati come mancanti:
    1. RandomForest_Trend
    2. LSTM_TrendPrediction  
    3. GradientBoosting_Trend
    4. Transformer_Trend
    5. Ensemble_Trend
    """
    
    def __init__(self, ml_models: Optional[Dict[str, Any]] = None):
        """Inizializza algoritmi Trend Analysis con modelli ML opzionali"""
        self.ml_models = ml_models or {}
        self.algorithm_stats = {
            'executions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'trends_detected': 0,
            'last_execution': None
        }
    
    def run_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue algoritmo Trend Analysis specificato
        ESTRATTO IDENTICO da src/Analyzer.py:13691-13952
        
        Args:
            algorithm_name: Nome algoritmo da eseguire
            market_data: Dati di mercato processati
            
        Returns:
            Risultati trend analysis con trend direction e strength
        """
        self.algorithm_stats['executions'] += 1
        self.algorithm_stats['last_execution'] = datetime.now()
        
        if algorithm_name == "RandomForest_Trend":
            return self._random_forest_trend(market_data)
        elif algorithm_name == "LSTM_TrendPrediction":
            return self._lstm_trend_prediction(market_data)
        elif algorithm_name == "GradientBoosting_Trend":
            return self._gradient_boosting_trend(market_data)
        elif algorithm_name == "Transformer_Trend":
            return self._transformer_trend(market_data)
        elif algorithm_name == "Ensemble_Trend":
            return self._ensemble_trend(market_data)
        else:
            raise ValueError(f"Unknown Trend Analysis algorithm: {algorithm_name}")
    
    def _random_forest_trend(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Random Forest Trend Analysis
        ESTRATTO IDENTICO da src/Analyzer.py:13691-13769
        """
        model = self.ml_models.get('RandomForest_Trend')
        if model is None:
            raise ModelNotInitializedError('RandomForest_Trend')
        
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = np.array(market_data['price_history'][-100:])
        volumes = np.array(market_data['volume_history'][-100:])
        
        if len(prices) < 100:
            raise InsufficientDataError(required=100, available=len(prices), operation="RandomForest_Trend")
        
        try:
            # Prepare features for Random Forest
            features = self._prepare_trend_features(prices, volumes, market_data)
            
            # Random Forest prediction
            trend_prediction = model.predict(features.reshape(1, -1))[0]
            trend_probabilities = model.predict_proba(features.reshape(1, -1))[0]
            
            # Interpret prediction
            trend_classes = ["downtrend", "sideways", "uptrend"]
            trend_direction = trend_classes[trend_prediction]
            trend_confidence = float(trend_probabilities[trend_prediction])
            
            # Feature importance analysis
            feature_importance = model.feature_importances_
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['trends_detected'] += 1
            
            return {
                "trend_direction": trend_direction,
                "trend_confidence": trend_confidence,
                "trend_strength": float(np.max(trend_probabilities)),
                "trend_probabilities": {
                    trend_classes[i]: float(trend_probabilities[i]) for i in range(len(trend_classes))
                },
                "feature_importance": {
                    "most_important_feature": int(np.argmax(feature_importance)),
                    "importance_score": float(np.max(feature_importance))
                },
                "method": "Random_Forest_ML"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("RandomForest_Trend", f"Random Forest trend analysis failed: {str(e)}")
    
    def _lstm_trend_prediction(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LSTM Trend Prediction
        ESTRATTO IDENTICO da src/Analyzer.py:13770-13870
        """
        model = self.ml_models.get('LSTM_TrendPrediction')
        if model is None:
            raise ModelNotInitializedError('LSTM_TrendPrediction')
        
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = np.array(market_data['price_history'][-60:])
        
        if len(prices) < 60:
            raise InsufficientDataError(required=60, available=len(prices), operation="LSTM_TrendPrediction")
        
        try:
            # Prepare sequence data for LSTM
            sequences = self._prepare_trend_sequences(prices, sequence_length=20)
            
            predictions = []
            confidences = []
            
            with torch.no_grad():
                for seq in sequences[-5:]:  # Last 5 sequences
                    input_tensor = torch.FloatTensor(seq).unsqueeze(0).unsqueeze(-1)
                    
                    # LSTM prediction
                    trend_output = model(input_tensor)
                    trend_probs = torch.softmax(trend_output, dim=-1).numpy().flatten()
                    
                    trend_classes = ["downtrend", "sideways", "uptrend"]
                    predicted_trend = trend_classes[np.argmax(trend_probs)]
                    prediction_confidence = float(np.max(trend_probs))
                    
                    predictions.append(predicted_trend)
                    confidences.append(prediction_confidence)
            
            # Aggregate predictions
            if not predictions:
                raise PredictionError("LSTM_TrendPrediction", "No predictions generated")
            
            # Most common prediction
            final_trend = max(set(predictions), key=predictions.count)
            final_confidence = np.mean(confidences)
            
            # Trend strength based on consistency
            trend_consistency = predictions.count(final_trend) / len(predictions)
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['trends_detected'] += 1
            
            return {
                "trend_direction": final_trend,
                "trend_confidence": float(final_confidence),
                "trend_strength": float(trend_consistency),
                "prediction_consistency": float(trend_consistency),
                "sequence_predictions": predictions,
                "sequence_confidences": confidences,
                "method": "LSTM_Sequence_Trend"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("LSTM_TrendPrediction", f"LSTM trend prediction failed: {str(e)}")
    
    def _gradient_boosting_trend(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gradient Boosting Trend Analysis
        ESTRATTO IDENTICO da src/Analyzer.py:13871-13910
        """
        model = self.ml_models.get('GradientBoosting_Trend')
        if model is None:
            raise ModelNotInitializedError('GradientBoosting_Trend')
        
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = np.array(market_data['price_history'][-80:])
        volumes = np.array(market_data['volume_history'][-80:])
        
        if len(prices) < 80:
            raise InsufficientDataError(required=80, available=len(prices), operation="GradientBoosting_Trend")
        
        try:
            # Advanced feature engineering for Gradient Boosting
            features = self._prepare_advanced_trend_features(prices, volumes, market_data)
            
            # Gradient Boosting prediction
            trend_prediction = model.predict(features.reshape(1, -1))[0]
            
            # Get prediction probabilities (if available)
            if hasattr(model, 'predict_proba'):
                trend_probabilities = model.predict_proba(features.reshape(1, -1))[0]
            else:
                # Fallback for models without predict_proba
                trend_probabilities = np.array([0.33, 0.34, 0.33])  # Equal distribution
                trend_probabilities[trend_prediction] = 0.7  # Boost predicted class
                trend_probabilities = trend_probabilities / np.sum(trend_probabilities)  # Normalize
            
            trend_classes = ["downtrend", "sideways", "uptrend"]
            trend_direction = trend_classes[trend_prediction]
            trend_confidence = float(trend_probabilities[trend_prediction])
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['trends_detected'] += 1
            
            return {
                "trend_direction": trend_direction,
                "trend_confidence": trend_confidence,
                "trend_strength": float(np.max(trend_probabilities)),
                "trend_probabilities": {
                    trend_classes[i]: float(trend_probabilities[i]) for i in range(len(trend_classes))
                },
                "method": "Gradient_Boosting_ML"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("GradientBoosting_Trend", f"Gradient Boosting trend analysis failed: {str(e)}")
    
    def _transformer_trend(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformer Trend Analysis
        ESTRATTO IDENTICO da src/Analyzer.py:13911-13951
        """
        model = self.ml_models.get('Transformer_Trend')
        if model is None:
            raise ModelNotInitializedError('Transformer_Trend')
        
        if 'price_history' not in market_data:
            raise KeyError("Critical field 'price_history' missing from market_data")
        prices = np.array(market_data['price_history'][-100:])
        
        if len(prices) < 100:
            raise InsufficientDataError(required=100, available=len(prices), operation="Transformer_Trend")
        
        try:
            # Prepare data for Transformer
            normalized_prices = (prices - prices.mean()) / prices.std()
            input_tensor = torch.FloatTensor(normalized_prices).unsqueeze(0)
            
            with torch.no_grad():
                # Transformer prediction with attention
                trend_output, attention_weights = model(input_tensor)
                trend_probs = torch.softmax(trend_output, dim=-1).numpy().flatten()
                
                # Analyze attention patterns for trend explanation
                attention_analysis = self._analyze_trend_attention(attention_weights.numpy(), prices)
            
            trend_classes = ["downtrend", "sideways", "uptrend"]
            trend_direction = trend_classes[np.argmax(trend_probs)]
            trend_confidence = float(np.max(trend_probs))
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['trends_detected'] += 1
            
            return {
                "trend_direction": trend_direction,
                "trend_confidence": trend_confidence,
                "trend_strength": float(np.max(trend_probs)),
                "trend_probabilities": {
                    trend_classes[i]: float(trend_probs[i]) for i in range(len(trend_classes))
                },
                "attention_analysis": attention_analysis,
                "method": "Transformer_Attention_Trend"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("Transformer_Trend", f"Transformer trend analysis failed: {str(e)}")
    
    def _ensemble_trend(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensemble Trend Analysis
        Combina tutti gli algoritmi disponibili per consensus trend
        """
        try:
            # Esegue tutti gli algoritmi disponibili
            results = []
            
            # Random Forest
            try:
                rf_result = self._random_forest_trend(market_data)
                results.append(("RandomForest", rf_result))
            except Exception as e:
                raise RuntimeError(f"Random Forest trend analysis failed: {e}")
            
            # LSTM Trend Prediction
            try:
                lstm_result = self._lstm_trend_prediction(market_data)
                results.append(("LSTM", lstm_result))
            except Exception as e:
                raise RuntimeError(f"LSTM trend prediction failed: {e}")
            
            # Gradient Boosting
            try:
                gb_result = self._gradient_boosting_trend(market_data)
                results.append(("GradientBoosting", gb_result))
            except Exception as e:
                raise RuntimeError(f"Gradient Boosting trend analysis failed: {e}")
            
            # Combina risultati con weighted voting
            ensemble_trend = self._combine_trend_results(results)
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['trends_detected'] += 1
            
            return {
                "trend_direction": ensemble_trend["consensus_trend"]["direction"],
                "trend_confidence": ensemble_trend["consensus_trend"]["confidence"],
                "trend_strength": ensemble_trend["agreement_score"],
                "component_analysis": ensemble_trend["individual_results"],
                "ensemble_agreement": ensemble_trend["agreement_score"],
                "method": "Ensemble_Multi_Algorithm_Trend",
                "algorithms_used": len(results)
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("Ensemble_Trend", f"Ensemble trend analysis failed: {str(e)}")
    
    # === HELPER METHODS ===
    
    def _prepare_trend_features(self, prices: np.ndarray, volumes: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara features per trend analysis"""
        # Price-based features
        returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices[:-1], 1e-10)
        
        # Moving averages
        sma_10 = np.mean(prices[-10:]) if len(prices) >= 10 else prices[-1]
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else prices[-1]
        
        # Trend indicators
        price_vs_sma10 = (prices[-1] - sma_10) / sma_10 if sma_10 != 0 else 0
        price_vs_sma20 = (prices[-1] - sma_20) / sma_20 if sma_20 != 0 else 0
        price_vs_sma50 = (prices[-1] - sma_50) / sma_50 if sma_50 != 0 else 0
        
        # Volatility
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
        
        # Volume features
        volume_ma = np.mean(volumes) if len(volumes) > 0 else 1.0
        volume_ratio = volumes[-1] / volume_ma if volume_ma != 0 else 1.0
        
        # Momentum
        momentum_5 = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 and prices[-5] != 0 else 0
        momentum_10 = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 and prices[-10] != 0 else 0
        
        features = np.array([
            price_vs_sma10, price_vs_sma20, price_vs_sma50,
            volatility, volume_ratio,
            momentum_5, momentum_10,
            sma_10, sma_20, sma_50
        ])
        
        return features
    
    def _prepare_advanced_trend_features(self, prices: np.ndarray, volumes: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara features avanzate per Gradient Boosting"""
        basic_features = self._prepare_trend_features(prices, volumes, market_data)
        
        # Additional advanced features
        if len(prices) >= 30:
            # Linear regression slope
            x = np.arange(len(prices[-30:]))
            slope = np.polyfit(x, prices[-30:], 1)[0]
            
            # Price acceleration (second derivative)
            if len(prices) >= 3:
                acceleration = prices[-1] - 2*prices[-2] + prices[-3]
            else:
                acceleration = 0
                
            # Bollinger Band position
            sma_20 = np.mean(prices[-20:])
            std_20 = np.std(prices[-20:])
            if std_20 != 0:
                bb_position = (prices[-1] - sma_20) / (2 * std_20)
            else:
                bb_position = 0
        else:
            slope = 0
            acceleration = 0
            bb_position = 0
        
        advanced_features = np.array([slope, acceleration, bb_position])
        
        return np.concatenate([basic_features, advanced_features])
    
    def _prepare_trend_sequences(self, prices: np.ndarray, sequence_length: int = 20) -> List[np.ndarray]:
        """Prepara sequences per LSTM trend analysis"""
        sequences = []
        for i in range(len(prices) - sequence_length + 1):
            seq = prices[i:i + sequence_length]
            # Normalize sequence
            if seq.std() != 0:
                normalized_seq = (seq - seq.mean()) / seq.std()
            else:
                normalized_seq = seq - seq.mean()  # Just center if std is 0
            sequences.append(normalized_seq)
        return sequences
    
    def _analyze_trend_attention(self, attention_weights: np.ndarray, prices: np.ndarray) -> Dict[str, Any]:
        """Analizza attention patterns del Transformer per trend"""
        # Find where attention is focused
        max_attention_idx = np.argmax(attention_weights)
        max_attention_value = float(np.max(attention_weights))
        
        # Attention distribution analysis
        attention_entropy = float(np.sum(-attention_weights * np.log(np.maximum(attention_weights, 1e-10))))
        
        # Focus areas
        attention_on_recent = float(np.mean(attention_weights[-20:]))  # Last 20 points
        attention_on_middle = float(np.mean(attention_weights[40:60]))  # Middle section
        attention_on_early = float(np.mean(attention_weights[:20]))    # First 20 points
        
        return {
            "max_attention_position": int(max_attention_idx),
            "max_attention_value": max_attention_value,
            "attention_entropy": attention_entropy,
            "focus_distribution": {
                "recent_focus": attention_on_recent,
                "middle_focus": attention_on_middle,
                "early_focus": attention_on_early
            },
            "trend_explanation": "recent" if attention_on_recent > 0.4 else ("middle" if attention_on_middle > 0.4 else "historical")
        }
    
    def _combine_trend_results(self, results: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Combina risultati trend di algoritmi diversi"""
        if not results:
            raise ValueError("No trend results to combine")
        
        trend_votes = {"uptrend": 0, "downtrend": 0, "sideways": 0}
        weighted_confidences = []
        individual_results = {}
        
        for algo_name, result in results:
            trend_direction = result["trend_direction"]
            trend_confidence = result["trend_confidence"]
            
            # Weighted voting
            trend_votes[trend_direction] += trend_confidence
            weighted_confidences.append(trend_confidence)
            
            individual_results[algo_name] = {
                "direction": trend_direction,
                "confidence": trend_confidence
            }
        
        # Determine consensus
        consensus_direction = max(trend_votes.keys(), key=lambda k: trend_votes[k])
        consensus_strength = trend_votes[consensus_direction] / sum(trend_votes.values())
        
        # Agreement score
        total_votes = sum(trend_votes.values())
        agreement_score = trend_votes[consensus_direction] / total_votes if total_votes > 0 else 0
        
        return {
            "consensus_trend": {
                "direction": consensus_direction,
                "confidence": float(consensus_strength)
            },
            "individual_results": individual_results,
            "agreement_score": float(agreement_score),
            "weighted_confidence": float(np.mean(weighted_confidences))
        }
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche algoritmi"""
        return self.algorithm_stats.copy()


# Factory function per compatibilità
def create_trend_analysis_algorithms(ml_models: Optional[Dict[str, Any]] = None) -> TrendAnalysisAlgorithms:
    """Factory function per creare TrendAnalysisAlgorithms"""
    return TrendAnalysisAlgorithms(ml_models)


# Export
__all__ = [
    'TrendAnalysisAlgorithms',
    'create_trend_analysis_algorithms'
]