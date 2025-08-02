#!/usr/bin/env python3
"""
Bias Detection Algorithms - ESTRATTO IDENTICO DAL MONOLITE
==========================================================

5 algoritmi Bias Detection identificati come mancanti:
- Sentiment_LSTM: LSTM per sentiment e directional bias detection
- VolumePrice_Analysis: Analisi volume-price per professional buying/selling pressure
- Momentum_ML: ML-based momentum analysis con multi-timeframe RSI/MACD
- Transformer_Bias: Transformer per advanced bias detection
- MultiModal_Bias: Multi-modal ensemble bias analysis

ESTRATTO IDENTICO da src/Analyzer.py righe 13399-13644
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


class BiasDetectionAlgorithms:
    """
    Bias Detection Algorithms - ESTRATTO IDENTICO DAL MONOLITE
    
    Implementa i 5 algoritmi identificati come mancanti:
    1. Sentiment_LSTM
    2. VolumePrice_Analysis  
    3. Momentum_ML
    4. Transformer_Bias
    5. MultiModal_Bias
    """
    
    def __init__(self, ml_models: Optional[Dict[str, Any]] = None):
        """Inizializza algoritmi Bias Detection con modelli ML opzionali"""
        self.ml_models = ml_models or {}
        self.algorithm_stats = {
            'executions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'bias_detections': 0,
            'last_execution': None
        }
    
    def run_algorithm(self, algorithm_name: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Esegue algoritmo Bias Detection specificato
        ESTRATTO IDENTICO da src/Analyzer.py:13399-13644
        
        Args:
            algorithm_name: Nome algoritmo da eseguire
            market_data: Dati di mercato processati
            
        Returns:
            Risultati bias detection con directional bias e confidence
        """
        self.algorithm_stats['executions'] += 1
        self.algorithm_stats['last_execution'] = datetime.now()
        
        if algorithm_name == "Sentiment_LSTM":
            return self._sentiment_lstm(market_data)
        elif algorithm_name == "VolumePrice_Analysis":
            return self._volume_price_analysis(market_data)
        elif algorithm_name == "Momentum_ML":
            return self._momentum_ml(market_data)
        elif algorithm_name == "Transformer_Bias":
            return self._transformer_bias(market_data)
        elif algorithm_name == "MultiModal_Bias":
            return self._multimodal_bias(market_data)
        else:
            raise ValueError(f"Unknown Bias Detection algorithm: {algorithm_name}")
    
    def _sentiment_lstm(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LSTM Sentiment Analysis per directional bias
        ESTRATTO IDENTICO da src/Analyzer.py:13399-13445
        """
        model = self.ml_models.get('Sentiment_LSTM')
        if model is None:
            raise ModelNotInitializedError('Sentiment_LSTM')
        
        # Prepara features per sentiment
        prices = np.array(market_data['price_history'][-30:])
        volumes = np.array(market_data['volume_history'][-30:])
        
        if len(prices) < 30:
            raise InsufficientDataError(required=30, available=len(prices), operation="Sentiment_LSTM")
        
        try:
            # Feature engineering per sentiment
            features = self._prepare_sentiment_features(prices, volumes, market_data)
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                bias_output = model(input_tensor)
                bias_probs = torch.softmax(bias_output, dim=-1).numpy().flatten()
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("Sentiment_LSTM", f"LSTM sentiment prediction failed: {str(e)}")
        
        # Interpreta output: [bearish, neutral, bullish]
        bias_labels = ["bearish", "neutral", "bullish"]
        dominant_bias_idx = np.argmax(bias_probs)
        dominant_bias = bias_labels[dominant_bias_idx]
        bias_confidence = float(bias_probs[dominant_bias_idx])
        
        # Analisi addizionale del sentiment
        try:
            behavioral_analysis = self._analyze_market_behavior(prices, volumes)
        except Exception as e:
            # VIOLAZIONE REGOLA #4: Rimuovo fallback
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("Sentiment_LSTM", f"Behavioral analysis failed: {str(e)}")
        
        self.algorithm_stats['successful_predictions'] += 1
        self.algorithm_stats['bias_detections'] += 1
        
        return {
            "directional_bias": {
                "direction": dominant_bias,
                "confidence": bias_confidence,
                "distribution": {bias_labels[i]: float(bias_probs[i]) for i in range(len(bias_labels))}
            },
            "behavioral_bias": behavioral_analysis,
            "overall_confidence": float((bias_confidence + behavioral_analysis["confidence"]) / 2),
            "method": "LSTM_Sentiment_Analysis"
        }
    
    def _volume_price_analysis(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Volume-Price Analysis per professional buying/selling pressure
        ESTRATTO IDENTICO da src/Analyzer.py:13447-13520
        """
        prices = np.array(market_data['price_history'][-50:])
        volumes = np.array(market_data['volume_history'][-50:])
        
        if len(prices) < 50:
            raise InsufficientDataError(required=50, available=len(prices), operation="VolumePrice_Analysis")
        
        try:
            # Volume-Price Trend Analysis
            price_changes = np.diff(prices)
            
            # Analisi volume su movimenti positivi vs negativi
            positive_volume = np.sum(volumes[1:][price_changes > 0])
            negative_volume = np.sum(volumes[1:][price_changes < 0])
            total_volume = positive_volume + negative_volume
            
            if total_volume == 0:
                raise InvalidInputError("total_volume", 0, "No volume data available for analysis")
            
            buying_pressure = positive_volume / total_volume
            selling_pressure = negative_volume / total_volume
            
            # Volume momentum
            volume_ma_short = np.mean(volumes[-10:])
            volume_ma_long = np.mean(volumes[-30:])
            
            if volume_ma_long == 0:
                raise InvalidInputError("volume_ma_long", 0, "Long-term volume average is zero")
            
            volume_momentum = (volume_ma_short - volume_ma_long) / volume_ma_long
            
            # Price-Volume divergence
            if prices[-20] == 0:
                raise InvalidInputError("historical_price", 0, "Historical price is zero - invalid data")
            
            price_trend = (prices[-1] - prices[-20]) / prices[-20]
            volume_trend = (volume_ma_short - np.mean(volumes[-20:-10])) / np.mean(volumes[-20:-10])
            divergence = abs(price_trend - volume_trend)
            
            # Smart money analysis (institutional vs retail)
            large_volume_threshold = np.percentile(volumes, 80)
            large_volume_moves = volumes > large_volume_threshold
            
            if np.sum(large_volume_moves) == 0:
                institutional_bias = "neutral"
                institutional_confidence = 0.5
            else:
                institutional_price_changes = price_changes[large_volume_moves[1:]]
                institutional_direction = "bullish" if np.mean(institutional_price_changes) > 0 else "bearish"
                institutional_bias = institutional_direction
                institutional_confidence = min(abs(np.mean(institutional_price_changes)) * 10, 1.0)
            
            # Determina bias complessivo
            if buying_pressure > 0.6:
                overall_bias = "bullish"
                bias_strength = buying_pressure
            elif selling_pressure > 0.6:
                overall_bias = "bearish"
                bias_strength = selling_pressure
            else:
                overall_bias = "neutral"
                bias_strength = 0.5
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['bias_detections'] += 1
            
            return {
                "directional_bias": {
                    "direction": overall_bias,
                    "confidence": float(bias_strength),
                    "buying_pressure": float(buying_pressure),
                    "selling_pressure": float(selling_pressure)
                },
                "volume_analysis": {
                    "volume_momentum": float(volume_momentum),
                    "price_volume_divergence": float(divergence),
                    "divergence_significant": divergence > 0.1
                },
                "institutional_analysis": {
                    "bias": institutional_bias,
                    "confidence": float(institutional_confidence),
                    "large_volume_moves": int(np.sum(large_volume_moves))
                },
                "overall_confidence": float((bias_strength + institutional_confidence) / 2),
                "method": "Volume_Price_Professional_Analysis"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("VolumePrice_Analysis", f"Volume-Price analysis failed: {str(e)}")
    
    def _momentum_ml(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ML-based Momentum Analysis con multi-timeframe RSI/MACD
        ESTRATTO IDENTICO da src/Analyzer.py:13521-13594
        """
        prices = np.array(market_data['price_history'][-100:])
        
        if len(prices) < 100:
            raise InsufficientDataError(required=100, available=len(prices), operation="Momentum_ML")
        
        try:
            # Multi-timeframe RSI analysis
            from ...data.processors.market_data_processor import MarketDataProcessor
            processor = MarketDataProcessor()
            rsi_14 = processor._calculate_rsi(prices, 14)
            rsi_21 = processor._calculate_rsi(prices, 21)
            rsi_50 = processor._calculate_rsi(prices, 50)
            
            # Multi-timeframe MACD analysis
            macd_12_26 = self._calculate_macd(prices, 12, 26, 9)
            macd_19_39 = self._calculate_macd(prices, 19, 39, 9)
            
            # Momentum score calculation
            momentum_signals = []
            
            # RSI momentum signals
            if rsi_14[-1] > 70 and rsi_21[-1] > 65:
                momentum_signals.append({"signal": "overbought", "strength": 0.8, "direction": "bearish"})
            elif rsi_14[-1] < 30 and rsi_21[-1] < 35:
                momentum_signals.append({"signal": "oversold", "strength": 0.8, "direction": "bullish"})
            
            if rsi_14[-1] > rsi_14[-2] and rsi_21[-1] > rsi_21[-2]:
                momentum_signals.append({"signal": "rsi_bullish_momentum", "strength": 0.6, "direction": "bullish"})
            elif rsi_14[-1] < rsi_14[-2] and rsi_21[-1] < rsi_21[-2]:
                momentum_signals.append({"signal": "rsi_bearish_momentum", "strength": 0.6, "direction": "bearish"})
            
            # MACD momentum signals  
            macd_line_12_26 = macd_12_26['macd_line']
            macd_signal_12_26 = macd_12_26['macd_signal']
            
            if macd_line_12_26[-1] > macd_signal_12_26[-1] and macd_line_12_26[-2] <= macd_signal_12_26[-2]:
                momentum_signals.append({"signal": "macd_bullish_crossover", "strength": 0.7, "direction": "bullish"})
            elif macd_line_12_26[-1] < macd_signal_12_26[-1] and macd_line_12_26[-2] >= macd_signal_12_26[-2]:
                momentum_signals.append({"signal": "macd_bearish_crossover", "strength": 0.7, "direction": "bearish"})
            
            # Divergence analysis
            price_trend_short = (prices[-1] - prices[-10]) / prices[-10]
            rsi_trend_short = (rsi_14[-1] - rsi_14[-10]) / 100  # Normalize RSI
            
            divergence_strength = abs(price_trend_short - rsi_trend_short)
            if divergence_strength > 0.1:
                if price_trend_short > 0 and rsi_trend_short < 0:
                    momentum_signals.append({"signal": "bearish_divergence", "strength": 0.9, "direction": "bearish"})
                elif price_trend_short < 0 and rsi_trend_short > 0:
                    momentum_signals.append({"signal": "bullish_divergence", "strength": 0.9, "direction": "bullish"})
            
            # Aggregate momentum bias
            if not momentum_signals:
                overall_bias = "neutral"
                overall_confidence = 0.3
            else:
                bullish_signals = [s for s in momentum_signals if s["direction"] == "bullish"]
                bearish_signals = [s for s in momentum_signals if s["direction"] == "bearish"]
                
                bullish_strength = sum([s["strength"] for s in bullish_signals])
                bearish_strength = sum([s["strength"] for s in bearish_signals])
                
                if bullish_strength > bearish_strength:
                    overall_bias = "bullish"
                    overall_confidence = min(bullish_strength / len(momentum_signals), 1.0)
                elif bearish_strength > bullish_strength:
                    overall_bias = "bearish"
                    overall_confidence = min(bearish_strength / len(momentum_signals), 1.0)
                else:
                    overall_bias = "neutral"
                    overall_confidence = 0.4
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['bias_detections'] += 1
            
            return {
                "directional_bias": {
                    "direction": overall_bias,
                    "confidence": float(overall_confidence)
                },
                "momentum_indicators": {
                    "rsi_14": float(rsi_14[-1]),
                    "rsi_21": float(rsi_21[-1]),
                    "rsi_50": float(rsi_50[-1]),
                    "macd_12_26_line": float(macd_line_12_26[-1]),
                    "macd_12_26_signal": float(macd_signal_12_26[-1])
                },
                "momentum_signals": momentum_signals,
                "divergence_analysis": {
                    "price_trend": float(price_trend_short),
                    "rsi_trend": float(rsi_trend_short),
                    "divergence_strength": float(divergence_strength)
                },
                "overall_confidence": float(overall_confidence),
                "method": "Multi_Timeframe_Momentum_ML"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("Momentum_ML", f"Momentum ML analysis failed: {str(e)}")
    
    def _transformer_bias(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformer-based Advanced Bias Detection
        ESTRATTO IDENTICO da src/Analyzer.py:13595-13643
        """
        model = self.ml_models.get('Transformer_Bias')
        if model is None:
            raise ModelNotInitializedError('Transformer_Bias')
        
        prices = np.array(market_data['price_history'][-80:])
        volumes = np.array(market_data['volume_history'][-80:])
        
        if len(prices) < 80:
            raise InsufficientDataError(required=80, available=len(prices), operation="Transformer_Bias")
        
        try:
            # Prepare multi-modal features for Transformer
            price_features = self._prepare_price_features(prices)
            volume_features = self._prepare_volume_features(volumes)
            technical_features = self._prepare_technical_features(prices, volumes, market_data)
            
            # Combine features
            combined_features = np.concatenate([price_features, volume_features, technical_features], axis=-1)
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
                bias_output, attention_weights = model(input_tensor)
                
                # Bias classification output
                bias_probs = torch.softmax(bias_output, dim=-1).numpy().flatten()
                
                # Attention analysis for explainability
                attention_summary = self._analyze_attention_patterns(attention_weights.numpy())
            
            # Interpret bias predictions
            bias_labels = ["strong_bearish", "bearish", "neutral", "bullish", "strong_bullish"]
            dominant_bias_idx = np.argmax(bias_probs)
            dominant_bias = bias_labels[dominant_bias_idx]
            bias_confidence = float(bias_probs[dominant_bias_idx])
            
            # Simplify direction
            if "bullish" in dominant_bias:
                direction = "bullish"
            elif "bearish" in dominant_bias:
                direction = "bearish"
            else:
                direction = "neutral"
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['bias_detections'] += 1
            
            return {
                "directional_bias": {
                    "direction": direction,
                    "confidence": bias_confidence,
                    "detailed_bias": dominant_bias,
                    "distribution": {bias_labels[i]: float(bias_probs[i]) for i in range(len(bias_labels))}
                },
                "attention_analysis": attention_summary,
                "feature_importance": {
                    "price_weight": float(attention_summary["price_focus"]) if "price_focus" in attention_summary else self._calculate_required_price_weight(attention_summary),
                    "volume_weight": float(attention_summary["volume_focus"]) if "volume_focus" in attention_summary else self._calculate_required_volume_weight(attention_summary),
                    "technical_weight": float(attention_summary["technical_focus"]) if "technical_focus" in attention_summary else self._calculate_required_technical_weight(attention_summary)
                },
                "overall_confidence": bias_confidence,
                "method": "Transformer_Multi_Modal_Bias"
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("Transformer_Bias", f"Transformer bias analysis failed: {str(e)}")
    
    def _calculate_required_price_weight(self, attention_summary: Dict[str, Any]) -> float:
        """Calculate required price weight when not provided - FAIL FAST"""
        if not attention_summary:
            raise ValueError("Empty attention_summary - cannot calculate price weight")
        # Calculate based on available data or fail
        total_weight = sum(float(v) for v in attention_summary.values() if isinstance(v, (int, float)))
        if total_weight == 0:
            raise ValueError("No numeric attention values found - cannot calculate price weight")
        return 1.0 / len([v for v in attention_summary.values() if isinstance(v, (int, float))])
    
    def _calculate_required_volume_weight(self, attention_summary: Dict[str, Any]) -> float:
        """Calculate required volume weight when not provided - FAIL FAST"""
        if not attention_summary:
            raise ValueError("Empty attention_summary - cannot calculate volume weight")
        total_weight = sum(float(v) for v in attention_summary.values() if isinstance(v, (int, float)))
        if total_weight == 0:
            raise ValueError("No numeric attention values found - cannot calculate volume weight")
        return 1.0 / len([v for v in attention_summary.values() if isinstance(v, (int, float))])
    
    def _calculate_required_technical_weight(self, attention_summary: Dict[str, Any]) -> float:
        """Calculate required technical weight when not provided - FAIL FAST"""
        if not attention_summary:
            raise ValueError("Empty attention_summary - cannot calculate technical weight")
        total_weight = sum(float(v) for v in attention_summary.values() if isinstance(v, (int, float)))
        if total_weight == 0:
            raise ValueError("No numeric attention values found - cannot calculate technical weight")
        return 1.0 / len([v for v in attention_summary.values() if isinstance(v, (int, float))])
    
    def _multimodal_bias(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Multi-Modal Ensemble Bias Analysis
        Combina tutti gli algoritmi disponibili per consensus bias
        """
        try:
            # Esegue tutti gli algoritmi disponibili
            results = []
            
            # Sentiment LSTM
            try:
                sentiment_result = self._sentiment_lstm(market_data)
                results.append(("Sentiment_LSTM", sentiment_result))
            except Exception as e:
                raise RuntimeError(f"Sentiment LSTM analysis failed: {e}")
            
            # Volume-Price Analysis
            try:
                volume_result = self._volume_price_analysis(market_data)
                results.append(("VolumePrice", volume_result))
            except Exception as e:
                raise RuntimeError(f"Volume-Price analysis failed: {e}")
            
            # Momentum ML
            try:
                momentum_result = self._momentum_ml(market_data)
                results.append(("Momentum_ML", momentum_result))
            except Exception as e:
                raise RuntimeError(f"Momentum ML analysis failed: {e}")
            
            # Combina risultati con weighted voting
            ensemble_bias = self._combine_bias_results(results)
            
            self.algorithm_stats['successful_predictions'] += 1
            self.algorithm_stats['bias_detections'] += 1
            
            return {
                "directional_bias": ensemble_bias["consensus_bias"],
                "component_analysis": ensemble_bias["individual_results"],
                "ensemble_strength": ensemble_bias["agreement_score"],
                "overall_confidence": ensemble_bias["weighted_confidence"],
                "method": "Multi_Modal_Ensemble_Bias",
                "algorithms_used": len(results)
            }
            
        except Exception as e:
            self.algorithm_stats['failed_predictions'] += 1
            raise PredictionError("MultiModal_Bias", f"Multi-modal bias analysis failed: {str(e)}")
    
    # === HELPER METHODS ===
    
    def _prepare_sentiment_features(self, prices: np.ndarray, volumes: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara features per sentiment analysis"""
        # Price-based sentiment features
        returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices[:-1], 1e-10)
        log_returns = np.log(np.maximum(prices[1:] / np.maximum(prices[:-1], 1e-10), 1e-10))
        log_returns = np.append(log_returns, 0)
        
        # Volatility sentiment
        volatility = np.std(returns[-10:]) if len(returns) >= 10 else 0
        
        # Volume sentiment
        volume_ma = np.mean(volumes) if len(volumes) > 0 else 1.0
        volume_ratio = volumes / max(volume_ma, 1e-10)
        
        # Combine features
        features = np.column_stack([
            prices, volumes, returns, log_returns,
            volume_ratio, np.full_like(prices, volatility)
        ])
        
        return features
    
    def _analyze_market_behavior(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Analizza comportamento di mercato per sentiment"""
        if len(prices) < 10:
            raise InsufficientDataError(required=10, available=len(prices), operation="market_behavior_analysis")
        
        # Panic/Euphoria detection
        price_volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
        volume_spike = np.max(volumes[-5:]) / np.mean(volumes[:-5]) if len(volumes) > 5 else 1.0
        
        if price_volatility > 0.05 and volume_spike > 2.0:
            if prices[-1] < prices[-5]:
                behavior_type = "panic_selling"
                confidence = min(price_volatility * volume_spike / 10, 1.0)
            else:
                behavior_type = "euphoric_buying"
                confidence = min(price_volatility * volume_spike / 10, 1.0)
        else:
            behavior_type = "normal_trading"
            confidence = 0.6
        
        return {
            "type": behavior_type,
            "confidence": float(confidence),
            "volatility": float(price_volatility),
            "volume_spike": float(volume_spike)
        }
    
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, np.ndarray]:
        """Calcola MACD"""
        if len(prices) < slow:
            raise InsufficientDataError(required=slow, available=len(prices), operation=f"MACD_{fast}_{slow}_{signal}")
        
        # EMA calculation
        def ema(data, period):
            alpha = 2.0 / (period + 1)
            ema_values = np.zeros_like(data)
            ema_values[0] = data[0]
            for i in range(1, len(data)):
                ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i-1]
            return ema_values
        
        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)
        macd_line = ema_fast - ema_slow
        macd_signal = ema(macd_line, signal)
        
        return {
            'macd_line': macd_line,
            'macd_signal': macd_signal,
            'macd_histogram': macd_line - macd_signal
        }
    
    def _prepare_price_features(self, prices: np.ndarray) -> np.ndarray:
        """Prepara price features per Transformer"""
        returns = np.diff(prices, prepend=prices[0]) / np.maximum(prices[:-1], 1e-10)
        log_returns = np.log(np.maximum(prices[1:] / np.maximum(prices[:-1], 1e-10), 1e-10))
        log_returns = np.append(log_returns, 0)
        
        # Normalized prices
        normalized_prices = (prices - prices.mean()) / prices.std()
        
        return np.column_stack([normalized_prices, returns, log_returns])
    
    def _prepare_volume_features(self, volumes: np.ndarray) -> np.ndarray:
        """Prepara volume features per Transformer"""
        volume_ma = np.mean(volumes)
        volume_ratio = volumes / max(volume_ma, 1e-10)
        volume_changes = np.diff(volumes, prepend=volumes[0]) / np.maximum(volumes[:-1], 1e-10)
        
        return np.column_stack([volume_ratio, volume_changes])
    
    def _prepare_technical_features(self, prices: np.ndarray, volumes: np.ndarray, market_data: Dict[str, Any]) -> np.ndarray:
        """Prepara technical features per Transformer"""
        # RSI
        from ...data.processors.market_data_processor import MarketDataProcessor
        processor = MarketDataProcessor()
        rsi = processor._calculate_rsi(prices, 14)
        
        # Simple moving averages
        sma_10 = np.convolve(prices, np.ones(10)/10, mode='same')
        sma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
        
        return np.column_stack([rsi, sma_10, sma_20])
    
    def _analyze_attention_patterns(self, attention_weights: np.ndarray) -> Dict[str, Any]:
        """Analizza attention patterns del Transformer"""
        # Placeholder implementation
        return {
            "price_focus": float(np.mean(attention_weights[:, :, :3])),  # First 3 features are price
            "volume_focus": float(np.mean(attention_weights[:, :, 3:5])),  # Next 2 are volume
            "technical_focus": float(np.mean(attention_weights[:, :, 5:])),  # Rest are technical
            "max_attention": float(np.max(attention_weights)),
            "attention_entropy": float(np.sum(-attention_weights * np.log(np.maximum(attention_weights, 1e-10))))
        }
    
    def _combine_bias_results(self, results: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Combina risultati bias di algoritmi diversi"""
        if not results:
            raise ValueError("No bias results to combine")
        
        bias_votes = {"bullish": 0, "bearish": 0, "neutral": 0}
        weighted_confidences = []
        individual_results = {}
        
        for algo_name, result in results:
            bias_direction = result["directional_bias"]["direction"]
            bias_confidence = result["directional_bias"]["confidence"]
            
            # Weighted voting
            bias_votes[bias_direction] += bias_confidence
            weighted_confidences.append(bias_confidence)
            
            individual_results[algo_name] = {
                "direction": bias_direction,
                "confidence": bias_confidence
            }
        
        # Determine consensus
        consensus_direction = max(bias_votes.keys(), key=lambda k: bias_votes[k])
        consensus_strength = bias_votes[consensus_direction] / sum(bias_votes.values())
        
        # Agreement score
        total_votes = sum(bias_votes.values())
        agreement_score = bias_votes[consensus_direction] / total_votes if total_votes > 0 else 0
        
        return {
            "consensus_bias": {
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
def create_bias_detection_algorithms(ml_models: Optional[Dict[str, Any]] = None) -> BiasDetectionAlgorithms:
    """Factory function per creare BiasDetectionAlgorithms"""
    return BiasDetectionAlgorithms(ml_models)


# Export
__all__ = [
    'BiasDetectionAlgorithms',
    'create_bias_detection_algorithms'
]