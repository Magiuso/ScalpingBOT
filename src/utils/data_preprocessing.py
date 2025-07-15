#!/usr/bin/env python3
"""
Advanced Data Preprocessor for ML Training Optimization
=====================================================

Preprocessor avanzato con normalizzazione intelligente, outlier detection 
e windowing adattivo per risolvere i problemi di training ML ricorrenti.

Author: ScalpingBOT Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import logging
from collections import deque
import warnings

warnings.filterwarnings('ignore')


@dataclass
class PreprocessingConfig:
    """Configurazione per il preprocessing avanzato"""
    
    # Outlier detection
    outlier_threshold: float = 3.0  # Z-score threshold
    outlier_method: str = 'isolation_forest'  # 'isolation_forest', 'elliptic_envelope', 'one_class_svm'
    outlier_contamination: float = 0.1  # Percentuale outliers attesa
    
    # Normalization
    normalization_method: str = 'auto'  # 'auto', 'standard', 'robust', 'minmax', 'quantile'
    quantile_output_distribution: str = 'uniform'  # 'uniform', 'normal'
    
    # Adaptive windowing
    volatility_threshold: float = 0.02  # Soglia volatilità per window adaptation
    min_window_size: int = 50
    max_window_size: int = 200
    adaptive_windowing: bool = True
    
    # Data stability
    stability_window: int = 100  # Window per tracking stabilità
    drift_threshold: float = 0.1  # Soglia per data drift detection
    
    # Performance
    cache_scalers: bool = True
    parallel_processing: bool = True


class DataStabilityTracker:
    """Tracker per monitorare la stabilità dei dati nel tempo"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.mean_history = deque(maxlen=window_size)
        self.std_history = deque(maxlen=window_size)
        
    def update(self, data: np.ndarray):
        """Aggiorna le statistiche di stabilità"""
        current_mean = np.mean(data)
        current_std = np.std(data)
        
        self.history.append(data.copy())
        self.mean_history.append(current_mean)
        self.std_history.append(current_std)
    
    def detect_drift(self, threshold: float = 0.1) -> Dict[str, Any]:
        """Detecta drift nei dati"""
        if len(self.mean_history) < 10:
            return {'drift_detected': False, 'confidence': 0.0}
        
        recent_means = list(self.mean_history)[-10:]
        overall_mean = np.mean(list(self.mean_history))
        recent_mean = np.mean(recent_means)
        
        drift_magnitude = abs(recent_mean - overall_mean) / (abs(overall_mean) + 1e-8)
        drift_detected = drift_magnitude > threshold
        
        return {
            'drift_detected': drift_detected,
            'drift_magnitude': drift_magnitude,
            'confidence': min(1.0, drift_magnitude / threshold),
            'recent_mean': recent_mean,
            'overall_mean': overall_mean
        }
    
    def get_stability_score(self) -> float:
        """Calcola score di stabilità (0-1, higher is better)"""
        if len(self.std_history) < 5:
            return 0.5
        
        std_of_stds = np.std(list(self.std_history))
        mean_std = np.mean(list(self.std_history))
        
        # Normalizza il coefficient of variation
        cv = std_of_stds / (mean_std + 1e-8)
        stability_score = 1.0 / (1.0 + cv)
        
        return float(min(1.0, max(0.0, stability_score)))


class AdvancedDataPreprocessor:
    """
    Preprocessor avanzato con normalizzazione intelligente e outlier detection
    
    Features:
    - Normalizzazione adattiva basata sulla distribuzione dei dati
    - Outlier detection con multiple strategie
    - Windowing adattivo basato sulla volatilità
    - Data drift detection
    - Caching intelligente dei scalers
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()
        self.scalers: Dict[str, Any] = {}
        self.outlier_detectors: Dict[str, Any] = {}
        self.stability_tracker: Dict[str, DataStabilityTracker] = {}
        self.preprocessing_stats: Dict[str, Dict] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        print(f"📊 AdvancedDataPreprocessor initialized with config: {self.config.normalization_method}")
    
    def smart_normalize(self, data: np.ndarray, column_name: str = 'default') -> np.ndarray:
        """
        Normalizzazione intelligente che adatta il metodo ai dati
        
        Args:
            data: Array numpy da normalizzare
            column_name: Nome identificativo per caching
            
        Returns:
            np.ndarray: Dati normalizzati
        """
        if data.size == 0:
            return data
        
        # Inizializza tracker se necessario
        if column_name not in self.stability_tracker:
            self.stability_tracker[column_name] = DataStabilityTracker(self.config.stability_window)
        
        # Aggiorna tracker stabilità
        self.stability_tracker[column_name].update(data)
        
        # Determina metodo di normalizzazione
        if self.config.normalization_method == 'auto':
            normalization_method = self._select_normalization_method(data)
        else:
            normalization_method = self.config.normalization_method
        
        # Cache key per scaler
        scaler_key = f"{column_name}_{normalization_method}"
        
        # Riutilizza scaler se disponibile e stabile
        if (self.config.cache_scalers and 
            scaler_key in self.scalers and 
            self._is_data_stable(column_name)):
            
            scaler = self.scalers[scaler_key]
            normalized_data = scaler.transform(data.reshape(-1, 1)).flatten()
            
        else:
            # Crea nuovo scaler
            scaler = self._create_scaler(normalization_method)
            
            try:
                normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
                
                # Cache scaler se configurato
                if self.config.cache_scalers:
                    self.scalers[scaler_key] = scaler
                    
            except Exception as e:
                self.logger.warning(f"Normalization fallback for {column_name}: {e}")
                # Fallback a robust scaler
                scaler = RobustScaler()
                normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Salva statistiche
        self._update_preprocessing_stats(column_name, data, normalized_data, normalization_method)
        
        return normalized_data
    
    def _select_normalization_method(self, data: np.ndarray) -> str:
        """Seleziona automaticamente il metodo di normalizzazione ottimale"""
        
        # Calcola statistiche distribuzione
        skewness = self._calculate_skewness(data)
        outlier_ratio = self._estimate_outlier_ratio(data)
        data_range = np.max(data) - np.min(data)
        
        # Decision tree per metodo ottimale
        if outlier_ratio > 0.1:
            # Molti outliers → Robust Scaler
            return 'robust'
        elif abs(skewness) > 2.0:
            # Distribuzione molto skewed → Quantile Transform
            return 'quantile'
        elif data_range > 0 and np.min(data) >= 0:
            # Dati positivi con range definito → MinMax
            return 'minmax'
        else:
            # Distribuzione normale → Standard Scaler
            return 'standard'
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calcola skewness della distribuzione"""
        try:
            from scipy import stats
            skew_value = stats.skew(data)
            return float(skew_value)
        except ImportError:
            # Fallback manual calculation
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0.0
            
            n = len(data)
            skewness = (n / ((n-1) * (n-2))) * np.sum(((data - mean) / std) ** 3)
            return float(skew_value)
    
    def _estimate_outlier_ratio(self, data: np.ndarray, threshold: float = 3.0) -> float:
        """Stima la percentuale di outliers usando Z-score"""
        if len(data) < 10:
            return 0.0
        
        z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-8))
        outliers = int(np.sum(z_scores > threshold))
        return outliers / len(data)
    
    def _create_scaler(self, method: str) -> Any:
        """Crea scaler basato sul metodo specificato"""
        if method == 'standard':
            return StandardScaler()
        elif method == 'robust':
            return RobustScaler()
        elif method == 'minmax':
            return MinMaxScaler()
        elif method == 'quantile':
            # Fix literal type issue by using conditional instantiation
            if self.config.quantile_output_distribution == 'uniform':
                from sklearn.preprocessing import QuantileTransformer as QT_Uniform
                return QT_Uniform(output_distribution='uniform', random_state=42)
            else:
                from sklearn.preprocessing import QuantileTransformer as QT_Normal  
                return QT_Normal(output_distribution='normal', random_state=42)
        else:
            # Default fallback
            return RobustScaler()
    
    def _is_data_stable(self, column_name: str) -> bool:
        """Verifica se i dati sono stabili per riutilizzare scaler"""
        if column_name not in self.stability_tracker:
            return False
        
        stability_score = self.stability_tracker[column_name].get_stability_score()
        drift_info = self.stability_tracker[column_name].detect_drift(self.config.drift_threshold)
        
        return stability_score > 0.7 and not drift_info['drift_detected']
    
    def detect_and_handle_outliers(self, data: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """
        Detection e gestione outliers con multiple strategie
        
        Args:
            data: Array numpy da processare
            method: Metodo di detection ('isolation_forest', 'elliptic_envelope', 'one_class_svm')
            
        Returns:
            np.ndarray: Dati con outliers gestiti
        """
        if data.size == 0:
            return data
        
        method = method or self.config.outlier_method
        
        # Reshape per sklearn se necessario
        data_reshaped = data.reshape(-1, 1) if data.ndim == 1 else data
        
        try:
            # Crea detector
            detector = self._create_outlier_detector(method)
            
            # Fit e predict
            outlier_labels = detector.fit_predict(data_reshaped)
            
            # -1 = outlier, 1 = inlier in sklearn
            outlier_mask = outlier_labels == -1
            
            if np.any(outlier_mask):
                # Gestisci outliers
                cleaned_data = self._handle_outliers(data, outlier_mask)
                
                outlier_count = np.sum(outlier_mask)
                outlier_percentage = (outlier_count / len(data)) * 100
                
                print(f"🔍 Outliers detected: {outlier_count}/{len(data)} ({outlier_percentage:.1f}%) with {method}")
                
                return cleaned_data
            
        except Exception as e:
            self.logger.warning(f"Outlier detection failed: {e}, using Z-score fallback")
            # Fallback a Z-score method
            return self._zscore_outlier_handling(data)
        
        return data
    
    def _create_outlier_detector(self, method: str) -> Any:
        """Crea detector di outliers basato sul metodo"""
        if method == 'isolation_forest':
            return IsolationForest(
                contamination=self.config.outlier_contamination,
                random_state=42,
                n_estimators=100
            )
        elif method == 'elliptic_envelope':
            return EllipticEnvelope(
                contamination=self.config.outlier_contamination,
                random_state=42
            )
        elif method == 'one_class_svm':
            return OneClassSVM(
                nu=self.config.outlier_contamination,
                gamma='scale'
            )
        else:
            # Default
            return IsolationForest(contamination=self.config.outlier_contamination, random_state=42)
    
    def _handle_outliers(self, data: np.ndarray, outlier_mask: np.ndarray) -> np.ndarray:
        """Gestisce outliers tramite clipping ai percentili"""
        cleaned_data = data.copy()
        
        # Calcola percentili dei dati normali
        normal_data = data[~outlier_mask]
        if len(normal_data) > 0:
            lower_bound = np.percentile(normal_data, 5)
            upper_bound = np.percentile(normal_data, 95)
            
            # Clip outliers ai bounds
            cleaned_data = np.clip(cleaned_data, lower_bound, upper_bound)
        
        return cleaned_data
    
    def _zscore_outlier_handling(self, data: np.ndarray) -> np.ndarray:
        """Fallback outlier handling usando Z-score"""
        z_scores = np.abs((data - np.mean(data)) / (np.std(data) + 1e-8))
        outlier_mask = z_scores > self.config.outlier_threshold
        
        if np.any(outlier_mask):
            # Clip to mean ± 3*std
            mean_val = np.mean(data)
            std_val = np.std(data)
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            
            return np.clip(data, lower_bound, upper_bound)
        
        return data
    
    def adaptive_windowing(self, data: np.ndarray, volatility_threshold: Optional[float] = None) -> int:
        """
        Determina dimensione finestra ottimale basata sulla volatilità del mercato
        
        Args:
            data: Serie temporale dei dati
            volatility_threshold: Soglia volatilità per adaptation
            
        Returns:
            int: Dimensione finestra ottimale
        """
        if not self.config.adaptive_windowing or len(data) < self.config.min_window_size:
            return self.config.min_window_size
        
        threshold = volatility_threshold or self.config.volatility_threshold
        
        # Calcola volatilità rolling
        window_size = min(50, len(data) // 4)
        volatilities = []
        
        for i in range(window_size, len(data)):
            window_data = data[i-window_size:i]
            volatility = np.std(window_data) / (np.mean(np.abs(window_data)) + 1e-8)
            volatilities.append(volatility)
        
        if not volatilities:
            return self.config.min_window_size
        
        avg_volatility = np.mean(volatilities)
        
        # Adatta dimensione finestra inversamente alla volatilità
        if avg_volatility > threshold * 2:
            # Alta volatilità → finestra piccola
            optimal_window = self.config.min_window_size
        elif avg_volatility > threshold:
            # Media volatilità → finestra media
            optimal_window = (self.config.min_window_size + self.config.max_window_size) // 2
        else:
            # Bassa volatilità → finestra grande
            optimal_window = self.config.max_window_size
        
        return max(self.config.min_window_size, min(self.config.max_window_size, optimal_window))
    
    def _update_preprocessing_stats(self, column_name: str, original_data: np.ndarray, 
                                  processed_data: np.ndarray, method: str):
        """Aggiorna statistiche di preprocessing"""
        self.preprocessing_stats[column_name] = {
            'method_used': method,
            'original_mean': np.mean(original_data),
            'original_std': np.std(original_data),
            'processed_mean': np.mean(processed_data),
            'processed_std': np.std(processed_data),
            'stability_score': self.stability_tracker[column_name].get_stability_score(),
            'outlier_ratio': self._estimate_outlier_ratio(original_data)
        }
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Genera report completo del preprocessing"""
        return {
            'preprocessing_stats': self.preprocessing_stats,
            'cached_scalers': len(self.scalers),
            'tracked_columns': len(self.stability_tracker),
            'config': self.config.__dict__
        }
    
    def reset_cache(self):
        """Reset cache scalers e tracker"""
        self.scalers.clear()
        self.stability_tracker.clear()
        self.preprocessing_stats.clear()
        print("🔄 Preprocessing cache reset")


# Helper functions
def create_optimized_preprocessor() -> AdvancedDataPreprocessor:
    """Factory function per creare preprocessor ottimizzato"""
    config = PreprocessingConfig(
        outlier_threshold=3.0,
        outlier_method='isolation_forest',
        normalization_method='auto',
        adaptive_windowing=True,
        cache_scalers=True
    )
    return AdvancedDataPreprocessor(config)


def test_preprocessor():
    """Test delle funzionalità del preprocessor"""
    print("🧪 Testing AdvancedDataPreprocessor...")
    
    # Crea dati di test
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    skewed_data = np.random.exponential(2, 1000)
    outlier_data = np.concatenate([normal_data, [10, -10, 15, -15]])  # Aggiungi outliers
    
    preprocessor = create_optimized_preprocessor()
    
    # Test normalizzazione
    normalized_normal = preprocessor.smart_normalize(normal_data, 'normal')
    normalized_skewed = preprocessor.smart_normalize(skewed_data, 'skewed')
    
    # Test outlier detection
    cleaned_outlier = preprocessor.detect_and_handle_outliers(outlier_data)
    
    # Test adaptive windowing
    optimal_window = preprocessor.adaptive_windowing(normal_data)
    
    print(f"✅ Normal data: mean={np.mean(normalized_normal):.3f}, std={np.std(normalized_normal):.3f}")
    print(f"✅ Skewed data: mean={np.mean(normalized_skewed):.3f}, std={np.std(normalized_skewed):.3f}")
    print(f"✅ Outlier cleaned: {len(outlier_data)} → {len(cleaned_outlier)} samples")
    print(f"✅ Optimal window size: {optimal_window}")
    
    # Report
    report = preprocessor.get_preprocessing_report()
    print(f"📊 Preprocessing report: {len(report['preprocessing_stats'])} columns processed")
    
    return preprocessor


if __name__ == "__main__":
    test_preprocessor()