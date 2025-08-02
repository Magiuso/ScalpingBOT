#!/usr/bin/env python3
"""
TrainingMonitor - Comprehensive Training Observability
=====================================================

Sistema di monitoraggio completo per il training ML che fornisce 
observability dettagliata, metriche in tempo reale, alerting e 
reporting avanzato per diagnosticare problemi di training.

Features:
- Real-time training metrics visualization
- Advanced anomaly detection
- Memory usage monitoring
- Performance profiling
- Training health scoring
- Automated alerting system
- Comprehensive reporting

Author: ScalpingBOT Team
Version: 1.0.0
"""

import numpy as np
import torch
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import logging
import json
import warnings
from pathlib import Path
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

warnings.filterwarnings('ignore')


@dataclass
class MonitorConfig:
    """Configurazione per TrainingMonitor"""
    
    # Monitoring intervals
    metrics_update_interval: float = 1.0      # Secondi tra aggiornamenti metriche
    memory_check_interval: float = 5.0        # Secondi tra check memoria
    health_check_interval: float = 10.0       # Secondi tra health checks
    
    # Data retention
    metrics_history_size: int = 10000          # Numero massimo di metriche storiche
    memory_history_size: int = 1000            # Numero massimo di misurazioni memoria
    
    # Alerting thresholds
    memory_usage_threshold: float = 0.85       # 85% utilizzo memoria
    gpu_memory_threshold: float = 0.90         # 90% utilizzo GPU memoria
    loss_stagnation_threshold: int = 100       # Steps senza miglioramento
    gradient_explosion_threshold: float = 100.0
    learning_rate_min_threshold: float = 1e-8
    
    # Health scoring weights
    loss_improvement_weight: float = 0.3
    gradient_stability_weight: float = 0.2
    memory_efficiency_weight: float = 0.2
    convergence_speed_weight: float = 0.3
    
    # Visualization settings
    enable_plots: bool = True
    plot_update_interval: float = 30.0         # Secondi tra aggiornamenti plot
    save_plots: bool = True
    plots_dir: str = "./test_analyzer_data"
    
    # Reporting
    enable_detailed_logging: bool = True
    save_metrics_to_file: bool = True
    metrics_file_format: str = "json"          # "json", "csv", "both"


class MetricsCollector:
    """Collettore di metriche training in tempo reale"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        
        # Metrics storage
        self.training_metrics = deque(maxlen=config.metrics_history_size)
        self.validation_metrics = deque(maxlen=config.metrics_history_size)
        self.gradient_metrics = deque(maxlen=config.metrics_history_size)
        self.memory_metrics = deque(maxlen=config.memory_history_size)
        
        # Timestamp tracking
        self.start_time = time.time()
        self.last_metric_time = self.start_time
        
        # Threading
        self.lock = threading.RLock()
    
    def add_training_metric(self, step: int, loss: float, learning_rate: float, 
                           grad_norm: float, accuracy: Optional[float] = None):
        """Aggiunge metrica di training"""
        
        with self.lock:
            metric = {
                'timestamp': time.time(),
                'step': step,
                'loss': loss,
                'learning_rate': learning_rate,
                'grad_norm': grad_norm,
                'accuracy': accuracy,
                'elapsed_time': time.time() - self.start_time
            }
            
            self.training_metrics.append(metric)
            self.gradient_metrics.append({
                'timestamp': metric['timestamp'],
                'step': step,
                'grad_norm': grad_norm
            })
    
    def add_validation_metric(self, step: int, val_loss: float, 
                            val_accuracy: Optional[float] = None):
        """Aggiunge metrica di validation"""
        
        with self.lock:
            metric = {
                'timestamp': time.time(),
                'step': step,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'elapsed_time': time.time() - self.start_time
            }
            
            self.validation_metrics.append(metric)
    
    def add_memory_metric(self):
        """Aggiunge metrica di utilizzo memoria"""
        
        with self.lock:
            # CPU Memory
            memory_info = psutil.virtual_memory()
            cpu_memory_used = memory_info.used / (1024**3)  # GB
            cpu_memory_percent = memory_info.percent
            
            # GPU Memory (if available)
            gpu_memory_used = 0.0
            gpu_memory_percent = 0.0
            
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            
            metric = {
                'timestamp': time.time(),
                'cpu_memory_gb': cpu_memory_used,
                'cpu_memory_percent': cpu_memory_percent,
                'gpu_memory_gb': gpu_memory_used,
                'gpu_memory_percent': gpu_memory_percent,
                'elapsed_time': time.time() - self.start_time
            }
            
            self.memory_metrics.append(metric)
    
    def get_latest_metrics(self, count: int = 100) -> Dict[str, Any]:
        """Ottieni ultime metriche"""
        
        with self.lock:
            return {
                'training': list(self.training_metrics)[-count:],
                'validation': list(self.validation_metrics)[-count:],
                'memory': list(self.memory_metrics)[-count:],
                'gradient': list(self.gradient_metrics)[-count:]
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Ottieni summary delle metriche"""
        
        with self.lock:
            summary = {}
            
            # Training metrics summary
            if self.training_metrics:
                recent_train = list(self.training_metrics)[-50:]
                losses = [m['loss'] for m in recent_train]
                lrs = [m['learning_rate'] for m in recent_train]
                grad_norms = [m['grad_norm'] for m in recent_train]
                
                summary['training'] = {
                    'current_loss': losses[-1] if losses else 0.0,
                    'mean_loss': np.mean(losses),
                    'loss_std': np.std(losses),
                    'current_lr': lrs[-1] if lrs else 0.0,
                    'mean_grad_norm': np.mean(grad_norms),
                    'max_grad_norm': np.max(grad_norms),
                    'total_steps': len(self.training_metrics)
                }
            
            # Validation metrics summary
            if self.validation_metrics:
                recent_val = list(self.validation_metrics)[-20:]
                val_losses = [m['val_loss'] for m in recent_val]
                
                summary['validation'] = {
                    'current_val_loss': val_losses[-1] if val_losses else 0.0,
                    'best_val_loss': min(val_losses) if val_losses else float('inf'),
                    'val_loss_trend': self._calculate_trend(val_losses),
                    'validation_count': len(self.validation_metrics)
                }
            
            # Memory summary
            if self.memory_metrics:
                recent_memory = list(self.memory_metrics)[-20:]
                cpu_usage = [m['cpu_memory_percent'] for m in recent_memory]
                gpu_usage = [m['gpu_memory_percent'] for m in recent_memory]
                
                summary['memory'] = {
                    'current_cpu_percent': cpu_usage[-1] if cpu_usage else 0.0,
                    'current_gpu_percent': gpu_usage[-1] if gpu_usage else 0.0,
                    'max_cpu_percent': max(cpu_usage) if cpu_usage else 0.0,
                    'max_gpu_percent': max(gpu_usage) if gpu_usage else 0.0
                }
            
            return summary
    
    def _calculate_trend(self, values: List[float], window: int = 10) -> str:
        """Calcola trend dei valori (improving, declining, stable)"""
        
        if len(values) < window:
            return "insufficient_data"
        
        recent = values[-window:]
        earlier = values[-window*2:-window] if len(values) >= window*2 else values[:-window]
        
        if not earlier:
            return "insufficient_data"
        
        recent_mean = np.mean(recent)
        earlier_mean = np.mean(earlier)
        
        relative_change = (recent_mean - earlier_mean) / (abs(earlier_mean) + 1e-8)
        
        if relative_change < -0.05:  # 5% improvement
            return "improving"
        elif relative_change > 0.05:  # 5% deterioration
            return "declining"
        else:
            return "stable"


class AnomalyDetector:
    """Detector di anomalie nel training"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.anomalies_detected = []
        
    def detect_anomalies(self, metrics_collector: MetricsCollector) -> List[Dict[str, Any]]:
        """Detecta anomalie nelle metriche"""
        
        anomalies = []
        summary = metrics_collector.get_metrics_summary()
        
        # Loss explosion
        if 'training' in summary:
            current_loss = summary['training']['current_loss']
            mean_loss = summary['training']['mean_loss']
            
            if current_loss > mean_loss * 5:  # Loss 5x superiore alla media
                anomalies.append({
                    'type': 'loss_explosion',
                    'severity': 'high',
                    'message': f"Loss explosion detected: {current_loss:.6f} vs mean {mean_loss:.6f}",
                    'timestamp': time.time()
                })
        
        # Gradient explosion
        if 'training' in summary:
            max_grad_norm = summary['training']['max_grad_norm']
            
            if max_grad_norm > self.config.gradient_explosion_threshold:
                anomalies.append({
                    'type': 'gradient_explosion',
                    'severity': 'high',
                    'message': f"Gradient explosion: {max_grad_norm:.2f}",
                    'timestamp': time.time()
                })
        
        # Memory pressure
        if 'memory' in summary:
            cpu_percent = summary['memory']['current_cpu_percent']
            gpu_percent = summary['memory']['current_gpu_percent']
            
            if cpu_percent > self.config.memory_usage_threshold * 100:
                anomalies.append({
                    'type': 'high_cpu_memory',
                    'severity': 'medium',
                    'message': f"High CPU memory usage: {cpu_percent:.1f}%",
                    'timestamp': time.time()
                })
            
            if gpu_percent > self.config.gpu_memory_threshold * 100:
                anomalies.append({
                    'type': 'high_gpu_memory',
                    'severity': 'high',
                    'message': f"High GPU memory usage: {gpu_percent:.1f}%",
                    'timestamp': time.time()
                })
        
        # Learning rate too low
        if 'training' in summary:
            current_lr = summary['training']['current_lr']
            
            if current_lr < self.config.learning_rate_min_threshold:
                anomalies.append({
                    'type': 'learning_rate_too_low',
                    'severity': 'medium',
                    'message': f"Learning rate very low: {current_lr:.2e}",
                    'timestamp': time.time()
                })
        
        # Validation stagnation
        if 'validation' in summary:
            trend = summary['validation']['val_loss_trend']
            
            if trend == "stable":
                anomalies.append({
                    'type': 'validation_stagnation',
                    'severity': 'low',
                    'message': "Validation loss has stagnated",
                    'timestamp': time.time()
                })
        
        # Store detected anomalies
        self.anomalies_detected.extend(anomalies)
        
        return anomalies


class HealthScorer:
    """Calcolatore di health score del training"""
    
    def __init__(self, config: MonitorConfig):
        self.config = config
        
    def calculate_health_score(self, metrics_collector: MetricsCollector) -> Dict[str, Any]:
        """Calcola health score complessivo (0-100)"""
        
        summary = metrics_collector.get_metrics_summary()
        
        # Component scores
        loss_score = self._calculate_loss_improvement_score(summary)
        gradient_score = self._calculate_gradient_stability_score(summary)
        memory_score = self._calculate_memory_efficiency_score(summary)
        convergence_score = self._calculate_convergence_speed_score(summary)
        
        # Weighted overall score
        overall_score = (
            loss_score * self.config.loss_improvement_weight +
            gradient_score * self.config.gradient_stability_weight +
            memory_score * self.config.memory_efficiency_weight +
            convergence_score * self.config.convergence_speed_weight
        )
        
        return {
            'overall_score': max(0, min(100, overall_score)),
            'component_scores': {
                'loss_improvement': loss_score,
                'gradient_stability': gradient_score,
                'memory_efficiency': memory_score,
                'convergence_speed': convergence_score
            },
            'health_status': self._get_health_status(overall_score)
        }
    
    def _calculate_loss_improvement_score(self, summary: Dict[str, Any]) -> float:
        """Score per miglioramento loss (0-100)"""
        
        if 'validation' not in summary:
            return 50.0  # Neutral score
        
        trend = summary['validation']['val_loss_trend']
        
        if trend == "improving":
            return 90.0
        elif trend == "stable":
            return 50.0
        elif trend == "declining":
            return 10.0
        else:
            return 50.0
    
    def _calculate_gradient_stability_score(self, summary: Dict[str, Any]) -> float:
        """Score per stabilità gradienti (0-100)"""
        
        if 'training' not in summary:
            return 50.0
        
        mean_grad = summary['training']['mean_grad_norm']
        max_grad = summary['training']['max_grad_norm']
        
        # Penalize exploding gradients
        if max_grad > self.config.gradient_explosion_threshold:
            return 10.0
        
        # Penalize vanishing gradients
        if mean_grad < 1e-6:
            return 20.0
        
        # Good gradient range
        if 1e-4 <= mean_grad <= 10.0:
            return 85.0
        
        return 60.0
    
    def _calculate_memory_efficiency_score(self, summary: Dict[str, Any]) -> float:
        """Score per efficienza memoria (0-100)"""
        
        if 'memory' not in summary:
            return 50.0
        
        cpu_percent = summary['memory']['current_cpu_percent']
        gpu_percent = summary['memory']['current_gpu_percent']
        
        # Penalize high memory usage
        memory_penalty = 0
        
        if cpu_percent > 90:
            memory_penalty += 30
        elif cpu_percent > 80:
            memory_penalty += 15
        
        if gpu_percent > 95:
            memory_penalty += 40
        elif gpu_percent > 85:
            memory_penalty += 20
        
        return max(0, 100 - memory_penalty)
    
    def _calculate_convergence_speed_score(self, summary: Dict[str, Any]) -> float:
        """Score per velocità di convergenza (0-100)"""
        
        if 'training' not in summary:
            return 50.0
        
        total_steps = summary['training']['total_steps']
        
        # Simple heuristic: more steps might indicate slower convergence
        if total_steps < 100:
            return 90.0
        elif total_steps < 1000:
            return 75.0
        elif total_steps < 5000:
            return 60.0
        else:
            return 40.0
    
    def _get_health_status(self, score: float) -> str:
        """Determina status testuale da score numerico"""
        
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "fair"
        elif score >= 20:
            return "poor"
        else:
            return "critical"


class TrainingMonitor:
    """
    Monitor completo per training ML con observability avanzata
    
    Features:
    - Real-time metrics collection
    - Anomaly detection
    - Health scoring
    - Memory monitoring
    - Automated alerting
    - Visualization and reporting
    """
    
    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig()
        
        # Core components
        self.metrics_collector = MetricsCollector(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.health_scorer = HealthScorer(self.config)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable] = []
        
        # Visualization
        self.plots_dir = Path(self.config.plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        print(f"📊 TrainingMonitor initialized with {self.config.metrics_history_size} metrics buffer")
    
    def start_monitoring(self):
        """Avvia il monitoraggio in background"""
        
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="TrainingMonitor",
            daemon=True
        )
        self.monitoring_thread.start()
        
        print("🚀 Training monitoring started")
    
    def stop_monitoring(self):
        """Ferma il monitoraggio"""
        
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        print("⏹️ Training monitoring stopped")
    
    def _monitoring_loop(self):
        """Loop principale di monitoraggio"""
        
        last_memory_check = 0
        last_health_check = 0
        
        while self.is_monitoring and not self.stop_event.is_set():
            
            current_time = time.time()
            
            # Memory monitoring
            if current_time - last_memory_check >= self.config.memory_check_interval:
                self.metrics_collector.add_memory_metric()
                last_memory_check = current_time
            
            # Health check and anomaly detection
            if current_time - last_health_check >= self.config.health_check_interval:
                self._perform_health_check()
                last_health_check = current_time
            
            # Sleep
            time.sleep(1.0)
    
    def _perform_health_check(self):
        """Esegue check di salute completo"""
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(self.metrics_collector)
        
        # Calculate health score
        health_info = self.health_scorer.calculate_health_score(self.metrics_collector)
        
        # 🔧 RATE LIMITING per Health Score: log solo se cambiato o ogni 5 minuti
        current_score = health_info['overall_score']
        current_time = time.time()
        
        # Log health status con rate limiting intelligente
        if self.config.enable_detailed_logging:
            should_log = False
            
            # Log se score cambiato di più di 5 punti
            if not hasattr(self, '_last_health_score'):
                self._last_health_score = current_score
                self._last_health_log_time = current_time
                should_log = True
            elif abs(current_score - self._last_health_score) >= 5.0:
                should_log = True
                self._last_health_score = current_score
                self._last_health_log_time = current_time
            # Oppure ogni 5 minuti (300 secondi)
            elif current_time - self._last_health_log_time >= 300:
                should_log = True
                self._last_health_log_time = current_time
            
            if should_log:
                self.logger.info(f"Health Score: {health_info['overall_score']:.1f} ({health_info['health_status']})")
        
        # Trigger alerts for anomalies
        for anomaly in anomalies:
            self._trigger_alert(anomaly)
        
        # Trigger alert for critical health
        if health_info['overall_score'] < 20:
            self._trigger_alert({
                'type': 'critical_health',
                'severity': 'critical',
                'message': f"Training health critical: {health_info['overall_score']:.1f}",
                'timestamp': time.time()
            })
    
    def _trigger_alert(self, alert: Dict[str, Any]):
        """Triggera alert ai callback registrati"""
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def log_training_step(self, step: int, loss: float, learning_rate: float, 
                         grad_norm: float, accuracy: Optional[float] = None):
        """Log di un passo di training"""
        
        self.metrics_collector.add_training_metric(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            accuracy=accuracy
        )
    
    def log_validation_step(self, step: int, val_loss: float, 
                           val_accuracy: Optional[float] = None):
        """Log di validazione"""
        
        self.metrics_collector.add_validation_metric(
            step=step,
            val_loss=val_loss,
            val_accuracy=val_accuracy
        )
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Aggiunge callback per alert"""
        self.alert_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Ottieni status corrente completo"""
        
        metrics_summary = self.metrics_collector.get_metrics_summary()
        health_info = self.health_scorer.calculate_health_score(self.metrics_collector)
        recent_anomalies = [a for a in self.anomaly_detector.anomalies_detected 
                           if time.time() - a['timestamp'] < 300]  # Last 5 minutes
        
        return {
            'monitoring_active': self.is_monitoring,
            'uptime_seconds': time.time() - self.metrics_collector.start_time,
            'metrics_summary': metrics_summary,
            'health_info': health_info,
            'recent_anomalies': recent_anomalies,
            'total_anomalies': len(self.anomaly_detector.anomalies_detected)
        }
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Genera report completo del training"""
        
        status = self.get_current_status()
        latest_metrics = self.metrics_collector.get_latest_metrics(1000)
        
        # Training statistics
        training_stats = {}
        if latest_metrics['training']:
            losses = [m['loss'] for m in latest_metrics['training']]
            lrs = [m['learning_rate'] for m in latest_metrics['training']]
            
            training_stats = {
                'total_steps': len(latest_metrics['training']),
                'loss_statistics': {
                    'initial': losses[0] if losses else 0.0,
                    'final': losses[-1] if losses else 0.0,
                    'best': min(losses) if losses else 0.0,
                    'mean': np.mean(losses) if losses else 0.0,
                    'std': np.std(losses) if losses else 0.0
                },
                'learning_rate_statistics': {
                    'initial': lrs[0] if lrs else 0.0,
                    'final': lrs[-1] if lrs else 0.0,
                    'mean': np.mean(lrs) if lrs else 0.0
                }
            }
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'training_duration_seconds': status['uptime_seconds'],
            'current_status': status,
            'training_statistics': training_stats,
            'anomalies_summary': {
                'total_anomalies': len(self.anomaly_detector.anomalies_detected),
                'anomaly_types': self._summarize_anomalies()
            }
        }
    
    def _summarize_anomalies(self) -> Dict[str, int]:
        """Riassunto delle anomalie per tipo"""
        
        anomaly_counts = defaultdict(int)
        for anomaly in self.anomaly_detector.anomalies_detected:
            anomaly_counts[anomaly['type']] += 1
        
        return dict(anomaly_counts)
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Salva report su file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_{timestamp}.json"
        
        report = self.generate_training_report()
        
        filepath = self.plots_dir / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Training report saved to {filepath}")
        return str(filepath)


# Helper functions
def create_monitor_config(**kwargs) -> MonitorConfig:
    """Factory function per creare configurazione monitor"""
    
    default_config = {
        'metrics_update_interval': 1.0,
        'memory_check_interval': 5.0,
        'enable_plots': True,
        'save_metrics_to_file': True
    }
    
    default_config.update(kwargs)
    return MonitorConfig(**default_config)


def test_training_monitor():
    """Test delle funzionalità del TrainingMonitor"""
    print("🧪 Testing TrainingMonitor...")
    
    # Create configuration
    config = create_monitor_config(
        metrics_history_size=1000,
        enable_plots=False  # Disable plots for testing
    )
    
    # Create monitor
    monitor = TrainingMonitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate training steps
    for step in range(10):
        monitor.log_training_step(
            step=step,
            loss=1.0 / (step + 1),  # Decreasing loss
            learning_rate=1e-3,
            grad_norm=0.5
        )
        
        if step % 3 == 0:
            monitor.log_validation_step(
                step=step,
                val_loss=1.2 / (step + 1)
            )
        
        time.sleep(0.1)
    
    # Get status
    status = monitor.get_current_status()
    print(f"✅ Health Score: {status['health_info']['overall_score']:.1f}")
    print(f"✅ Total Steps: {len(monitor.metrics_collector.training_metrics)}")
    
    # Generate report
    report_path = monitor.save_report()
    print(f"✅ Report saved: {report_path}")
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    return monitor


if __name__ == "__main__":
    test_training_monitor()