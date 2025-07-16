#!/usr/bin/env python3
"""
Optimized Training Configuration
==============================

Configurazioni training ottimizzate che integrano tutti i moduli di ottimizzazione
ML per risolvere problemi ricorrenti di vanishing gradients, overfitting e 
instabilità training.

Integration modules:
- AdvancedDataPreprocessor
- OptimizedLSTM  
- AdaptiveTrainer
- TrainingMonitor

Author: ScalpingBOT Team
Version: 1.0.0
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

# Import our optimization modules
from .data_preprocessing import AdvancedDataPreprocessor, PreprocessingConfig, create_optimized_preprocessor
from .optimized_lstm import OptimizedLSTM, LSTMConfig, create_optimized_lstm_config
from .adaptive_trainer import AdaptiveTrainer, TrainingConfig, create_adaptive_trainer_config
from .training_monitor import TrainingMonitor, MonitorConfig, create_monitor_config


class OptimizationProfile(Enum):
    """Profili di ottimizzazione predefiniti"""
    HIGH_PERFORMANCE = "high_performance"      # Massima performance, minimal overhead
    STABLE_TRAINING = "stable_training"        # Focus su stabilità e convergenza
    RESEARCH_MODE = "research_mode"           # Debugging e sperimentazione completa
    PRODUCTION_READY = "production_ready"     # Bilanciato per deployment


class ModelType(Enum):
    """Tipi di modelli supportati"""
    LSTM_SUPPORT_RESISTANCE = "lstm_support_resistance"
    LSTM_PATTERN_RECOGNITION = "lstm_pattern_recognition"
    LSTM_BIAS_DETECTION = "lstm_bias_detection"
    LSTM_TREND_ANALYSIS = "lstm_trend_analysis"
    LSTM_VOLATILITY_PREDICTION = "lstm_volatility_prediction"
    LSTM_MOMENTUM_ANALYSIS = "lstm_momentum_analysis"


@dataclass
class OptimizedTrainingPipeline:
    """Pipeline completa di training ottimizzato"""
    
    # Model configuration
    model_type: ModelType = ModelType.LSTM_SUPPORT_RESISTANCE
    input_features: int = 64
    sequence_length: int = 50
    output_size: int = 1
    
    # Optimization profile
    optimization_profile: OptimizationProfile = OptimizationProfile.STABLE_TRAINING
    
    # Component configurations
    preprocessing_config: Optional[PreprocessingConfig] = None
    lstm_config: Optional[LSTMConfig] = None
    training_config: Optional[TrainingConfig] = None
    monitor_config: Optional[MonitorConfig] = None
    
    # Training data settings
    train_test_split: float = 0.8
    validation_split: float = 0.1
    batch_size: int = 32
    
    # Model saving
    save_directory: str = "./optimized_models"
    model_name_prefix: str = "optimized_lstm"
    
    def __post_init__(self):
        """Inizializza configurazioni automaticamente se non fornite"""
        
        # Auto-generate configs based on profile
        if self.preprocessing_config is None:
            self.preprocessing_config = self._create_preprocessing_config()
        
        if self.lstm_config is None:
            self.lstm_config = self._create_lstm_config()
        
        if self.training_config is None:
            self.training_config = self._create_training_config()
        
        if self.monitor_config is None:
            self.monitor_config = self._create_monitor_config()
    
    def _create_preprocessing_config(self) -> PreprocessingConfig:
        """Crea configurazione preprocessing basata sul profilo"""
        
        if self.optimization_profile == OptimizationProfile.HIGH_PERFORMANCE:
            return PreprocessingConfig(
                outlier_threshold=3.0,
                outlier_method='isolation_forest',
                normalization_method='robust',  # Faster than auto
                adaptive_windowing=False,  # Disable for speed
                cache_scalers=True,
                parallel_processing=True
            )
        
        elif self.optimization_profile == OptimizationProfile.STABLE_TRAINING:
            return PreprocessingConfig(
                outlier_threshold=2.5,  # More conservative
                outlier_method='isolation_forest',
                normalization_method='auto',  # Smart selection
                adaptive_windowing=True,
                volatility_threshold=0.015,  # Lower threshold
                cache_scalers=True,
                parallel_processing=True,
                drift_threshold=0.05  # More sensitive to drift
            )
        
        elif self.optimization_profile == OptimizationProfile.RESEARCH_MODE:
            return PreprocessingConfig(
                outlier_threshold=3.0,
                outlier_method='elliptic_envelope',  # More thorough
                normalization_method='auto',
                adaptive_windowing=True,
                cache_scalers=False,  # Always recompute for research
                parallel_processing=False,  # Single-threaded for debugging
                stability_window=200  # Larger window for analysis
            )
        
        else:  # PRODUCTION_READY
            return PreprocessingConfig(
                outlier_threshold=2.0,  # Conservative
                outlier_method='isolation_forest',
                normalization_method='robust',  # Stable and fast
                adaptive_windowing=True,
                cache_scalers=True,
                parallel_processing=True,
                drift_threshold=0.08  # Less sensitive for stability
            )
    
    def _create_lstm_config(self) -> LSTMConfig:
        """Crea configurazione LSTM basata sul profilo"""
        
        # Base configuration
        base_config = {
            'input_size': self.input_features,
            'output_size': self.output_size,
            'use_layer_norm': True,
            'use_skip_connections': True,
            'use_attention': True,
            'use_highway': True
        }
        
        if self.optimization_profile == OptimizationProfile.HIGH_PERFORMANCE:
            base_config.update({
                'hidden_size': 128,
                'num_layers': 2,  # Fewer layers for speed
                'dropout_rate': 0.1,
                'attention_heads': 4,
                'use_gradient_checkpointing': False,
                'gradient_clip_norm': 1.0
            })
        
        elif self.optimization_profile == OptimizationProfile.STABLE_TRAINING:
            base_config.update({
                'hidden_size': 256,
                'num_layers': 3,
                'dropout_rate': 0.3,  # More dropout for stability
                'attention_heads': 8,
                'attention_dropout': 0.2,
                'gradient_clip_norm': 0.5,  # Conservative clipping
                'gradient_clip_value': 0.5,
                'forget_gate_bias': 2.0,  # Strong anti-vanishing
                'weight_decay': 1e-3
            })
        
        elif self.optimization_profile == OptimizationProfile.RESEARCH_MODE:
            base_config.update({
                'hidden_size': 512,  # Larger for research
                'num_layers': 4,
                'dropout_rate': 0.25,
                'attention_heads': 16,
                'use_gradient_checkpointing': True,
                'gradient_clip_norm': 2.0,  # Less aggressive for analysis
                'weight_init_method': 'orthogonal'  # Research-friendly
            })
        
        else:  # PRODUCTION_READY
            base_config.update({
                'hidden_size': 192,
                'num_layers': 3,
                'dropout_rate': 0.2,
                'attention_heads': 6,
                'gradient_clip_norm': 1.0,
                'weight_decay': 5e-4,
                'use_gradient_checkpointing': False  # Disable for prod stability
            })
        
        return LSTMConfig(**base_config)
    
    def _create_training_config(self) -> TrainingConfig:
        """Crea configurazione training basata sul profilo"""
        
        # Base configuration
        base_config = {
            'initial_batch_size': self.batch_size,
            'use_mixed_precision': True,
            'early_stopping_restore_best_weights': True,
            'use_stochastic_weight_averaging': True
        }
        
        if self.optimization_profile == OptimizationProfile.HIGH_PERFORMANCE:
            base_config.update({
                'initial_learning_rate': 2e-3,  # Higher LR for speed
                'lr_scheduler_type': 'exponential',
                'early_stopping_patience': 10,  # Less patience
                'validation_frequency': 200,  # Less frequent validation
                'max_grad_norm': 1.0,
                'gradient_accumulation_steps': 2,
                'swa_start_epoch': 3  # Earlier SWA
            })
        
        elif self.optimization_profile == OptimizationProfile.STABLE_TRAINING:
            base_config.update({
                'initial_learning_rate': 5e-4,  # Conservative LR
                'lr_scheduler_type': 'plateau',
                'lr_patience': 8,
                'lr_factor': 0.7,  # Gentler reduction
                'early_stopping_patience': 25,  # More patience
                'early_stopping_min_delta': 1e-7,  # Smaller delta
                'validation_frequency': 50,  # Frequent validation
                'max_grad_norm': 0.5,  # Conservative clipping
                'warmup_steps': 200,  # Longer warmup
                'swa_start_epoch': 10
            })
        
        elif self.optimization_profile == OptimizationProfile.RESEARCH_MODE:
            base_config.update({
                'initial_learning_rate': 1e-3,
                'lr_scheduler_type': 'cosine',
                'early_stopping_patience': 50,  # Very patient
                'validation_frequency': 25,  # Very frequent
                'save_frequency': 100,  # Frequent saves
                'log_frequency': 10,  # Detailed logging
                'max_grad_norm': 2.0,  # Allow larger gradients
                'use_stochastic_weight_averaging': False  # Disable for clear analysis
            })
        
        else:  # PRODUCTION_READY
            base_config.update({
                'initial_learning_rate': 1e-3,
                'lr_scheduler_type': 'plateau',
                'early_stopping_patience': 20,
                'validation_frequency': 100,
                'max_grad_norm': 1.0,
                'warmup_steps': 100,
                'swa_start_epoch': 8,
                'amp_enabled': True  # Ensure AMP for production
            })
        
        return TrainingConfig(**base_config)
    
    def _create_monitor_config(self) -> MonitorConfig:
        """Crea configurazione monitoring basata sul profilo"""
        
        if self.optimization_profile == OptimizationProfile.HIGH_PERFORMANCE:
            return MonitorConfig(
                metrics_update_interval=5.0,
                memory_check_interval=10.0,
                health_check_interval=20.0,
                metrics_history_size=5000,
                enable_plots=False,
                memory_usage_threshold=0.9,
                gradient_explosion_threshold=50.0,
                enable_detailed_logging=True,
                save_metrics_to_file=True,
                plots_dir="./test_analyzer_data",
                metrics_file_format="json"
            )
        
        elif self.optimization_profile == OptimizationProfile.STABLE_TRAINING:
            return MonitorConfig(
                metrics_update_interval=2.0,
                memory_check_interval=5.0,
                health_check_interval=10.0,
                metrics_history_size=10000,
                enable_plots=True,
                memory_usage_threshold=0.8,
                gradient_explosion_threshold=10.0,
                loss_stagnation_threshold=50,
                enable_detailed_logging=True,
                save_metrics_to_file=True,
                plots_dir="./test_analyzer_data",
                metrics_file_format="json"
            )
        
        elif self.optimization_profile == OptimizationProfile.RESEARCH_MODE:
            return MonitorConfig(
                metrics_update_interval=1.0,
                memory_check_interval=2.0,
                health_check_interval=5.0,
                metrics_history_size=20000,
                enable_plots=True,
                save_plots=True,
                memory_usage_threshold=0.95,
                gradient_explosion_threshold=100.0,
                enable_detailed_logging=True,
                save_metrics_to_file=True,
                plots_dir="./test_analyzer_data",
                metrics_file_format="json"
            )
        
        else:  # PRODUCTION_READY
            return MonitorConfig(
                metrics_update_interval=3.0,
                memory_check_interval=8.0,
                health_check_interval=15.0,
                metrics_history_size=8000,
                enable_plots=False,
                memory_usage_threshold=0.85,
                gradient_explosion_threshold=20.0,
                enable_detailed_logging=False,
                save_metrics_to_file=True,
                plots_dir="./test_analyzer_data",
                metrics_file_format="json"
            )


class OptimizedTrainingManager:
    """
    Manager completo per training ottimizzato che orchestras tutti i componenti
    """
    
    def __init__(self, pipeline_config: OptimizedTrainingPipeline):
        self.config = pipeline_config
        
        # Initialize components
        self.preprocessor = AdvancedDataPreprocessor(pipeline_config.preprocessing_config)
        self.model = None  # Will be created in setup_model
        self.trainer = None  # Will be created in setup_trainer
        self.monitor = TrainingMonitor(pipeline_config.monitor_config)
        
        # Training state
        self.is_training = False
        self.training_history = {}
        
        # Create save directory
        Path(self.config.save_directory).mkdir(exist_ok=True)
        
        print(f"🔧 OptimizedTrainingManager initialized with {pipeline_config.optimization_profile.value} profile")
    
    def setup_model(self) -> OptimizedLSTM:
        """Setup del modello LSTM ottimizzato"""
        
        if self.config.lstm_config is None:
            raise ValueError("LSTM config not initialized")
        self.model = OptimizedLSTM(self.config.lstm_config)
        
        # Log model info
        model_info = self.model.get_model_info()
        print(f"🧠 Model created: {model_info['total_parameters']:,} parameters ({model_info['model_size_mb']:.2f} MB)")
        
        return self.model
    
    def setup_trainer(self) -> AdaptiveTrainer:
        """Setup del trainer adattivo"""
        
        if self.model is None:
            raise ValueError("Model must be setup before trainer")
        
        if self.config.training_config is None:
            raise ValueError("Training config not initialized")
        self.trainer = AdaptiveTrainer(
            model=self.model,
            config=self.config.training_config,
            save_dir=self.config.save_directory
        )
        
        return self.trainer
    
    def preprocess_data(self, data: np.ndarray, column_name: str = 'training_data') -> np.ndarray:
        """Preprocessing intelligente dei dati"""
        
        # Smart normalization
        normalized_data = self.preprocessor.smart_normalize(data, column_name)
        
        # Outlier detection and handling
        cleaned_data = self.preprocessor.detect_and_handle_outliers(normalized_data)
        
        # Adaptive windowing
        optimal_window = self.preprocessor.adaptive_windowing(cleaned_data)
        
        print(f"📊 Data preprocessed: {len(data)} → {len(cleaned_data)} samples, window={optimal_window}")
        
        return cleaned_data
    
    def train_model(self, train_data, train_targets, val_data=None, val_targets=None, 
                   num_epochs: int = 100) -> Dict[str, Any]:
        """Training completo del modello con monitoraggio"""
        
        if self.model is None or self.trainer is None:
            raise ValueError("Model and trainer must be setup before training")
        
        # Setup monitoring
        self.monitor.start_monitoring()
        
        # Create data loaders
        train_loader = self._create_data_loader(train_data, train_targets, self.config.batch_size)
        val_loader = self._create_data_loader(val_data, val_targets, self.config.batch_size) if val_data is not None else None
        
        # Loss function
        criterion = torch.nn.MSELoss()
        
        # Training loop
        self.is_training = True
        training_results = []
        
        try:
            for epoch in range(num_epochs):
                
                # Training step
                epoch_result = self.trainer.train_step(
                    data_loader=train_loader,
                    criterion=criterion,
                    validation_loader=val_loader
                )
                
                training_results.append(epoch_result)
                
                # Log to monitor
                self.monitor.log_training_step(
                    step=self.trainer.global_step,
                    loss=epoch_result['epoch_loss'],
                    learning_rate=epoch_result['current_lr'],
                    grad_norm=epoch_result['training_stats']['current_grad_norm']
                )
                
                if val_loader is not None:
                    self.monitor.log_validation_step(
                        step=self.trainer.global_step,
                        val_loss=epoch_result['training_stats']['current_val_loss']
                    )
                
                # Check early stopping
                if epoch_result['early_stop']:
                    print(f"🛑 Early stopping triggered at epoch {epoch}")
                    break
                
                # Progress report
                if epoch % 10 == 0:
                    health_info = self.monitor.get_current_status()['health_info']
                    print(f"Epoch {epoch}: Loss={epoch_result['epoch_loss']:.6f}, "
                          f"LR={epoch_result['current_lr']:.2e}, "
                          f"Health={health_info['overall_score']:.1f}")
        
        finally:
            self.is_training = False
            self.monitor.stop_monitoring()
        
        # Final results
        final_results = {
            'training_completed': True,
            'total_epochs': len(training_results),
            'final_loss': training_results[-1]['epoch_loss'] if training_results else None,
            'training_history': training_results,
            'model_info': self.model.get_model_info(),
            'preprocessing_report': self.preprocessor.get_preprocessing_report(),
            'training_summary': self.trainer.get_training_summary(),
            'monitoring_report': self.monitor.generate_training_report()
        }
        
        # Save results
        self._save_training_results(final_results)
        
        return final_results
    
    def _create_data_loader(self, data, targets, batch_size):
        """Crea PyTorch DataLoader"""
        
        if data is None or targets is None:
            return None
        
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(data),
            torch.FloatTensor(targets)
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Salva risultati training"""
        
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.model_name_prefix}_{self.config.model_type.value}_{timestamp}.json"
        filepath = Path(self.config.save_directory) / filename
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=convert_for_json)
        
        print(f"💾 Training results saved to {filepath}")
    
    def save_model(self, filename: Optional[str] = None) -> str:
        """Salva modello ottimizzato"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config.model_name_prefix}_{self.config.model_type.value}_{timestamp}.pt"
        
        filepath = Path(self.config.save_directory) / filename
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_config': asdict(self.config.lstm_config) if self.config.lstm_config else {},
            'pipeline_config': asdict(self.config),
            'model_info': self.model.get_model_info()
        }
        
        if self.trainer is not None:
            save_dict['trainer_state'] = self.trainer.get_training_summary()
        
        torch.save(save_dict, filepath)
        print(f"💾 Model saved to {filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: str) -> bool:
        """Carica modello ottimizzato"""
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            # Reconstruct model
            model_config = LSTMConfig(**checkpoint['model_config'])
            self.model = OptimizedLSTM(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False


# Factory functions for quick setup
def create_stable_training_pipeline(model_type: ModelType, input_features: int, **kwargs) -> OptimizedTrainingPipeline:
    """Crea pipeline ottimizzata per training stabile"""
    
    return OptimizedTrainingPipeline(
        model_type=model_type,
        input_features=input_features,
        optimization_profile=OptimizationProfile.STABLE_TRAINING,
        **kwargs
    )


def create_high_performance_pipeline(model_type: ModelType, input_features: int, **kwargs) -> OptimizedTrainingPipeline:
    """Crea pipeline ottimizzata per alta performance"""
    
    return OptimizedTrainingPipeline(
        model_type=model_type,
        input_features=input_features,
        optimization_profile=OptimizationProfile.HIGH_PERFORMANCE,
        **kwargs
    )


def create_research_pipeline(model_type: ModelType, input_features: int, **kwargs) -> OptimizedTrainingPipeline:
    """Crea pipeline ottimizzata per ricerca"""
    
    return OptimizedTrainingPipeline(
        model_type=model_type,
        input_features=input_features,
        optimization_profile=OptimizationProfile.RESEARCH_MODE,
        **kwargs
    )


def test_optimized_training():
    """Test completo del sistema di training ottimizzato"""
    print("🧪 Testing OptimizedTrainingManager...")
    
    # Create pipeline configuration
    pipeline = create_stable_training_pipeline(
        model_type=ModelType.LSTM_SUPPORT_RESISTANCE,
        input_features=10,
        sequence_length=20,
        batch_size=16
    )
    
    # Create manager
    manager = OptimizedTrainingManager(pipeline)
    
    # Setup model and trainer
    model = manager.setup_model()
    trainer = manager.setup_trainer()
    
    # Test preprocessing
    dummy_data = np.random.randn(1000, 10)
    preprocessed_data = manager.preprocess_data(dummy_data)
    
    print(f"✅ Pipeline created with {pipeline.optimization_profile.value} profile")
    print(f"✅ Model: {model.get_model_info()['total_parameters']:,} parameters")
    print(f"✅ Preprocessed data: {len(preprocessed_data)} samples")
    
    # Save model
    model_path = manager.save_model()
    print(f"✅ Model saved: {model_path}")
    
    return manager


if __name__ == "__main__":
    test_optimized_training()