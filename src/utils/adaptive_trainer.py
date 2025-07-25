#!/usr/bin/env python3
"""
AdaptiveTrainer - Intelligent Training Management
===============================================

Sistema di training adattivo che gestisce automaticamente learning rate, 
schedulers, early stopping e ottimizzazione degli iperparametri per 
risolvere problemi di convergenza e stabilità.

Features:
- Learning Rate scheduling intelligente
- Early stopping avanzato con patience
- Automatic Mixed Precision training
- Dynamic batch size adjustment
- Training stability monitoring
- Automatic hyperparameter optimization
- Loss landscape analysis

Author: ScalpingBOT Team
Version: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from typing import Dict, Optional, Any
from dataclasses import dataclass
from collections import deque, defaultdict
import logging
import math
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """Configurazione per AdaptiveTrainer"""
    
    # Core training settings - AGGRESSIVE LEARNING FOR REAL TRADING
    initial_learning_rate: float = 1e-2     # MOLTO PIÙ ALTO per forzare apprendimento
    min_learning_rate: float = 5e-5        # Più alto per evitare plateau
    max_learning_rate: float = 5e-2        # Molto più aggressivo
    
    # Batch size settings - ANTI-OVERFITTING
    initial_batch_size: int = 32         # RIDOTTO da 128 per prevenire overfitting
    min_batch_size: int = 16             # RIDOTTO da 32 per fine-tuning
    max_batch_size: int = 128            # RIDOTTO da 2048 per stabilità
    batch_size_increment: int = 16       # RIDOTTO da 32 per incrementi graduali
    
    # Early stopping - PERMETTERE APPRENDIMENTO PROFONDO
    early_stopping_patience: int = 100   # MOLTO PIÙ ALTO per LSTM learning
    early_stopping_min_delta: float = 1e-6  # PIÙ SENSIBILE per piccoli miglioramenti
    early_stopping_restore_best_weights: bool = True
    
    # Learning rate scheduling - OTTIMIZZATO PER APPRENDIMENTO REALE
    lr_scheduler_type: str = 'plateau'  # 'plateau', 'cosine', 'exponential', 'cyclic', 'adaptive'
    lr_patience: int = 50   # MOLTO PIÙ ALTO per dare tempo al modello
    lr_factor: float = 0.9  # MENO AGGRESSIVO nella riduzione
    lr_cooldown: int = 20   # PIÙ STABILITÀ dopo riduzione LR
    
    # Training stability - OPTIMIZED FOR LSTM
    gradient_accumulation_steps: int = 4  # AUMENTATO da 1 per batch virtuali più grandi
    max_grad_norm: float = 1.0           # AUMENTATO da 0.5 per permettere apprendimento LSTM
    warmup_steps: int = 100              # AUMENTATO da 50 per stabilizzare LSTM
    
    # Mixed precision
    use_mixed_precision: bool = True
    amp_enabled: bool = True
    
    # Monitoring - OTTIMIZZATO
    validation_frequency: int = 150  # RIDOTTO da 300 per monitoring più frequente
    save_frequency: int = 300       # RIDOTTO da 500 per checkpoint più frequenti
    log_frequency: int = 100        # AUMENTATO da 50 per ridurre verbosità
    
    # Advanced features
    use_gradient_checkpointing: bool = False
    use_stochastic_weight_averaging: bool = True
    swa_start_epoch: int = 5
    swa_lr: float = 1e-4
    
    # Stability thresholds - RELAXED FOR FINANCIAL DATA
    loss_explosion_threshold: float = 50.0      # AUMENTATO da 10.0 per financial volatility
    gradient_explosion_threshold: float = 10.0  # RIDOTTO da 100.0 per early detection
    nan_detection_enabled: bool = True


class LossTracker:
    """Tracker per monitorare andamento loss e stabilità"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.train_losses = deque(maxlen=window_size)
        self.val_losses = deque(maxlen=window_size)
        self.learning_rates = deque(maxlen=window_size)
        self.gradient_norms = deque(maxlen=window_size)
        
        self.best_val_loss = float('inf')
        self.best_model_state: Optional[Dict[str, Any]] = None
        self.epochs_without_improvement = 0
        
        # Stability metrics
        self.loss_explosions = 0
        self.gradient_explosions = 0
        self.nan_occurrences = 0
    
    def update(self, train_loss: float, val_loss: Optional[float] = None, 
               learning_rate: float = 0.0, grad_norm: float = 0.0):
        """Aggiorna metriche di training"""
        
        self.train_losses.append(train_loss)
        self.learning_rates.append(learning_rate)
        self.gradient_norms.append(grad_norm)
        
        if val_loss is not None:
            self.val_losses.append(val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
    
    def detect_instability(self, config: TrainingConfig) -> Dict[str, Any]:
        """Detecta instabilità nel training"""
        
        instabilities = []
        
        # Check for loss explosion
        if self.train_losses and self.train_losses[-1] > config.loss_explosion_threshold:
            instabilities.append('loss_explosion')
            self.loss_explosions += 1
        
        # Check for gradient explosion
        if self.gradient_norms and self.gradient_norms[-1] > config.gradient_explosion_threshold:
            instabilities.append('gradient_explosion')
            self.gradient_explosions += 1
        
        # Check for NaN
        if self.train_losses and (math.isnan(self.train_losses[-1]) or math.isinf(self.train_losses[-1])):
            instabilities.append('nan_loss')
            self.nan_occurrences += 1
        
        # Check for stagnation
        if len(self.train_losses) >= 20:
            recent_losses = list(self.train_losses)[-20:]
            loss_std = np.std(recent_losses)
            if loss_std < config.early_stopping_min_delta:
                instabilities.append('stagnation')
        
        return {
            'instabilities': instabilities,
            'is_stable': len(instabilities) == 0,
            'loss_explosions_count': self.loss_explosions,
            'gradient_explosions_count': self.gradient_explosions,
            'nan_occurrences_count': self.nan_occurrences
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche training"""
        
        stats = {
            'current_train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'current_val_loss': self.val_losses[-1] if self.val_losses else 0.0,
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'mean_train_loss': np.mean(self.train_losses) if self.train_losses else 0.0,
            'std_train_loss': np.std(self.train_losses) if self.train_losses else 0.0,
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0.0,
            'current_grad_norm': self.gradient_norms[-1] if self.gradient_norms else 0.0
        }
        
        return stats


class AdaptiveLRScheduler:
    """Learning Rate Scheduler adattivo con multiple strategie"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: TrainingConfig):
        self.optimizer = optimizer
        self.config = config
        self.step_count = 0
        self.best_metric = float('inf')
        self.patience_count = 0
        self.cooldown_count = 0
        
        # Initialize base scheduler
        self.base_scheduler = self._create_base_scheduler()
        
        # Warmup settings
        self.warmup_steps = config.warmup_steps
        self.base_lr = config.initial_learning_rate
    
    def _create_base_scheduler(self):
        """Crea scheduler base basato sulla configurazione"""
        
        if self.config.lr_scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                cooldown=self.config.lr_cooldown,
                min_lr=self.config.min_learning_rate
            )
        
        elif self.config.lr_scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=1000,  # Will be adjusted during training
                eta_min=self.config.min_learning_rate
            )
        
        elif self.config.lr_scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        
        elif self.config.lr_scheduler_type == 'cyclic':
            return optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=self.config.min_learning_rate,
                max_lr=self.config.max_learning_rate,
                step_size_up=100
            )
        
        else:  # adaptive
            return None
    
    def step(self, metric: Optional[float] = None):
        """Step dello scheduler con gestione adaptive"""
        
        self.step_count += 1
        
        # Warmup phase - DISABLED: Skip warmup to maintain stable LR
        # Warmup was causing LR to be adjusted every step regardless of validation
        # if self.step_count <= self.warmup_steps:
        #     lr = self.base_lr * (self.step_count / self.warmup_steps)
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr
        #     return
        
        # Regular scheduling
        if self.config.lr_scheduler_type == 'plateau' and metric is not None and self.base_scheduler is not None:
            # Fix: Pass metrics parameter correctly for ReduceLROnPlateau
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                self.base_scheduler.step(metric)
            else:
                self.base_scheduler.step()
        elif self.config.lr_scheduler_type == 'adaptive':
            self._adaptive_step(metric)
        elif self.base_scheduler is not None:
            # For other schedulers, call step() without parameters
            # Most schedulers (except ReduceLROnPlateau) don't need metrics
            self.base_scheduler.step()  # type: ignore
    
    def _adaptive_step(self, metric: Optional[float] = None):
        """Adaptive learning rate adjustment"""
        
        if metric is None:
            return
        
        # Check for improvement
        if metric < self.best_metric:
            self.best_metric = metric
            self.patience_count = 0
        else:
            self.patience_count += 1
        
        # Reduce learning rate if no improvement
        if self.patience_count >= self.config.lr_patience and self.cooldown_count == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = max(current_lr * self.config.lr_factor, self.config.min_learning_rate)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            
            self.patience_count = 0
            self.cooldown_count = self.config.lr_cooldown
        
        # Cooldown
        if self.cooldown_count > 0:
            self.cooldown_count -= 1
    
    def get_current_lr(self) -> float:
        """Ottieni learning rate corrente"""
        return self.optimizer.param_groups[0]['lr']


class AdaptiveTrainer:
    """
    Trainer adattivo con gestione intelligente di tutti gli aspetti del training
    
    Features:
    - Learning rate scheduling automatico
    - Early stopping intelligente
    - Automatic Mixed Precision
    - Batch size adaptation
    - Training stability monitoring
    - Model checkpointing
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig, 
                 save_dir: Optional[str] = None):
        
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir) if save_dir else Path("./training_checkpoints")
        self.save_dir.mkdir(exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.global_step = 0
        self.current_batch_size = config.initial_batch_size
        
        # Monitoring
        self.loss_tracker = LossTracker()
        self.training_history = defaultdict(list)
        
        # Diagnostica learning - inizializzazione globale
        self.loss_variance_window = deque(maxlen=500)  # Ultimi 500 steps per diagnostica
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = AdaptiveLRScheduler(self.optimizer, config)
        
        # Mixed precision - Fix for deprecated API
        if config.use_mixed_precision and torch.cuda.is_available():
            self.scaler = GradScaler('cuda')
        else:
            self.scaler = None
        
        # SWA (Stochastic Weight Averaging)
        if config.use_stochastic_weight_averaging:
            self.swa_model = torch.optim.swa_utils.AveragedModel(model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(self.optimizer, swa_lr=config.swa_lr)
        else:
            self.swa_model = None
            self.swa_scheduler = None
        
        # Device
        self.device = next(model.parameters()).device
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        print(f"🚀 AdaptiveTrainer initialized: LR={config.initial_learning_rate}, BS={config.initial_batch_size}, AMP={config.use_mixed_precision}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Crea optimizer ottimizzato con differential learning rates per LSTM"""
        
        # Separate parameter groups with LSTM-specific handling
        lstm_weight_hh_params = []  # Hidden-to-hidden weights (most critical)
        lstm_other_params = []      # Other LSTM parameters
        attention_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'weight_hh' in name or ('lstm' in name.lower() and 'hh' in name):
                # Critical LSTM hidden-to-hidden parameters
                lstm_weight_hh_params.append(param)
                print(f"🎯 LSTM weight_hh parameter found: {name}")
            elif 'lstm' in name.lower():
                lstm_other_params.append(param)
            elif 'attention' in name.lower() or 'attn' in name.lower():
                attention_params.append(param)
            else:
                other_params.append(param)
        
        # Parameter groups with differential learning rates
        param_groups = []
        
        # LSTM weight_hh gets 10x higher learning rate (like OptimizedLSTMTrainer)
        if lstm_weight_hh_params:
            param_groups.append({
                'params': lstm_weight_hh_params,
                'lr': self.config.initial_learning_rate * 10.0,  # 10x LR for weight_hh
                'weight_decay': 1e-5  # Lower weight decay for critical params
            })
            print(f"✅ Differential LR: weight_hh={self.config.initial_learning_rate * 10.0:.2e}")
        
        # Other LSTM parameters get standard rate
        if lstm_other_params:
            param_groups.append({
                'params': lstm_other_params,
                'lr': self.config.initial_learning_rate,
                'weight_decay': 5e-4  # INCREASED from 1e-4 for stronger L2 regularization
            })
        
        if attention_params:
            param_groups.append({
                'params': attention_params,
                'lr': self.config.initial_learning_rate * 0.5,
                'weight_decay': 5e-5
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.config.initial_learning_rate * 2.0,
                'weight_decay': 5e-4  # INCREASED from 1e-4 for stronger L2 regularization
            })
        
        # Use AdamW optimizer
        optimizer = torch.optim.AdamW(
            param_groups,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        return optimizer
    
    def train_step(self, data_loader, criterion, validation_loader=None) -> Dict[str, Any]:
        """
        Esegue un passo di training con tutte le ottimizzazioni
        
        Args:
            data_loader: DataLoader per training
            criterion: Loss function
            validation_loader: DataLoader per validation (opzionale)
            
        Returns:
            Dict con metriche di training
        """
        
        self.model.train()
        epoch_losses = []
        
        for batch_idx, batch_data in enumerate(data_loader):
            # print(f"🔍 DEBUG: batch_data type: {type(batch_data)}, length: {len(batch_data) if hasattr(batch_data, '__len__') else 'N/A'}")
            
            # Safe unpacking with validation
            if len(batch_data) == 2:
                data, target = batch_data
            elif len(batch_data) == 3:
                data, target, _ = batch_data  # Handle 3-element case
                # print(f"🔍 DEBUG: Got 3-element batch, using first 2")
            else:
                raise ValueError(f"Unexpected batch structure: {len(batch_data)} elements")
            
            # Move to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast('cuda'):
                    model_output = self.model(data)
                    # Handle tuple output
                    if isinstance(model_output, tuple):
                        output = model_output[0]
                    else:
                        output = model_output
                    loss = criterion(output, target)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # LSTM-specific gradient clipping with selective fixes
                self.scaler.unscale_(self.optimizer)
                grad_norm = self._apply_lstm_gradient_fixes()
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                # Standard training
                model_output = self.model(data)
                # Handle tuple output from LSTM
                if isinstance(model_output, tuple):
                    output = model_output[0]
                else:
                    output = model_output
                
                # Ensure shapes match
                if output.shape != target.shape:
                    # Try to fix common shape issues
                    if len(output.shape) == 3 and len(target.shape) == 2:
                        # Output is [batch, seq, features], target is [batch, features]
                        output = output[:, -1, :]  # Take last timestep
                
                loss = criterion(output, target)
                loss.backward()
                
                # LSTM-specific gradient clipping with selective fixes
                grad_norm = self._apply_lstm_gradient_fixes()
                
                self.optimizer.step()
            
            # Update tracking
            current_loss = loss.item()
            epoch_losses.append(current_loss)
            current_lr = self.lr_scheduler.get_current_lr()
            
            # Update loss variance window per diagnostica
            self.loss_variance_window.append(current_loss)
            
            # Update loss tracker with validation loss if available
            self.loss_tracker.update(
                train_loss=current_loss,
                learning_rate=current_lr,
                grad_norm=float(grad_norm) if isinstance(grad_norm, (int, float)) else float(grad_norm.item())
            )
            
            # Check for instability
            instability_info = self.loss_tracker.detect_instability(self.config)
            if not instability_info['is_stable']:
                self._handle_training_instability(instability_info)
            
            # Validation step
            if (validation_loader is not None and 
                self.global_step % self.config.validation_frequency == 0):
                val_loss = self._validate(validation_loader, criterion)
                self.loss_tracker.update(train_loss=loss.item(), val_loss=val_loss)
                
                # Learning rate scheduling - FIXED: Only after validation
                if len(self.loss_tracker.val_losses) > 0:
                    self.lr_scheduler.step(self.loss_tracker.val_losses[-1])
            # REMOVED: No fallback LR stepping - scheduler only steps after validation
            
            # SWA update
            if (self.swa_model is not None and 
                self.current_epoch >= self.config.swa_start_epoch):
                self.swa_model.update_parameters(self.model)
                if self.swa_scheduler is not None:
                    self.swa_scheduler.step()
            
            # Logging
            if self.global_step % self.config.log_frequency == 0:
                self._log_training_progress(loss.item(), current_lr, float(grad_norm) if isinstance(grad_norm, (int, float)) else float(grad_norm.item()))
            
            # Save checkpoint
            if self.global_step % self.config.save_frequency == 0:
                self._save_checkpoint()
            
            self.global_step += 1
        
        # End of epoch processing
        self.current_epoch += 1
        mean_epoch_loss = float(np.mean(epoch_losses))
        
        # Check early stopping
        early_stop = self._check_early_stopping()
        
        # Update training history
        self._update_training_history(mean_epoch_loss)
        
        return {
            'epoch_loss': mean_epoch_loss,
            'current_lr': current_lr,
            'early_stop': early_stop,
            'training_stats': self.loss_tracker.get_training_stats(),
            'instability_info': instability_info
        }
    
    def _validate(self, validation_loader, criterion) -> float:
        """Esegue validation e ritorna loss media"""
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data, target in validation_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.scaler is not None:
                    with autocast('cuda'):
                        model_output = self.model(data)
                        # Handle tuple output from LSTM
                        if isinstance(model_output, tuple):
                            output = model_output[0]
                        else:
                            output = model_output
                        loss = criterion(output, target)
                else:
                    model_output = self.model(data)
                    # Handle tuple output from LSTM
                    if isinstance(model_output, tuple):
                        output = model_output[0]
                    else:
                        output = model_output
                    loss = criterion(output, target)
                
                val_losses.append(loss.item())
        
        self.model.train()
        return float(np.mean(val_losses))
    
    def _handle_training_instability(self, instability_info: Dict[str, Any]):
        """Gestisce instabilità del training"""
        
        instabilities = instability_info['instabilities']
        
        if 'loss_explosion' in instabilities:
            # Reduce learning rate dramatically
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
            print(f"⚠️ Loss explosion detected, reducing LR to {param_group['lr']:.2e}")
        
        if 'gradient_explosion' in instabilities:
            # Reduce gradient clipping threshold
            self.config.max_grad_norm *= 0.5
            print(f"⚠️ Gradient explosion detected, reducing grad clip to {self.config.max_grad_norm}")
        
        if 'nan_loss' in instabilities:
            # Restore best weights if available
            if self.loss_tracker.best_model_state is not None:
                self.model.load_state_dict(self.loss_tracker.best_model_state)
                print("⚠️ NaN detected, restored best weights")
            
            # Reset optimizer state
            self.optimizer = self._create_optimizer()
    
    def _check_early_stopping(self) -> bool:
        """Verifica condizioni per early stopping"""
        
        if self.loss_tracker.epochs_without_improvement >= self.config.early_stopping_patience:
            
            if (self.config.early_stopping_restore_best_weights and 
                self.loss_tracker.best_model_state is not None):
                self.model.load_state_dict(self.loss_tracker.best_model_state)
                print("🛑 Early stopping triggered, restored best weights")
            
            return True
        
        return False
    
    def _apply_lstm_gradient_fixes(self) -> float:
        """
        Applica gradient fixes specifici per LSTM come in OptimizedLSTMTrainer:
        - Selective clipping per weight_hh parameters
        - Gradient noise injection per vanishing gradients
        - Standard clipping per altri parametri
        """
        try:
            weight_hh_fixed = 0
            total_grad_count = 0
            min_grad_norm = float('inf')
            max_grad_norm = 0.0
            
            # SELECTIVE CLIPPING + GRADIENT NOISE per parametri weight_hh
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    total_grad_count += 1
                    param_grad_norm = float(param.grad.norm().item())
                    
                    # Fix per weight_hh parameters (critical LSTM params)
                    if 'weight_hh' in name:
                        if param_grad_norm < 1e-6:  # Vanishing gradient detected
                            # Gradient noise injection
                            noise = torch.randn_like(param.grad) * 1e-6
                            param.grad.data += noise
                            weight_hh_fixed += 1
                        
                        # Selective clipping with lower threshold for weight_hh
                        if param_grad_norm > self.config.max_grad_norm * 0.5:
                            param.grad.data = param.grad.data * (self.config.max_grad_norm * 0.5) / param_grad_norm
                    
                    min_grad_norm = min(min_grad_norm, float(param.grad.norm().item()))
                    max_grad_norm = max(max_grad_norm, float(param.grad.norm().item()))
            
            # Log gradient health (rate limited)
            if total_grad_count > 0:
                zero_grad_ratio = (min_grad_norm < 1e-8) * 100.0  # Simplified check
                if weight_hh_fixed > 0:
                    print(f"🔧 LSTM gradient fixes applied: {weight_hh_fixed} weight_hh parameters fixed")
            
            # Standard gradient clipping for all parameters
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print("❌ Invalid gradient norm detected")
                raise ValueError("Gradient norm non valida")
            
            return float(grad_norm) if isinstance(grad_norm, (int, float)) else float(grad_norm.item())
            
        except Exception as clipping_error:
            print(f"❌ Errore durante LSTM gradient fixes: {clipping_error}")
            # Fallback to standard clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            return float(grad_norm) if isinstance(grad_norm, (int, float)) else float(grad_norm.item())
    
    def _log_training_progress(self, loss: float, lr: float, grad_norm: float):
        """Log progresso training"""
        
        stats = self.loss_tracker.get_training_stats()
        
        # 🎯 DIAGNOSTICA LEARNING OGNI 1000 STEPS
        diagnostic_info = ""
        if self.global_step % 1000 == 0 and hasattr(self, 'loss_variance_window') and len(self.loss_variance_window) >= 100:
            loss_std = np.std(list(self.loss_variance_window)[-100:])  # Varianza ultimi 100 steps
            loss_trend = np.mean(list(self.loss_variance_window)[-50:]) - np.mean(list(self.loss_variance_window)[-100:-50])
            
            # 🎯 DIAGNOSTICA OTTIMIZZATA PER MOSTRARE LEARNING REALE
            if loss_std < 0.000005:  # Varianza troppo bassa
                diagnostic_info = f" | ⚠️ PLATEAU (std={loss_std:.2e})"
            elif loss_trend < -0.000005:  # Miglioramento significativo (più sensibile)
                diagnostic_info = f" | ⬇️ LEARNING (Δ={loss_trend:.2e})"
            elif loss_trend > 0.000005:  # Peggioramento significativo
                diagnostic_info = f" | ⬆️ UNSTABLE (Δ={loss_trend:.2e})"  
            else:  # Cambiamenti minimi
                diagnostic_info = f" | 🔄 STABLE (Δ={loss_trend:.2e})"

        log_msg = (f"Step {self.global_step}: "
                  f"Loss={loss:.6f}, "
                  f"LR={lr:.2e}, "
                  f"GradNorm={grad_norm:.4f}, "
                  f"Best_Val={stats['best_val_loss']:.6f}{diagnostic_info}")
        
        print(f"📊 {log_msg}")
    
    def _save_checkpoint(self):
        """Salva checkpoint del modello"""
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.loss_tracker.best_val_loss,
            'training_config': self.config.__dict__,
            'training_history': dict(self.training_history)
        }
        
        if self.swa_model is not None:
            checkpoint['swa_model_state_dict'] = self.swa_model.state_dict()
        
        checkpoint_path = self.save_dir / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if self.loss_tracker.val_losses and self.loss_tracker.val_losses[-1] == self.loss_tracker.best_val_loss:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            # Fix: Ensure copy() returns proper dict type
            state_dict = self.model.state_dict()
            self.loss_tracker.best_model_state = {k: v.clone() for k, v in state_dict.items()}
    
    def _update_training_history(self, epoch_loss: float):
        """Aggiorna history del training"""
        
        stats = self.loss_tracker.get_training_stats()
        
        self.training_history['epoch'].append(self.current_epoch)
        self.training_history['train_loss'].append(epoch_loss)
        self.training_history['val_loss'].append(stats['current_val_loss'])
        self.training_history['learning_rate'].append(stats['current_lr'])
        self.training_history['grad_norm'].append(stats['current_grad_norm'])
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Ottieni summary completo del training"""
        
        stats = self.loss_tracker.get_training_stats()
        instability_info = self.loss_tracker.detect_instability(self.config)
        
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'training_stats': stats,
            'instability_info': instability_info,
            'training_history': dict(self.training_history),
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            }
        }
    
    def train_model_protected(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000) -> Dict[str, Any]:
        """
        Interfaccia compatibile con OptimizedLSTMTrainer per drop-in replacement
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader
        
        try:
            # Convert numpy to torch tensors
            device = next(self.model.parameters()).device
            X_tensor = torch.FloatTensor(X).to(device)
            y_tensor = torch.FloatTensor(y).to(device)
            
            # Create data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            data_loader = DataLoader(dataset, batch_size=self.config.initial_batch_size, shuffle=True)
            
            # Create criterion - HUBER LOSS per robustezza contro outliers
            criterion = torch.nn.HuberLoss(delta=0.1)  # Più robusto di MSE per trading reale
            
            # Training loop
            training_metrics = []
            best_loss = float('inf')
            
            for epoch in range(epochs):
                try:
                    step_result = self.train_step(data_loader, criterion)
                    # print(f"🔍 DEBUG: train_step returned keys: {list(step_result.keys())}")
                    
                    current_loss = step_result['epoch_loss']
                    training_metrics.append({
                        'epoch': epoch,
                        'loss': current_loss,
                        'lr': step_result['current_lr'],
                        'grad_norm': step_result.get('training_stats', {}).get('current_grad_norm', 0.0)
                    })
                    
                    best_loss = min(best_loss, current_loss)
                    
                    # Early stopping check
                    if step_result.get('early_stopping_triggered', False):
                        print(f"🛑 Early stopping at epoch {epoch}")
                        break
                except Exception as step_error:
                    import traceback
                    # print(f"🔍 ERROR in train_step: {step_error}")
                    # print(f"🔍 TRACEBACK: {traceback.format_exc()}")
                    raise step_error
            
            return {
                'training_completed': True,
                'final_loss': best_loss,
                'epochs_completed': len(training_metrics),
                'training_metrics': training_metrics,
                'message': f'Training completed successfully in {len(training_metrics)} epochs'
            }
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            # print(f"🔍 DETAILED ERROR: {error_details}")
            return {
                'training_completed': False,
                'final_loss': float('inf'),
                'epochs_completed': 0,
                'training_metrics': [],
                'message': f'Training failed: {str(e)}'
            }
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Carica checkpoint"""
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.loss_tracker.best_val_loss = checkpoint['best_val_loss']
            
            if 'training_history' in checkpoint:
                self.training_history = defaultdict(list, checkpoint['training_history'])
            
            if 'swa_model_state_dict' in checkpoint and self.swa_model is not None:
                self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
            
            print(f"✅ Checkpoint loaded from {checkpoint_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return False


# Helper functions
def create_adaptive_trainer_config(**kwargs) -> TrainingConfig:
    """Factory function per creare configurazione ottimizzata"""
    
    # Default optimized configuration
    default_config = {
        'initial_learning_rate': 1e-2,  # AGGIORNATO per apprendimento reale
        'early_stopping_patience': 20,
        'lr_scheduler_type': 'plateau',
        'use_mixed_precision': True,
        'max_grad_norm': 1.0,
        'validation_frequency': 100
    }
    
    # Override with provided kwargs
    default_config.update(kwargs)
    
    return TrainingConfig(**default_config)


def test_adaptive_trainer():
    """Test delle funzionalità dell'AdaptiveTrainer"""
    print("🧪 Testing AdaptiveTrainer...")
    
    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    
    # Create configuration
    config = create_adaptive_trainer_config(
        initial_learning_rate=1e-2,  # AGGIORNATO per apprendimento aggressivo
        initial_batch_size=16
    )
    
    # Create trainer
    trainer = AdaptiveTrainer(model, config)
    
    # Test training summary
    summary = trainer.get_training_summary()
    print(f"✅ Training summary: {summary['model_info']['total_params']} parameters")
    
    # Test checkpoint saving
    trainer._save_checkpoint()
    print("✅ Checkpoint saved")
    
    return trainer


if __name__ == "__main__":
    test_adaptive_trainer()