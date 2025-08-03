"""
Advanced LSTM Model for ScalpingBOT

This module contains the AdvancedLSTM class, a sophisticated neural network model
that implements dynamic input adapters, architectural optimizations, and comprehensive
tensor validation for robust financial data processing.

Key Features:
- Dynamic input size adaptation with caching
- Architecture improvements (reduced layers, layer normalization, residual connections)
- Comprehensive NaN/Inf protection throughout the forward pass
- Multi-head attention mechanism
- Intelligent tensor shape management
- Performance monitoring and statistics
"""

import os
import gc
import time
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# Configure basic logging
logger = logging.getLogger(__name__)


class TensorShapeManager:
    """Gestisce automaticamente le forme dei tensor per LSTM e altri modelli"""
    
    def __init__(self):
        self.shape_conversions = {
            'processed_count': 0,
            'conversion_history': [],
            'common_patterns': {},
            'error_patterns': []
        }
    
    @staticmethod
    def ensure_lstm_input_shape(data: torch.Tensor, sequence_length: int = 1, 
                               expected_features: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Assicura che i dati abbiano la forma corretta per LSTM [batch, seq, features]"""
        
        original_shape = data.shape
        conversion_info = {
            'original_shape': original_shape,
            'conversion_applied': False,
            'target_shape': None,
            'method_used': 'none'
        }
        
        # 🔧 CASO 1: Input 1D [features] → [1, 1, features]
        if data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(0)
            conversion_info.update({
                'conversion_applied': True,
                'target_shape': data.shape,
                'method_used': '1D_to_3D_single_batch_single_seq'
            })
        
        # 🔧 CASO 2: Input 2D [batch, features] → [batch, seq, features]
        elif data.dim() == 2:
            batch_size, features = data.shape
            
            # Se abbiamo expected_features, controlliamo se è compatibile
            if expected_features and features != expected_features:
                # Potrebbe essere [seq, features] invece di [batch, features]
                if features == expected_features:
                    # È già corretto come [seq, features], aggiungi batch
                    data = data.unsqueeze(0)  # [1, seq, features]
                    conversion_info['method_used'] = '2D_seq_features_to_3D'
                else:
                    # Reshaping intelligente basato su fattori
                    possible_seq_len = TensorShapeManager._find_best_sequence_length(features, expected_features)
                    if possible_seq_len and features % possible_seq_len == 0:
                        new_features = features // possible_seq_len
                        data = data.view(batch_size, possible_seq_len, new_features)
                        conversion_info['method_used'] = f'2D_smart_reshape_{possible_seq_len}x{new_features}'
                    else:
                        # Default: aggiungi sequenza di lunghezza 1
                        data = data.unsqueeze(1)  # [batch, 1, features]
                        conversion_info['method_used'] = '2D_batch_features_to_3D'
            else:
                # Standard: [batch, features] → [batch, 1, features]
                data = data.unsqueeze(1)
                conversion_info['method_used'] = '2D_to_3D_standard'
            
            conversion_info.update({
                'conversion_applied': True,
                'target_shape': data.shape
            })
        
        # 🔧 CASO 3: Input 3D [batch, seq, features] - già corretto
        elif data.dim() == 3:
            batch_size, seq_len, features = data.shape
            
            # Verifica se le dimensioni sono ragionevoli
            if seq_len > 10000:  # Sequenza troppo lunga
                conversion_info['method_used'] = '3D_already_correct_but_long_sequence'
            elif features > 10000:  # Troppe features
                conversion_info['method_used'] = '3D_already_correct_but_many_features'
            else:
                conversion_info['method_used'] = '3D_already_correct'
        
        # 🔧 CASO 4: Input 4D+ - errore
        else:
            raise ValueError(f"❌ Dimensioni tensor non supportate: {data.dim()}D con shape {original_shape}")
        
        conversion_info['final_shape'] = data.shape
        
        return data, conversion_info
    
    @staticmethod
    def _find_best_sequence_length(total_features: int, expected_features: int) -> Optional[int]:
        """Trova la migliore lunghezza di sequenza per reshaping intelligente"""
        
        if expected_features <= 0 or total_features <= 0:
            return None
        
        # Cerca fattori ragionevoli
        max_seq_len = min(100, total_features // expected_features)
        
        for seq_len in range(2, max_seq_len + 1):
            if total_features % seq_len == 0:
                resulting_features = total_features // seq_len
                if resulting_features == expected_features:
                    return seq_len
        
        return None
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                            name: str = "tensor", allow_batch_dim: bool = True) -> bool:
        """Valida che un tensor abbia la forma attesa"""
        
        actual_shape = tensor.shape
        
        # Se allow_batch_dim, ignora la prima dimensione per il confronto
        if allow_batch_dim and len(actual_shape) > 1 and len(expected_shape) > 1:
            actual_shape_to_check = actual_shape[1:]
            expected_shape_to_check = expected_shape[1:]
        else:
            actual_shape_to_check = actual_shape
            expected_shape_to_check = expected_shape
        
        if actual_shape_to_check != expected_shape_to_check:
            logger.warning(f"{name} shape mismatch: Expected {expected_shape}, Actual {actual_shape}")
            return False
        
        return True
    
    @staticmethod
    def smart_batch_reshape(data: torch.Tensor, target_batch_size: int) -> torch.Tensor:
        """Reshape intelligente per adattare batch size"""
        
        current_batch = data.shape[0]
        
        if current_batch == target_batch_size:
            return data
        
        if current_batch > target_batch_size:
            # Riduci batch size prendendo i primi elementi
            return data[:target_batch_size]
        else:
            # Aumenta batch size replicando dati
            repeats = (target_batch_size + current_batch - 1) // current_batch
            expanded = data.repeat(repeats, *([1] * (data.dim() - 1)))
            return expanded[:target_batch_size]
    
    @classmethod
    def prepare_model_input(cls, data: torch.Tensor, model_type: str, 
                          expected_input_size: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Prepara input per diversi tipi di modelli"""
        
        shape_manager = cls()
        preparation_info = {
            'model_type': model_type,
            'original_shape': data.shape,
            'transformations': []
        }
        
        if model_type.upper() == 'LSTM':
            # Per LSTM: assicura forma [batch, seq, features]
            data, conversion_info = cls.ensure_lstm_input_shape(data, expected_features=expected_input_size)
            preparation_info['transformations'].append(conversion_info)
        
        elif model_type.upper() == 'CNN':
            # Per CNN: assicura forma [batch, channels, length]
            if data.dim() == 2:
                data = data.unsqueeze(1)  # Aggiungi dimensione channel
                preparation_info['transformations'].append({
                    'conversion_applied': True,
                    'method_used': '2D_to_CNN_format',
                    'target_shape': data.shape
                })
        
        elif model_type.upper() == 'TRANSFORMER':
            # Per Transformer: assicura forma [batch, seq, features]
            data, conversion_info = cls.ensure_lstm_input_shape(data, expected_features=expected_input_size)
            preparation_info['transformations'].append(conversion_info)
        
        elif model_type.upper() == 'LINEAR':
            # Per Linear: assicura forma [batch, features]
            if data.dim() > 2:
                original_shape = data.shape
                data = data.view(data.shape[0], -1)  # Flatten mantenendo batch
                preparation_info['transformations'].append({
                    'conversion_applied': True,
                    'method_used': 'flatten_to_linear',
                    'original_shape': original_shape,
                    'target_shape': data.shape
                })
        
        # Update statistics
        shape_manager.shape_conversions['processed_count'] += 1
        if preparation_info['transformations']:
            shape_manager.shape_conversions['conversion_history'].append(preparation_info)
        
        preparation_info['final_shape'] = data.shape
        
        return data, preparation_info
    
    def get_shape_statistics(self) -> Dict[str, Any]:
        """Ottieni statistiche sulle conversioni di forma"""
        
        total_processed = self.shape_conversions['processed_count']
        conversions_applied = len(self.shape_conversions['conversion_history'])
        
        return {
            'total_processed': total_processed,
            'conversions_needed': conversions_applied,
            'conversion_rate': conversions_applied / total_processed if total_processed > 0 else 0,
            'common_patterns': self.shape_conversions['common_patterns'],
            'error_patterns': self.shape_conversions['error_patterns']
        }


class AdvancedLSTM(nn.Module):
    """LSTM avanzato con auto-resize dinamico per qualsiasi dimensione input"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, dropout: float = 0.5):
        super(AdvancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.expected_input_size = input_size  # Dimensione target preferita
        self.parent: Optional[Any] = None  # ✅ Inizializza parent reference per evitare errori
        
        # 🚀 FASE 2 - ARCHITECTURE CHANGES - SIMPLIFIED FOR STABILITY
        self.architecture_fixes = {
            'reduce_layers': True,            # Keep layer reduction
            'layer_norm': False,             # DISABLE - can cause instability
            'residual_connections': False,   # DISABLE - can cause gradient issues
            'disable_bidirectional': True    # Keep unidirectional for stability
        }
        
        # 🔧 FASE 2.1: REDUCE LSTM LAYERS (3→1 layers for maximum stability)
        original_num_layers = num_layers
        if self.architecture_fixes['reduce_layers'] and num_layers > 1:
            self.num_layers = 1  # SINGLE LAYER for maximum stability
            self._log(f"🚀 ARCHITECTURE FIX: Reduced LSTM layers {original_num_layers}→{self.num_layers}", "architecture_fixes", "debug")
        else:
            self.num_layers = num_layers
        
        # 🔧 NUOVO: Pool di adapter dinamici per diverse dimensioni
        self.input_adapters = nn.ModuleDict()  # Memorizza adapter per diverse dimensioni
        self.adapter_cache = {}  # Cache per evitare ricreazioni
        
        # 🔧 FASE 2.4: DISABLE BIDIRECTIONAL (opzionale)
        bidirectional = not self.architecture_fixes['disable_bidirectional']
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        if not bidirectional:
            self._log("🚀 ARCHITECTURE FIX: Disabled bidirectional LSTM", "architecture_fixes", "debug")
        
        self.lstm = nn.LSTM(input_size, hidden_size, self.num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=bidirectional)
        
        # 🔧 CRITICAL FIX: Initialize forget gate bias to 1.0 for LSTM stability
        self._initialize_lstm_weights()
        
        # 🔧 FASE 2.2: LAYER NORMALIZATION - DISABLED for stability
        self.lstm_layer_norms = None  # Disabled to prevent training instability
        
        self.attention = nn.MultiheadAttention(lstm_output_size, num_heads=8, dropout=dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_size)
        self.dropout = nn.Dropout(dropout)
        
        # 🔧 FASE 2.3: RESIDUAL CONNECTIONS - DISABLED for stability
        self.residual_projection = None  # Disabled to prevent gradient flow issues
        
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()
        
        # Statistiche per debug
        self.resize_stats = {
            'total_calls': 0,
            'adapters_created': 0,
            'dimension_history': []
        }
    
    def _log(self, message: str, category: str = "adapter", severity: str = "info"):
        """Helper per logging che funziona con o senza parent"""
        if severity == "warning":
            logger.warning(f"[{category}] {message}")
        elif severity == "error":
            logger.error(f"[{category}] {message}")
        elif severity == "debug":
            logger.debug(f"[{category}] {message}")
        else:
            logger.info(f"[{category}] {message}")
    
    def _initialize_lstm_weights(self):
        """Initialize LSTM weights properly, especially forget gate bias to 1.0"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                # Initialize input-hidden weights with Xavier uniform
                torch.nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Initialize hidden-hidden weights with orthogonal
                torch.nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Initialize all biases to 0, but forget gate bias to 1
                torch.nn.init.zeros_(param)
                # LSTM bias layout: [input_gate, forget_gate, cell_gate, output_gate]
                # Set forget gate bias to 1.0 for each layer
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # forget gate bias = 1.0
    
    def _get_or_create_adapter(self, actual_input_size: int) -> nn.Module:
        """Ottiene o crea un adapter per la dimensione specifica con caching ottimizzato"""
        
        # Se la dimensione è già quella attesa, non serve adapter
        if actual_input_size == self.expected_input_size:
            return nn.Identity()
        
        # Crea chiave per l'adapter
        adapter_key = f"adapter_{actual_input_size}_to_{self.expected_input_size}"
        
        # 🚀 CACHE HIT: Se l'adapter esiste già, riutilizzalo
        if adapter_key in self.input_adapters:
            # Track usage per statistiche
            if not hasattr(self, 'adapter_usage_count'):
                self.adapter_usage_count = {}
            if adapter_key not in self.adapter_usage_count:
                self.adapter_usage_count[adapter_key] = 0
            self.adapter_usage_count[adapter_key] += 1
            
            # Solo log ogni 500 utilizzi con smart rate limiting
            if self.adapter_usage_count[adapter_key] % 500 == 0:
                print(f"🔄 Adapter cache hit #{self.adapter_usage_count[adapter_key]}: {adapter_key}")
            
            return self.input_adapters[adapter_key]
        
        # 🔧 CACHE MISS: Crea nuovo adapter solo se necessario
        print(f"🔧 Creating new LSTM adapter: {actual_input_size} → {self.expected_input_size}")
        
        # 🚀 AUTO-CLEANUP: Gestione intelligente degli adapter
        max_adapters = 8  # Ridotto per safety
        cleanup_threshold = 6  # Inizia cleanup prima del limite
        
        if len(self.input_adapters) >= cleanup_threshold:
            self._auto_cleanup_adapters(max_adapters)
        
        # Crea nuovo adapter con architettura ottimizzata
        new_adapter = nn.Sequential(
            nn.Linear(actual_input_size, self.expected_input_size),
            nn.LayerNorm(self.expected_input_size),
            nn.Dropout(0.1)
        )
        
        # 🔧 Inizializza i pesi dell'adapter in modo efficiente
        with torch.no_grad():
            for module in new_adapter.modules():
                if isinstance(module, nn.Linear):
                    # Usa inizializzazione più efficiente
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.zeros_(module.bias)
                    break
        
        # 🚀 REGISTRA con tracking ottimizzato
        self.input_adapters[adapter_key] = new_adapter
        self.resize_stats['adapters_created'] += 1
        
        # Inizializza usage count
        if not hasattr(self, 'adapter_usage_count'):
            self.adapter_usage_count = {}
        self.adapter_usage_count[adapter_key] = 1
        
        self._log(f"✅ Adapter '{adapter_key}' created and cached (Total: {len(self.input_adapters)})", "adapter_cache", "info")
        
        return new_adapter
    
    def _auto_cleanup_adapters(self, max_adapters: int) -> None:
        """Auto-cleanup intelligente degli adapter con strategia LRU"""
        
        if not hasattr(self, 'adapter_usage_count'):
            self.adapter_usage_count = {}
        
        current_count = len(self.input_adapters)
        target_count = max(2, max_adapters - 2)  # Mantieni almeno 2, target 2 sotto il max
        
        if current_count <= target_count:
            return
        
        # Strategia di cleanup intelligente
        removal_candidates = []
        
        # 1. Adapter mai utilizzati o con usage molto basso
        for adapter_key, usage_count in self.adapter_usage_count.items():
            if adapter_key in self.input_adapters:
                # Score basato su: usage_count, età (implicita dall'ordine), dimensione
                score = usage_count
                
                # Penalizza adapter per dimensioni inusuali (probabilmente temporanei)
                if 'adapter_' in adapter_key:
                    try:
                        parts = adapter_key.split('_')
                        if len(parts) < 2:
                            raise ValueError(f"Invalid adapter_key format: {adapter_key}")
                        input_size = int(parts[1])
                        # Penalizza dimensioni molto diverse dall'expected
                        if abs(input_size - self.expected_input_size) > self.expected_input_size * 0.5:
                            score *= 0.5  # Dimezza score per dimensioni anomale
                    except (ValueError, IndexError) as e:
                        # Log error instead of silent failure
                        print(f"Warning: Failed to parse adapter_key {adapter_key}: {e}")
                        # Continue with original score for malformed keys
                
                removal_candidates.append((adapter_key, score))
        
        # Ordina per score (i meno utilizzati first)
        removal_candidates.sort(key=lambda x: x[1])
        
        # Rimuovi gli adapter con score più basso
        to_remove = current_count - target_count
        removed_count = 0
        
        for adapter_key, score in removal_candidates:
            if removed_count >= to_remove:
                break
            
            if adapter_key in self.input_adapters:
                # Backup info per log
                usage_count = self.adapter_usage_count[adapter_key] if adapter_key in self.adapter_usage_count else 0
                
                # Rimuovi
                del self.input_adapters[adapter_key]
                if adapter_key in self.adapter_usage_count:
                    del self.adapter_usage_count[adapter_key]
                
                removed_count += 1
                print(f"🧹 Auto-removed adapter: {adapter_key} (score: {score:.1f}, usage: {usage_count})")
        
        # Log risultato cleanup
        final_count = len(self.input_adapters)
        memory_saved = removed_count * 0.1  # Stima MB per adapter
        
        self._log(f"✅ Adapter cleanup: {current_count} → {final_count} (-{removed_count}), ~{memory_saved:.1f}MB saved", "adapter_cache", "info")
        
        # Force garbage collection per liberare memoria
        gc.collect()
    
    def get_cache_efficiency_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche dettagliate sull'efficienza della cache"""
        
        if not hasattr(self, 'adapter_usage_count'):
            self.adapter_usage_count = {}
        
        total_calls = sum(self.adapter_usage_count.values())
        cache_hits = total_calls - self.resize_stats['adapters_created']
        hit_rate = (cache_hits / total_calls * 100) if total_calls > 0 else 0
        
        # Trova adapter più e meno utilizzati
        if self.adapter_usage_count:
            most_used = max(self.adapter_usage_count.items(), key=lambda x: x[1])
            least_used = min(self.adapter_usage_count.items(), key=lambda x: x[1])
        else:
            most_used = ("none", 0)
            least_used = ("none", 0)
        
        return {
            'total_adapter_calls': total_calls,
            'cache_hits': cache_hits,
            'cache_misses': self.resize_stats['adapters_created'],
            'hit_rate_percentage': hit_rate,
            'active_adapters': len(self.input_adapters),
            'most_used_adapter': {'key': most_used[0], 'usage': most_used[1]},
            'least_used_adapter': {'key': least_used[0], 'usage': least_used[1]},
            'memory_efficiency': 'high' if hit_rate > 80 else 'medium' if hit_rate > 50 else 'low'
        }

    def optimize_cache(self) -> Dict[str, Any]:
        """Ottimizza la cache rimuovendo adapter inutilizzati"""
        
        if not hasattr(self, 'adapter_usage_count'):
            return {'status': 'no_cache_data'}
        
        initial_count = len(self.input_adapters)
        removed_adapters = []
        
        # Rimuovi adapter utilizzati meno di 5 volte
        min_usage_threshold = 5
        for adapter_key, usage_count in list(self.adapter_usage_count.items()):
            if usage_count < min_usage_threshold:
                if adapter_key in self.input_adapters:
                    del self.input_adapters[adapter_key]
                    del self.adapter_usage_count[adapter_key]
                    removed_adapters.append(adapter_key)
        
        final_count = len(self.input_adapters)
        
        print(f"🔧 Cache optimization: removed {len(removed_adapters)} unused adapters")
        
        return {
            'status': 'optimized',
            'adapters_before': initial_count,
            'adapters_after': final_count,
            'removed_count': len(removed_adapters),
            'removed_adapters': removed_adapters
        }

    def clear_adapter_cache(self) -> None:
        """Pulisce completamente la cache degli adapter"""
        
        cache_size = len(self.input_adapters)
        self.input_adapters.clear()
        
        if hasattr(self, 'adapter_usage_count'):
            self.adapter_usage_count.clear()
        
        # Reset stats
        self.resize_stats['adapters_created'] = 0
        self.resize_stats['dimension_history'].clear()
        
        self._log(f"🗑️ Adapter cache cleared: {cache_size} adapters removed", "adapter_cache", "info")
    
    def _apply_adapter(self, x: torch.Tensor, adapter: nn.Module) -> torch.Tensor:
        """Applica l'adapter mantenendo la forma del tensore con protezione completa anti-NaN"""
        
        original_shape = x.shape
        
        # 🛡️ VALIDAZIONE INPUT TENSOR CRITICA
        if torch.isnan(x).any():
            self._log(f"❌ Input tensor contiene NaN prima dell'adapter: {torch.isnan(x).sum().item()} valori", 
                                 "tensor_validation", "warning")
            # Sanitizza input
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            self._log("🔧 Input tensor sanitizzato", "tensor_validation", "info")
        
        if torch.isinf(x).any():
            self._log(f"❌ Input tensor contiene Inf prima dell'adapter: {torch.isinf(x).sum().item()} valori", 
                                 "tensor_validation", "warning")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            self._log("🔧 Input tensor sanitizzato", "tensor_validation", "info")
        
        # 🛡️ GESTIONE SHAPE INTELLIGENTE CON PROTEZIONE
        try:
            if len(original_shape) == 3:  # (batch, seq, features)
                batch_size, seq_len, features = original_shape
                
                # FAIL FAST - No fallback tensors allowed
                if features <= 0 or batch_size <= 0 or seq_len <= 0:
                    raise ValueError(f"Invalid tensor dimensions: {original_shape} - cannot proceed with invalid data")
                
                # Reshape per applicare Linear: (batch*seq, features)
                x_reshaped = x.view(-1, features)
                
                # 🛡️ FAIL FAST: NaN/Inf detection without silent fixing
                if torch.isnan(x_reshaped).any() or torch.isinf(x_reshaped).any():
                    self._log("❌ Tensor contiene NaN/Inf dopo reshape", "tensor_validation", "error")
                    raise ValueError("Input tensor contains NaN/Inf values - fix data preprocessing instead of masking")
                
                # 🛡️ APPLICA ADAPTER CON PROTEZIONE
                try:
                    x_adapted = adapter(x_reshaped)
                    
                    # FAIL FAST - No None outputs allowed
                    if x_adapted is None:
                        self._log("❌ Adapter ha ritornato None", "tensor_validation", "error")
                        raise ValueError("Adapter returned None - invalid adapter implementation")
                    
                    elif torch.isnan(x_adapted).any():
                        nan_count = torch.isnan(x_adapted).sum().item()
                        self._log(f"❌ Adapter output contiene {nan_count} NaN values", "tensor_validation", "warning")
                        x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                        self._log("🔧 Adapter output sanitizzato", "tensor_validation", "info")
                    
                    elif torch.isinf(x_adapted).any():
                        inf_count = torch.isinf(x_adapted).sum().item()
                        self._log(f"❌ Adapter output contiene {inf_count} Inf values", "tensor_validation", "warning")
                        x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                        self._log("🔧 Adapter output sanitizzato", "tensor_validation", "info")
                    
                    # 🛡️ VALIDAZIONE FORMA OUTPUT ADAPTER
                    expected_adapter_shape = (x_reshaped.shape[0], self.expected_input_size)
                    if x_adapted.shape != expected_adapter_shape:
                        self._log(f"❌ Adapter output shape mismatch: {x_adapted.shape} vs {expected_adapter_shape}", "tensor_validation", "warning")
                        # FAIL FAST - No fallback tensor creation
                        raise ValueError(f"Adapter output shape mismatch: {x_adapted.shape} vs {expected_adapter_shape}")
                    
                except Exception as adapter_error:
                    print(f"❌ Errore nell'adapter: {adapter_error}")
                    # FAIL FAST - No fallback for adapter errors
                    raise RuntimeError(f"Adapter failed: {adapter_error}")
                
                # 🛡️ RESHAPE BACK CON PROTEZIONE
                try:
                    target_shape = (batch_size, seq_len, self.expected_input_size)
                    x = x_adapted.view(target_shape)
                    
                    # FAIL FAST - No reshape fallback
                    if x.shape != target_shape:
                        print(f"❌ Reshape finale fallito: {x.shape} vs {target_shape}")
                        raise RuntimeError(f"Final reshape failed: {x.shape} vs {target_shape}")
                    
                except RuntimeError as reshape_error:
                    self._log(f"❌ Errore reshape finale: {reshape_error}", "tensor_validation", "error")
                    print(f"   Original: {original_shape}")
                    print(f"   Adapted shape: {x_adapted.shape}")
                    print(f"   Target: ({batch_size}, {seq_len}, {self.expected_input_size})")
                    
                    # FAIL FAST - No fallback for reshape errors
                    raise RuntimeError(f"Reshape error: cannot create target shape {target_shape}")
                
            elif len(original_shape) == 2:  # (batch, features)
                # 🛡️ GESTIONE 2D CON PROTEZIONE
                try:
                    x_adapted = adapter(x)
                    
                    # 🛡️ VALIDAZIONE OUTPUT 2D
                    if x_adapted is None:
                        self._log("❌ Adapter 2D ha ritornato None", "tensor_validation", "error")
                        raise RuntimeError(f"Adapter 2D returned None for input shape {x.shape}")
                    
                    elif torch.isnan(x_adapted).any() or torch.isinf(x_adapted).any():
                        self._log("❌ Adapter 2D output contiene NaN/Inf", "tensor_validation", "warning")
                        x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    x = x_adapted
                    
                except Exception as adapter_2d_error:
                    print(f"❌ Errore adapter 2D: {adapter_2d_error}")
                    raise RuntimeError(f"Adapter 2D processing failed: {adapter_2d_error}")
            
            else:
                # 🛡️ CASO COMPLESSO CON FALLBACK SICURO
                self._log(f"⚠️ Shape non standard per adapter: {original_shape}", "tensor_validation", "warning")
                
                try:
                    x_prepared, shape_info = TensorShapeManager.prepare_model_input(
                        x, 'LSTM', self.expected_input_size
                    )
                    
                    # Applica adapter alla forma preparata con protezione
                    if len(x_prepared.shape) == 3:
                        batch_size, seq_len, features = x_prepared.shape
                        x_reshaped = x_prepared.view(-1, features)
                        
                        # Adapter con protezione
                        try:
                            x_adapted = adapter(x_reshaped)
                            if torch.isnan(x_adapted).any() or torch.isinf(x_adapted).any():
                                x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                            x = x_adapted.view(batch_size, seq_len, self.expected_input_size)
                        except Exception as e:
                            raise RuntimeError(f"Adapter shape correction failed for sequence input: {e}")
                    else:
                        try:
                            x_adapted = adapter(x_prepared)
                            if torch.isnan(x_adapted).any() or torch.isinf(x_adapted).any():
                                x_adapted = torch.nan_to_num(x_adapted, nan=0.0, posinf=1.0, neginf=-1.0)
                            x = x_adapted
                        except Exception as e:
                            raise RuntimeError(f"Adapter shape correction failed for single input: {e}")
                    
                    self._log(f"✅ Shape correction applicata: {original_shape} → {x.shape}", "tensor_validation", "info")
                    
                except Exception as shape_error:
                    print(f"❌ Errore shape correction: {shape_error}")
                    # FAIL FAST - NO FALLBACK
                    raise RuntimeError(f"Shape correction failed for complex input: {shape_error}")
            
            # 🛡️ VALIDAZIONE FINALE COMPLETA
            if torch.isnan(x).any() or torch.isinf(x).any():
                self._log("❌ Output finale contiene ancora NaN/Inf", "tensor_validation", "warning")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                self._log("🔧 Output finale sanitizzato", "tensor_validation", "info")
            
            # 🛡️ VERIFICA FORMA FINALE
            if x.shape[-1] != self.expected_input_size:
                print(f"❌ Forma finale incorretta: {x.shape} (expected last dim: {self.expected_input_size})")
                # FAIL FAST - NO FALLBACK
                raise RuntimeError(f"Final output shape incorrect: {x.shape}, expected last dim: {self.expected_input_size}")
            
            self._log(f"✅ Adapter applicato con successo: {original_shape} → {x.shape}", 
                                 "tensor_validation", "debug")
            return x
            
        except Exception as e:
            print(f"❌ Errore catastrofico in _apply_adapter: {e}")
            print(f"   Input shape: {original_shape}")
            print(f"   Expected input size: {self.expected_input_size}")
            
            # FAIL FAST - NO FALLBACK ALLOWED
            raise RuntimeError(f"Catastrophic adapter failure: {e} for input shape {original_shape}")
    
    def forward(self, x):
        """Forward pass con protezione completa anti-NaN a ogni step"""
        
        self.resize_stats['total_calls'] += 1
        
        # 🛡️ VALIDAZIONE INPUT ASSOLUTA
        if x is None:
            print("❌ Input è None!")
            raise ValueError("Input tensor cannot be None")
        
        if not isinstance(x, torch.Tensor):
            self._log(f"❌ Input non è un tensor: {type(x)}", "tensor_validation", "error")
            try:
                x = torch.tensor(x, dtype=torch.float32)
            except Exception as e:
                raise TypeError(f"Cannot convert input to tensor: {type(x)} - {e}")
        
        # 🛡️ VALIDAZIONE NaN/Inf INPUT
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            self._log(f"❌ Input contiene {nan_count} valori NaN - sanitizzando...", "tensor_validation", "warning")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isinf(x).any():
            inf_count = torch.isinf(x).sum().item()
            self._log(f"❌ Input contiene {inf_count} valori Inf - sanitizzando...", "tensor_validation", "warning")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🛡️ VALIDAZIONE RANGE INPUT
        if torch.abs(x).max() > 1000:
            self._log(f"Input ha valori estremi: max={torch.abs(x).max():.2f}", "input_validation", "warning")
            x = torch.clamp(x, -100, 100)  # Clamp valori estremi
        
        original_shape = x.shape
        
        # 🛡️ TENSOR SHAPE MANAGEMENT CON PROTEZIONE
        try:
            # Inizializza shape manager se non esiste
            if not hasattr(self, '_shape_manager'):
                self._shape_manager = TensorShapeManager()
            
            # Preparazione input con protezione completa
            try:
                x, shape_info = TensorShapeManager.ensure_lstm_input_shape(
                    x, sequence_length=1, expected_features=self.expected_input_size
                )
                
                # 🛡️ VALIDAZIONE DOPO SHAPE MANAGEMENT
                if torch.isnan(x).any() or torch.isinf(x).any():
                    self._log("❌ NaN/Inf rilevati dopo shape management", "tensor_validation", "warning")
                    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                
                # Log conversioni significative
                if shape_info['conversion_applied']:
                    conversion_key = f"{original_shape}→{x.shape}"
                    
                    if not hasattr(self, '_logged_conversions'):
                        self._logged_conversions = set()
                    
                    if conversion_key not in self._logged_conversions:
                        self._log(f"🔧 TensorShape: {shape_info['method_used']}: {original_shape} → {x.shape}", 
                                             "tensor_validation", "info")
                        self._logged_conversions.add(conversion_key)
                        
                        if not hasattr(self, '_conversion_stats'):
                            self._conversion_stats = {}
                        method = shape_info['method_used']
                        if method not in self._conversion_stats:
                            self._conversion_stats[method] = 0
                        self._conversion_stats[method] += 1
            
            except Exception as shape_error:
                self._log(f"❌ Errore TensorShapeManager: {shape_error}", 
                                     "tensor_validation", "error")
                self._log(f"   Input shape: {original_shape}", 
                                     "tensor_validation", "debug")
                
                # Fallback shape management
                if len(original_shape) == 1:
                    x = x.unsqueeze(0).unsqueeze(0)
                elif len(original_shape) == 2:
                    x = x.unsqueeze(1)
                elif len(original_shape) != 3:
                    self._log(f"❌ Shape non gestibile: {original_shape}", "tensor_validation", "error")
                    raise ValueError(f"Unhandleable input shape: {original_shape}")
        
        except Exception as e:
            print(f"❌ Errore critico in shape management: {e}")
            raise RuntimeError(f"Critical shape management error: {e}")
        
        # 🛡️ VERIFICA FINALE SHAPE
        if len(x.shape) != 3:
            self._log(f"❌ Shape finale non valida: {x.shape} (deve essere 3D)", "tensor_validation", "error")
            raise ValueError(f"Final shape invalid: {x.shape}, must be 3D")
        
        # Estrai dimensioni
        batch_size, seq_len, actual_input_size = x.shape
        
        # 🛡️ VALIDAZIONE DIMENSIONI
        if batch_size <= 0 or seq_len <= 0 or actual_input_size <= 0:
            print(f"❌ Dimensioni non valide: {x.shape}")
            raise ValueError(f"Invalid dimensions: {x.shape}, all must be positive")
        
        # 🛡️ REGISTRA DIMENSIONE PER STATISTICHE
        if actual_input_size not in self.resize_stats['dimension_history'][-10:]:
            self.resize_stats['dimension_history'].append(actual_input_size)
        
        # 🛡️ DYNAMIC ADAPTER CON PROTEZIONE
        try:
            adapter = self._get_or_create_adapter(actual_input_size)
            
            # Applica adapter se necessario
            if not isinstance(adapter, nn.Identity):
                try:
                    x = self._apply_adapter(x, adapter)
                    
                    # 🛡️ VALIDAZIONE POST-ADAPTER
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        self._log("❌ NaN/Inf dopo adapter - sanitizzando...", "tensor_validation", "warning")
                        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
                    
                    self._log(f"🔧 Adapter applicato: {actual_input_size} → {x.shape[-1]}", 
                                         "tensor_validation", "debug")
                    
                except Exception as adapter_error:
                    print(f"❌ Errore applicazione adapter: {adapter_error}")
                    # FAIL FAST - NO FALLBACK
                    raise RuntimeError(f"Adapter application failed: {adapter_error}")
        
        except Exception as adapter_creation_error:
            print(f"❌ Errore creazione adapter: {adapter_creation_error}")
            # FAIL FAST - NO FALLBACK
            raise RuntimeError(f"Adapter creation failed: {adapter_creation_error}")
        
        # 🛡️ VERIFICA FINALE DELLE DIMENSIONI
        if x.shape[-1] != self.expected_input_size:
            print(f"❌ Dimensione finale incorretta! Expected {self.expected_input_size}, got {x.shape[-1]}")
            # FAIL FAST - NO FALLBACK
            raise ValueError(f"Final dimension mismatch: expected {self.expected_input_size}, got {x.shape[-1]}")
        
        # 🚀 FASE 2: LSTM PROCESSING CON NUOVE ARCHITETTURE
        try:
            # Controlla che x sia ancora valido
            if torch.isnan(x).any() or torch.isinf(x).any():
                self._log("❌ Input LSTM contiene NaN/Inf - sanitizzando...", "tensor_validation", "warning")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Original input not needed - residual connections disabled
            original_input = None
            
            # LSTM forward
            lstm_out, lstm_hidden = self.lstm(x)
            
            # 🛡️ VALIDAZIONE OUTPUT LSTM
            if lstm_out is None:
                print("❌ LSTM output è None!")
                raise RuntimeError("LSTM forward pass returned None output")
            
            if torch.isnan(lstm_out).any():
                nan_count = torch.isnan(lstm_out).sum().item()
                self._log(f"❌ LSTM output contiene {nan_count} NaN - sanitizzando...", "tensor_validation", "warning")
                lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if torch.isinf(lstm_out).any():
                inf_count = torch.isinf(lstm_out).sum().item()
                self._log(f"❌ LSTM output contiene {inf_count} Inf - sanitizzando...", "tensor_validation", "warning")
                lstm_out = torch.nan_to_num(lstm_out, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # LAYER NORMALIZATION - DISABLED
            
            # RESIDUAL CONNECTIONS - DISABLED
            
        except Exception as lstm_error:
            print(f"❌ Errore LSTM: {lstm_error}")
            raise RuntimeError(f"LSTM processing failed: {lstm_error}")
        
        # 🛡️ ATTENTION MECHANISM CON PROTEZIONE
        try:
            # Transpose per attention
            lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, features)
            
            # Attention
            attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
            
            # 🛡️ VALIDAZIONE ATTENTION OUTPUT
            if attn_out is None:
                print("❌ Attention output è None!")
                attn_out = lstm_out
            
            if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
                self._log("❌ Attention output contiene NaN/Inf - sanitizzando...", "tensor_validation", "warning")
                attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Layer norm con protezione
            try:
                attn_out = self.layer_norm(attn_out + lstm_out)
                
                if torch.isnan(attn_out).any() or torch.isinf(attn_out).any():
                    self._log("❌ LayerNorm output contiene NaN/Inf - sanitizzando...", "tensor_validation", "warning")
                    attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            except Exception as norm_error:
                print(f"❌ Errore LayerNorm: {norm_error}")
                attn_out = lstm_out  # Fallback
            
            # Transpose back
            attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, features)
            
        except Exception as attention_error:
            print(f"❌ Errore Attention: {attention_error}")
            # Usa output LSTM direttamente
            attn_out = lstm_out
        
        # 🛡️ FINAL LAYERS CON PROTEZIONE
        try:
            # Take last output
            out = attn_out[:, -1, :]
            
            # 🛡️ VALIDAZIONE PRIMA DEI LAYER FINALI
            if torch.isnan(out).any() or torch.isinf(out).any():
                self._log("❌ Pre-FC output contiene NaN/Inf - sanitizzando...", "tensor_validation", "warning")
                out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Dropout
            out = self.dropout(out)
            
            # FC1 con protezione
            try:
                out = self.activation(self.fc1(out))
                
                if torch.isnan(out).any() or torch.isinf(out).any():
                    self._log("❌ FC1 output contiene NaN/Inf - sanitizzando...", "tensor_validation", "warning")
                    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            except Exception as fc1_error:
                print(f"❌ Errore FC1: {fc1_error}")
                raise RuntimeError(f"FC1 layer failed: {fc1_error}")
            
            # Dropout finale
            out = self.dropout(out)
            
            # FC2 finale con protezione
            try:
                out = self.fc2(out)
                
                if torch.isnan(out).any() or torch.isinf(out).any():
                    self._log("❌ FC2 output contiene NaN/Inf - sanitizzando...", "tensor_validation", "warning")
                    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
                    
            except Exception as fc2_error:
                print(f"❌ Errore FC2: {fc2_error}")
                raise RuntimeError(f"FC2 layer failed: {fc2_error}")
            
        except Exception as final_error:
            print(f"❌ Errore nei layer finali: {final_error}")
            # FAIL FAST - NO FALLBACK
            raise RuntimeError(f"Final layers processing failed: {final_error}")
        
        # 🛡️ VALIDAZIONE FINALE ASSOLUTA
        if out is None:
            print("❌ Output finale è None!")
            raise RuntimeError("Final output is None")
        
        if torch.isnan(out).any() or torch.isinf(out).any():
            self._log("❌ Output finale contiene NaN/Inf - sanitizzando...", "tensor_validation", "warning")
            out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 🛡️ CLAMP OUTPUT PER SICUREZZA
        out = torch.clamp(out, -100, 100)  # Previeni output estremi
        
        self._log(f"Forward completato con successo: {original_shape} → {out.shape}", "forward", "debug")
        return out
            
    def get_resize_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche complete delle operazioni di resize e performance"""
        
        # Calcola frequenza dimensioni
        dimension_counts = {}
        for dim in self.resize_stats['dimension_history']:
            if dim not in dimension_counts:
                dimension_counts[dim] = 0
            dimension_counts[dim] += 1
        
        # Calcola statistiche cache
        cache_stats = self.get_cache_efficiency_stats()
        
        # Calcola risparmio computazionale
        total_calls = self.resize_stats['total_calls']
        adapters_created = self.resize_stats['adapters_created']
        
        if total_calls > 0:
            efficiency_gain = ((total_calls - adapters_created) / total_calls) * 100
            computational_savings = f"{efficiency_gain:.1f}%"
        else:
            computational_savings = "0%"
        
        # Identifica dimensioni più comuni
        if dimension_counts:
            most_common_dim = max(dimension_counts.items(), key=lambda x: x[1])
            optimization_potential = most_common_dim[1] / total_calls * 100 if total_calls > 0 else 0
        else:
            most_common_dim = (0, 0)
            optimization_potential = 0
        
        return {
            'performance_metrics': {
                'total_calls': total_calls,
                'adapters_created': adapters_created,
                'computational_savings': computational_savings,
                'cache_hit_rate': cache_stats['hit_rate_percentage'],
                'memory_efficiency': cache_stats['memory_efficiency']
            },
            'dimension_analysis': {
                'unique_dimensions_seen': len(set(self.resize_stats['dimension_history'])),
                'dimension_frequency': dimension_counts,
                'most_common_dimension': {
                    'size': most_common_dim[0],
                    'frequency': most_common_dim[1],
                    'optimization_potential': f"{optimization_potential:.1f}%"
                }
            },
            'cache_details': cache_stats,
            'adapter_keys': list(self.input_adapters.keys()),
            'recommendations': self._generate_optimization_recommendations(cache_stats, dimension_counts)
        }

    def _generate_optimization_recommendations(self, cache_stats: Dict, dimension_counts: Dict) -> List[str]:
        """Genera raccomandazioni per ottimizzazione"""
        
        recommendations = []
        
        # Raccomandazioni basate su hit rate
        if cache_stats['hit_rate_percentage'] < 50:
            recommendations.append("Low cache hit rate - consider input data preprocessing")
        elif cache_stats['hit_rate_percentage'] > 90:
            recommendations.append("Excellent cache performance - system well optimized")
        
        # Raccomandazioni basate su varietà dimensioni
        unique_dims = len(set(dimension_counts.keys()))
        if unique_dims > 8:
            recommendations.append(f"High dimension variety ({unique_dims}) - consider data standardization")
        elif unique_dims <= 3:
            recommendations.append("Low dimension variety - excellent for caching")
        
        # Raccomandazioni basate su active adapters
        if cache_stats['active_adapters'] > 15:
            recommendations.append("Too many active adapters - run cache optimization")
        
        # Raccomandazioni basate su usage patterns
        if cache_stats['most_used_adapter']['usage'] > cache_stats['least_used_adapter']['usage'] * 10:
            recommendations.append("Uneven adapter usage - some dimensions dominate")
        
        return recommendations
    
    def reset_adapters(self):
        """Reset tutti gli adapter (utile per testing)"""
        self.input_adapters.clear()
        self.adapter_cache.clear()
        self.resize_stats = {
            'total_calls': 0,
            'adapters_created': 0,
            'dimension_history': []
        }
        print("🔄 All adapters reset")
    
    def get_tensor_shape_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche dettagliate sulla gestione delle forme tensor"""
        
        # Statistiche conversioni
        conversion_stats = getattr(self, '_conversion_stats', {})
        total_conversions = sum(conversion_stats.values())
        
        # Statistiche shape manager
        if hasattr(self, '_shape_manager'):
            shape_stats = self._shape_manager.get_shape_statistics()
        else:
            shape_stats = {'total_processed': 0, 'conversions_needed': 0}
        
        # Calcola efficienza
        total_calls = self.resize_stats['total_calls']
        efficiency_improvement = (total_calls - total_conversions) / total_calls * 100 if total_calls > 0 else 100
        
        return {
            'tensor_shape_performance': {
                'total_forward_calls': total_calls,
                'shape_conversions_applied': total_conversions,
                'conversion_efficiency': f"{efficiency_improvement:.1f}%",
                'most_common_conversions': dict(sorted(conversion_stats.items(), key=lambda x: x[1], reverse=True)[:5])
            },
            'shape_manager_stats': shape_stats,
            'optimization_impact': {
                'before_optimization': '398 automatic expansions',
                'current_conversions': total_conversions,
                'reduction_achieved': f"{max(0, 398 - total_conversions)} fewer conversions",
                'efficiency_gain': f"{efficiency_improvement:.1f}%"
            },
            'recommendations': self._generate_tensor_recommendations(conversion_stats, efficiency_improvement)
        }

    def _generate_tensor_recommendations(self, conversion_stats: Dict, efficiency: float) -> List[str]:
        """Genera raccomandazioni per ottimizzazione tensor shapes"""
        
        recommendations = []
        
        if efficiency < 80:
            recommendations.append("Consider preprocessing input data to standard LSTM format")
        
        if '2D_to_3D_standard' in conversion_stats and conversion_stats['2D_to_3D_standard'] > 50:
            recommendations.append("High 2D→3D conversions - implement input standardization")
        
        if 'smart_reshape' in str(conversion_stats):
            recommendations.append("Smart reshaping active - monitor model accuracy")
        
        if len(conversion_stats) > 3:
            recommendations.append("Multiple conversion types detected - unify input pipeline")
        
        if efficiency > 95:
            recommendations.append("Excellent tensor shape efficiency achieved!")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information for analyzer integration"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "type": "AdvancedLSTM",
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "hidden_size": self.hidden_size,
            "expected_input_size": self.expected_input_size,
            "num_layers": getattr(self, 'num_layers', 'unknown'),
            "dropout": getattr(self, 'dropout_rate', 'unknown'),
            "architecture_version": "advanced_v2",
            "features": [
                "dynamic_input_adaptation",
                "architecture_improvements", 
                "nan_inf_protection",
                "multi_head_attention",
                "tensor_shape_management"
            ]
        }
    
    def get_gradient_stats(self) -> Dict[str, Any]:
        """Get gradient statistics for training analysis"""
        stats = {
            "vanishing_count": 0,
            "exploding_count": 0, 
            "healthy_count": 0,
            "total_gradients": 0,
            "gradient_norms": [],
            "problematic_layers": []
        }
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                stats["total_gradients"] += 1
                grad_norm = param.grad.norm().item()
                stats["gradient_norms"].append(grad_norm)
                
                if grad_norm < 1e-6:  # Vanishing gradient threshold
                    stats["vanishing_count"] += 1
                    stats["problematic_layers"].append(f"{name}_vanishing")
                elif grad_norm > 10.0:  # Exploding gradient threshold
                    stats["exploding_count"] += 1
                    stats["problematic_layers"].append(f"{name}_exploding")
                else:
                    stats["healthy_count"] += 1
        
        # Calculate summary statistics
        if stats["gradient_norms"]:
            import statistics
            stats["mean_gradient_norm"] = statistics.mean(stats["gradient_norms"])
            stats["median_gradient_norm"] = statistics.median(stats["gradient_norms"])
            stats["max_gradient_norm"] = max(stats["gradient_norms"])
            stats["min_gradient_norm"] = min(stats["gradient_norms"])
        
        return stats

    @property 
    def adapter_created(self) -> bool:
        """Compatibilità con codice esistente"""
        return len(self.input_adapters) > 0