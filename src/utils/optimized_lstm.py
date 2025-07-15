#!/usr/bin/env python3
"""
OptimizedLSTM - Neural Network Optimizations
===========================================

LSTM ottimizzato con layer normalization, skip connections, attention mechanism 
e tecniche anti-vanishing gradient per risolvere problemi di training ricorrenti.

Features:
- Layer Normalization per stabilità training
- Gradient Clipping intelligente
- Skip Connections per flusso gradiente
- Attention Mechanism per focus su feature rilevanti
- Highway Connections per accelerare convergenza
- Anti-vanishing gradient techniques

Author: ScalpingBOT Team
Version: 1.0.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import warnings

warnings.filterwarnings('ignore')


@dataclass
class LSTMConfig:
    """Configurazione per OptimizedLSTM"""
    
    # Architecture
    input_size: int = 64
    hidden_size: int = 128
    num_layers: int = 2
    output_size: int = 1
    
    # Regularization
    dropout_rate: float = 0.2
    weight_decay: float = 1e-4
    
    # Normalization
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    norm_eps: float = 1e-5
    
    # Skip connections
    use_skip_connections: bool = True
    skip_connection_interval: int = 2
    
    # Attention
    use_attention: bool = True
    attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Highway connections
    use_highway: bool = True
    highway_bias_init: float = -1.0
    
    # Gradient handling
    gradient_clip_value: float = 1.0
    gradient_clip_norm: float = 5.0
    use_gradient_checkpointing: bool = False
    
    # Initialization
    weight_init_method: str = 'xavier_uniform'  # 'xavier_uniform', 'kaiming_normal', 'orthogonal'
    bias_init_value: float = 0.0
    forget_gate_bias: float = 1.0  # LSTM forget gate bias (anti-vanishing)


class LayerNormLSTMCell(nn.Module):
    """LSTM Cell con Layer Normalization integrata"""
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, 
                 layer_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.layer_norm = layer_norm
        
        # Input transformations
        self.input_transform = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.hidden_transform = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
        
        # Layer normalization
        if layer_norm:
            self.ln_input = nn.LayerNorm(4 * hidden_size)
            self.ln_hidden = nn.LayerNorm(4 * hidden_size)
            self.ln_cell = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inizializzazione ottimizzata dei pesi"""
        # Xavier uniform per input e hidden transforms
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.hidden_transform.weight)
        
        if self.bias:
            # Bias forget gate a 1 per combattere vanishing gradients
            with torch.no_grad():
                forget_gate_bias_idx = slice(self.hidden_size, 2 * self.hidden_size)
                if hasattr(self.input_transform, 'bias') and self.input_transform.bias is not None:
                    self.input_transform.bias[forget_gate_bias_idx].fill_(1.0)
                if hasattr(self.hidden_transform, 'bias') and self.hidden_transform.bias is not None:
                    self.hidden_transform.bias[forget_gate_bias_idx].fill_(1.0)
    
    def forward(self, input_tensor: torch.Tensor, hidden_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass del LSTM cell con layer normalization
        
        Args:
            input_tensor: Input tensor [batch_size, input_size]
            hidden_state: (h_t-1, c_t-1) tuple
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (h_t, c_t)
        """
        h_prev, c_prev = hidden_state
        
        # Linear transformations
        input_proj = self.input_transform(input_tensor)
        hidden_proj = self.hidden_transform(h_prev)
        
        # Layer normalization
        if self.layer_norm:
            input_proj = self.ln_input(input_proj)
            hidden_proj = self.ln_hidden(hidden_proj)
        
        # Combined projections
        combined = input_proj + hidden_proj
        
        # Split into gates
        input_gate, forget_gate, cell_gate, output_gate = torch.chunk(combined, 4, dim=1)
        
        # Apply activations
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)
        
        # Update cell state
        c_new = forget_gate * c_prev + input_gate * cell_gate
        
        # Layer norm on cell state
        if self.layer_norm:
            c_new_norm = self.ln_cell(c_new)
        else:
            c_new_norm = c_new
        
        # Update hidden state
        h_new = output_gate * torch.tanh(c_new_norm)
        
        # Apply dropout
        if self.dropout is not None and self.training:
            h_new = self.dropout(h_new)
        
        return h_new, c_new


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism per LSTM features"""
    
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Linear projections
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inizializzazione pesi attention"""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass attention mechanism
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: Optional mask [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Attention output [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        query = self.query_proj(hidden_states)  # [batch, seq_len, hidden_size]
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Output projection
        output = self.output_proj(context)
        
        # Residual connection + layer norm
        output = self.layer_norm(output + hidden_states)
        
        return output


class HighwayNetwork(nn.Module):
    """Highway Network per accelerare l'apprendimento"""
    
    def __init__(self, size: int, bias_init: float = -1.0):
        super().__init__()
        
        self.transform_gate = nn.Linear(size, size)
        self.carry_gate = nn.Linear(size, size)
        self.transform_layer = nn.Linear(size, size)
        
        # Initialize carry gate bias to negative value (prefer carrying)
        nn.init.constant_(self.carry_gate.bias, bias_init)
        
        # Initialize other weights
        nn.init.xavier_uniform_(self.transform_gate.weight)
        nn.init.xavier_uniform_(self.transform_layer.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass highway network
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Highway output
        """
        # Transform gate (how much to transform)
        transform_gate = torch.sigmoid(self.transform_gate(x))
        
        # Carry gate (how much to carry through)
        carry_gate = torch.sigmoid(self.carry_gate(x))
        
        # Transform layer
        transformed = torch.relu(self.transform_layer(x))
        
        # Highway equation: T(x) * H(x) + C(x) * x
        # Where T + C = 1 (approximately)
        output = transform_gate * transformed + carry_gate * x
        
        return output


class OptimizedLSTM(nn.Module):
    """
    LSTM ottimizzato con tutte le tecniche anti-vanishing gradient
    
    Features:
    - Layer Normalization
    - Skip Connections
    - Multi-Head Attention
    - Highway Networks
    - Gradient Clipping intelligente
    - Inizializzazione ottimizzata
    """
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # LSTM layers with layer normalization
        self.lstm_layers = nn.ModuleList()
        
        for i in range(config.num_layers):
            input_size = config.input_size if i == 0 else config.hidden_size
            
            lstm_cell = LayerNormLSTMCell(
                input_size=input_size,
                hidden_size=config.hidden_size,
                layer_norm=config.use_layer_norm,
                dropout=config.dropout_rate if i < config.num_layers - 1 else 0.0
            )
            
            self.lstm_layers.append(lstm_cell)
        
        # Attention mechanism
        if config.use_attention:
            self.attention = MultiHeadAttention(
                hidden_size=config.hidden_size,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout
            )
        
        # Highway networks for skip connections
        if config.use_highway:
            self.highway_networks = nn.ModuleList([
                HighwayNetwork(config.hidden_size, config.highway_bias_init)
                for _ in range(config.num_layers)
            ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size // 2, config.output_size)
        )
        
        # Layer normalization for final output
        if config.use_layer_norm:
            self.output_norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize all weights
        self._initialize_weights()
        
        print(f"🧠 OptimizedLSTM initialized: {config.num_layers} layers, {config.hidden_size} hidden, attention={config.use_attention}")
    
    def _initialize_weights(self):
        """Inizializzazione ottimizzata di tutti i pesi"""
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                if self.config.weight_init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(param)
                elif self.config.weight_init_method == 'kaiming_normal':
                    nn.init.kaiming_normal_(param)
                elif self.config.weight_init_method == 'orthogonal':
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.xavier_uniform_(param)
            
            elif 'bias' in name:
                nn.init.constant_(param, self.config.bias_init_value)
        
        # Special initialization for output projection
        for module in self.output_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _apply_gradient_clipping(self):
        """Applica gradient clipping intelligente"""
        
        # Clip gradients by norm
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip_norm)
        
        # Clip gradients by value
        if self.config.gradient_clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.parameters(), self.config.gradient_clip_value)
    
    def _detect_vanishing_gradients(self) -> Dict[str, float]:
        """Detecta vanishing/exploding gradients"""
        
        gradient_stats = {}
        
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_stats[name] = grad_norm
                
                # Log problematic gradients
                if grad_norm < 1e-7:
                    self.logger.debug(f"Vanishing gradient detected in {name}: {grad_norm:.2e}")
                elif grad_norm > 10.0:
                    self.logger.debug(f"Exploding gradient detected in {name}: {grad_norm:.2e}")
        
        return gradient_stats
    
    def forward(self, x: torch.Tensor, hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass dell'OptimizedLSTM
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            hidden_states: Lista di hidden states per ogni layer
            
        Returns:
            Tuple[torch.Tensor, List]: (output, new_hidden_states)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = []
            for _ in range(self.config.num_layers):
                h_0 = torch.zeros(batch_size, self.config.hidden_size, device=x.device, dtype=x.dtype)
                c_0 = torch.zeros(batch_size, self.config.hidden_size, device=x.device, dtype=x.dtype)
                hidden_states.append((h_0, c_0))
        
        # Process through LSTM layers
        layer_outputs = []
        new_hidden_states = []
        current_input = x
        
        for layer_idx, (lstm_cell, hidden_state) in enumerate(zip(self.lstm_layers, hidden_states)):
            
            # Process sequence through this layer
            layer_hidden_outputs = []
            current_hidden = hidden_state
            
            for t in range(seq_len):
                # LSTM cell forward
                h_new, c_new = lstm_cell(current_input[:, t, :], current_hidden)
                current_hidden = (h_new, c_new)
                layer_hidden_outputs.append(h_new)
            
            # Stack outputs for this layer
            layer_output = torch.stack(layer_hidden_outputs, dim=1)  # [batch, seq_len, hidden]
            
            # Apply highway network if enabled
            if self.config.use_highway and hasattr(self, 'highway_networks'):
                layer_output = self.highway_networks[layer_idx](layer_output)
            
            # Skip connections every N layers
            if (self.config.use_skip_connections and 
                layer_idx > 0 and 
                layer_idx % self.config.skip_connection_interval == 0 and
                layer_output.shape == layer_outputs[layer_idx - self.config.skip_connection_interval].shape):
                
                layer_output = layer_output + layer_outputs[layer_idx - self.config.skip_connection_interval]
            
            layer_outputs.append(layer_output)
            new_hidden_states.append(current_hidden)
            current_input = layer_output
        
        # Final layer output
        final_output = layer_outputs[-1]
        
        # Apply attention mechanism
        if self.config.use_attention and hasattr(self, 'attention'):
            final_output = self.attention(final_output)
        
        # Apply output normalization
        if self.config.use_layer_norm and hasattr(self, 'output_norm'):
            final_output = self.output_norm(final_output)
        
        # Project to output size
        # Take only the last timestep for final prediction
        last_output = final_output[:, -1, :]  # [batch_size, hidden_size]
        prediction = self.output_projection(last_output)  # [batch_size, output_size]
        
        return prediction, new_hidden_states
    
    def get_gradient_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche sui gradienti"""
        
        gradient_stats = self._detect_vanishing_gradients()
        
        if gradient_stats:
            grad_norms = list(gradient_stats.values())
            stats = {
                'mean_grad_norm': np.mean(grad_norms),
                'max_grad_norm': np.max(grad_norms),
                'min_grad_norm': np.min(grad_norms),
                'std_grad_norm': np.std(grad_norms),
                'vanishing_count': sum(1 for g in grad_norms if g < 1e-7),
                'exploding_count': sum(1 for g in grad_norms if g > 10.0),
                'total_params': len(grad_norms)
            }
        else:
            stats = {
                'mean_grad_norm': 0.0,
                'max_grad_norm': 0.0,
                'min_grad_norm': 0.0,
                'std_grad_norm': 0.0,
                'vanishing_count': 0,
                'exploding_count': 0,
                'total_params': 0
            }
        
        return stats
    
    def prepare_optimizer(self, learning_rate: float = 1e-3) -> torch.optim.Optimizer:
        """Prepara optimizer ottimizzato per questo modello"""
        
        # Separate parameters for different learning rates
        lstm_params = []
        attention_params = []
        output_params = []
        
        for name, param in self.named_parameters():
            if 'lstm' in name.lower():
                lstm_params.append(param)
            elif 'attention' in name.lower():
                attention_params.append(param)
            else:
                output_params.append(param)
        
        # Parameter groups with different learning rates
        param_groups = [
            {'params': lstm_params, 'lr': learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': attention_params, 'lr': learning_rate * 0.5, 'weight_decay': self.config.weight_decay * 0.5},
            {'params': output_params, 'lr': learning_rate * 2.0, 'weight_decay': self.config.weight_decay}
        ]
        
        # Use AdamW optimizer with proper weight decay
        optimizer = torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999))
        
        return optimizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """Informazioni sul modello"""
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'num_layers': self.config.num_layers,
            'hidden_size': self.config.hidden_size,
            'features': {
                'layer_norm': self.config.use_layer_norm,
                'attention': self.config.use_attention,
                'highway': self.config.use_highway,
                'skip_connections': self.config.use_skip_connections
            }
        }


# Helper functions
def create_optimized_lstm_config(**kwargs) -> LSTMConfig:
    """Factory function per creare configurazione ottimizzata"""
    
    # Default optimized configuration
    default_config = {
        'hidden_size': 128,
        'num_layers': 3,
        'dropout_rate': 0.2,
        'use_layer_norm': True,
        'use_attention': True,
        'use_highway': True,
        'use_skip_connections': True,
        'gradient_clip_norm': 1.0,
        'weight_init_method': 'xavier_uniform'
    }
    
    # Override with provided kwargs
    default_config.update(kwargs)
    
    return LSTMConfig(**default_config)


def test_optimized_lstm():
    """Test delle funzionalità dell'OptimizedLSTM"""
    print("🧪 Testing OptimizedLSTM...")
    
    # Create test configuration
    config = create_optimized_lstm_config(
        input_size=10,
        hidden_size=64,
        num_layers=2,
        output_size=1
    )
    
    # Create model
    model = OptimizedLSTM(config)
    
    # Test forward pass
    batch_size, seq_len = 4, 20
    x = torch.randn(batch_size, seq_len, config.input_size)
    
    # Forward pass
    output, hidden_states = model(x)
    
    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Hidden states: {len(hidden_states)} layers")
    
    # Test model info
    model_info = model.get_model_info()
    print(f"✅ Model parameters: {model_info['total_parameters']:,}")
    print(f"✅ Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Test optimizer preparation
    optimizer = model.prepare_optimizer(learning_rate=1e-3)
    print(f"✅ Optimizer: {type(optimizer).__name__}")
    
    # Test gradient computation
    loss = output.sum()
    loss.backward()
    
    grad_stats = model.get_gradient_stats()
    print(f"✅ Gradient stats: mean={grad_stats['mean_grad_norm']:.6f}")
    
    # Apply gradient clipping
    model._apply_gradient_clipping()
    print("✅ Gradient clipping applied")
    
    return model


if __name__ == "__main__":
    test_optimized_lstm()