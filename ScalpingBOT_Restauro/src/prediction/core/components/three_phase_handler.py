#!/usr/bin/env python3
"""
Three Phase Handler - BIBBIA COMPLIANT
=====================================

Gestisce algoritmi con architettura 3-fasi (Training → Evaluation → Validation)
per il nuovo sistema modulare.

ARCHITETTURA 3-FASI:
1. TRAINING: Calcolo livelli/patterns dai dati storici
2. EVALUATION: Test su ticks futuri per calcolare confidence
3. VALIDATION: Applicazione real-time con livelli selezionati

ALGORITMI SUPPORTATI:
- PivotPoints_Classic: Training(30d) → Evaluation(100K ticks) → Validation(real-time)
- [Futuri algoritmi 3-fasi...]

BIBBIA RULES COMPLIANCE:
- ZERO FALLBACK: No default values, FAIL FAST validations
- NO TEST DATA: Only real market data  
- ONE ROAD: Single path for each phase
- CLEAN CODE: Constants, semantic naming
- FAIL FAST: Immediate validation, no silent failures

Author: ScalpingBOT Team
Version: 1.0.0 - NUOVO SISTEMA MODULARE
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import importlib
from pathlib import Path

# Import dell'algoritmo 3-fasi
from ScalpingBOT_Restauro.src.ml.algorithms.support_resistance.pivot_points_classic import PivotPointsClassic, create_pivot_points_classic


class ThreePhaseHandler:
    """
    Handler per algoritmi con architettura 3-fasi
    
    Coordina il flusso: Training → Evaluation → Validation
    per algoritmi che richiedono elaborazione multi-step.
    """
    
    # Algoritmi supportati con architettura 3-fasi
    THREE_PHASE_ALGORITHMS = {
        "PivotPoints_Classic"
    }
    
    # Fasi supportate
    SUPPORTED_PHASES = {
        "training", "evaluation", "validation_init", "validation_tick"
    }
    
    def __init__(self, data_path: str = "./analyzer_data"):
        """
        Initialize Three Phase Handler
        
        Args:
            data_path: Path base per salvataggio dati algoritmi
            
        Raises:
            ValueError: If data_path invalid - FAIL FAST
        """
        # FAIL FAST validation
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError(f"FAIL FAST: data_path must be non-empty string, got {data_path}")
            
        self.data_path = data_path
        
        # Registry delle istanze algoritmi attive
        self.algorithm_instances = {}
        
        # Phase tracking per asset/algoritmo
        self.phase_status = {}  # {asset: {algorithm: current_phase}}
        
        print(f"🔧 ThreePhaseHandler initialized - data_path: {data_path}")
    
    def is_three_phase_algorithm(self, algorithm_name: str) -> bool:
        """Check if algorithm uses 3-phase architecture"""
        if not isinstance(algorithm_name, str):
            raise TypeError(f"FAIL FAST: algorithm_name must be string, got {type(algorithm_name)}")
        return algorithm_name in self.THREE_PHASE_ALGORITHMS
    
    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported 3-phase algorithms"""
        return list(self.THREE_PHASE_ALGORITHMS)
    
    def create_algorithm_instance(self, algorithm_name: str, asset: str) -> Any:
        """
        Create instance of 3-phase algorithm
        
        Args:
            algorithm_name: Nome algoritmo da instanziare
            asset: Asset per cui creare l'istanza
            
        Returns:
            Algorithm instance
            
        Raises:
            ValueError: If algorithm not supported - FAIL FAST
        """
        # FAIL FAST validations
        if not isinstance(algorithm_name, str) or not algorithm_name.strip():
            raise ValueError(f"FAIL FAST: algorithm_name must be non-empty string, got {algorithm_name}")
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"FAIL FAST: asset must be non-empty string, got {asset}")
        
        if algorithm_name not in self.THREE_PHASE_ALGORITHMS:
            raise ValueError(f"FAIL FAST: Unsupported 3-phase algorithm: {algorithm_name}")
        
        # Create instance key
        instance_key = f"{asset}_{algorithm_name}"
        
        # Create algorithm instance based on name
        if algorithm_name == "PivotPoints_Classic":
            # BIBBIA COMPLIANT: Pass base data_path, PivotPoints will add asset internally
            instance = create_pivot_points_classic(data_path=self.data_path)
        else:
            # Future algorithms will be added here
            raise ValueError(f"FAIL FAST: Algorithm factory not implemented for {algorithm_name}")
        
        # Store instance
        self.algorithm_instances[instance_key] = instance
        
        # Initialize phase tracking
        if asset not in self.phase_status:
            self.phase_status[asset] = {}
        self.phase_status[asset][algorithm_name] = "created"
        
        print(f"✅ Created {algorithm_name} instance for {asset}")
        return instance
    
    def get_algorithm_instance(self, algorithm_name: str, asset: str) -> Optional[Any]:
        """Get existing algorithm instance"""
        # FAIL FAST validations
        if not isinstance(algorithm_name, str) or not algorithm_name.strip():
            raise ValueError(f"FAIL FAST: algorithm_name must be non-empty string")
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"FAIL FAST: asset must be non-empty string")
        
        instance_key = f"{asset}_{algorithm_name}"
        return self.algorithm_instances.get(instance_key)
    
    def run_training_phase(self, algorithm_name: str, asset: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute TRAINING phase for 3-phase algorithm
        
        Args:
            algorithm_name: Nome algoritmo
            asset: Asset da processare
            market_data: Dati di mercato per training
            
        Returns:
            Training results
            
        Raises:
            ValueError: If algorithm not supported or data invalid - FAIL FAST
        """
        # FAIL FAST validations
        if not self.is_three_phase_algorithm(algorithm_name):
            raise ValueError(f"FAIL FAST: {algorithm_name} is not a 3-phase algorithm")
        if not isinstance(market_data, dict):
            raise TypeError(f"FAIL FAST: market_data must be dict, got {type(market_data)}")
        
        # Get or create algorithm instance
        instance = self.get_algorithm_instance(algorithm_name, asset)
        if instance is None:
            instance = self.create_algorithm_instance(algorithm_name, asset)
        
        # Execute training phase
        print(f"🚀 Running TRAINING phase: {algorithm_name} for {asset}")
        
        try:
            results = instance.training_phase(market_data, asset)
            
            # Update phase status
            self.phase_status[asset][algorithm_name] = "training_completed"
            
            print(f"✅ TRAINING phase completed: {algorithm_name} for {asset}")
            return results
            
        except Exception as e:
            # FAIL FAST - re-raise training errors
            raise RuntimeError(f"FAIL FAST: Training phase failed for {algorithm_name}/{asset}: {e}")
    
    def run_evaluation_phase(self, algorithm_name: str, asset: str, future_ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute EVALUATION phase for 3-phase algorithm
        
        Args:
            algorithm_name: Nome algoritmo
            asset: Asset da processare  
            future_ticks: Ticks futuri per evaluation
            
        Returns:
            Evaluation results
            
        Raises:
            ValueError: If algorithm not ready or data invalid - FAIL FAST
        """
        # FAIL FAST validations
        if not self.is_three_phase_algorithm(algorithm_name):
            raise ValueError(f"FAIL FAST: {algorithm_name} is not a 3-phase algorithm")
        if not isinstance(future_ticks, list):
            raise TypeError(f"FAIL FAST: future_ticks must be list, got {type(future_ticks)}")
        
        # Check if training completed
        current_phase = self.phase_status.get(asset, {}).get(algorithm_name, "none")
        if current_phase != "training_completed":
            raise RuntimeError(f"FAIL FAST: Training must be completed before evaluation. Current phase: {current_phase}")
        
        # Get algorithm instance
        instance = self.get_algorithm_instance(algorithm_name, asset)
        if instance is None:
            raise RuntimeError(f"FAIL FAST: No algorithm instance found for {algorithm_name}/{asset}")
        
        # Execute evaluation phase
        print(f"📊 Running EVALUATION phase: {algorithm_name} for {asset}")
        
        try:
            results = instance.evaluation_phase(future_ticks, asset)
            
            # Update phase status
            self.phase_status[asset][algorithm_name] = "evaluation_completed"
            
            print(f"✅ EVALUATION phase completed: {algorithm_name} for {asset}")
            return results
            
        except Exception as e:
            # FAIL FAST - re-raise evaluation errors
            raise RuntimeError(f"FAIL FAST: Evaluation phase failed for {algorithm_name}/{asset}: {e}")
    
    def run_validation_init(self, algorithm_name: str, asset: str) -> Dict[str, Any]:
        """
        Execute VALIDATION INIT phase for 3-phase algorithm
        
        Args:
            algorithm_name: Nome algoritmo
            asset: Asset da processare
            
        Returns:
            Validation init results
            
        Raises:
            ValueError: If algorithm not ready - FAIL FAST
        """
        # FAIL FAST validations
        if not self.is_three_phase_algorithm(algorithm_name):
            raise ValueError(f"FAIL FAST: {algorithm_name} is not a 3-phase algorithm")
        
        # Check if evaluation completed
        current_phase = self.phase_status.get(asset, {}).get(algorithm_name, "none")
        if current_phase != "evaluation_completed":
            raise RuntimeError(f"FAIL FAST: Evaluation must be completed before validation. Current phase: {current_phase}")
        
        # Get algorithm instance
        instance = self.get_algorithm_instance(algorithm_name, asset)
        if instance is None:
            raise RuntimeError(f"FAIL FAST: No algorithm instance found for {algorithm_name}/{asset}")
        
        # Execute validation init
        print(f"🔮 Running VALIDATION INIT: {algorithm_name} for {asset}")
        
        try:
            results = instance.validation_phase_init(asset)
            
            # Update phase status
            self.phase_status[asset][algorithm_name] = "validation_ready"
            
            print(f"✅ VALIDATION INIT completed: {algorithm_name} for {asset}")
            return results
            
        except Exception as e:
            # FAIL FAST - re-raise validation errors
            raise RuntimeError(f"FAIL FAST: Validation init failed for {algorithm_name}/{asset}: {e}")
    
    def run_validation_tick_test(self, algorithm_name: str, asset: str, current_tick: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute VALIDATION TICK TEST for 3-phase algorithm
        
        Args:
            algorithm_name: Nome algoritmo
            asset: Asset da processare
            current_tick: Tick corrente per test
            
        Returns:
            Validation tick results
            
        Raises:
            ValueError: If algorithm not ready - FAIL FAST
        """
        # FAIL FAST validations
        if not self.is_three_phase_algorithm(algorithm_name):
            raise ValueError(f"FAIL FAST: {algorithm_name} is not a 3-phase algorithm")
        if not isinstance(current_tick, dict):
            raise TypeError(f"FAIL FAST: current_tick must be dict, got {type(current_tick)}")
        
        # Check if validation ready
        current_phase = self.phase_status.get(asset, {}).get(algorithm_name, "none")
        if current_phase != "validation_ready":
            raise RuntimeError(f"FAIL FAST: Validation init must be completed before tick testing. Current phase: {current_phase}")
        
        # Get algorithm instance
        instance = self.get_algorithm_instance(algorithm_name, asset)
        if instance is None:
            raise RuntimeError(f"FAIL FAST: No algorithm instance found for {algorithm_name}/{asset}")
        
        # Execute validation tick test
        try:
            return instance.validation_tick_test(current_tick)
        except Exception as e:
            # FAIL FAST - re-raise validation errors
            raise RuntimeError(f"FAIL FAST: Validation tick test failed for {algorithm_name}/{asset}: {e}")
    
    def get_phase_status(self, asset: Optional[str] = None) -> Dict[str, Any]:
        """Get current phase status for assets and algorithms"""
        if asset is None:
            return self.phase_status.copy()
        return self.phase_status.get(asset, {})
    
    def reset_algorithm(self, algorithm_name: str, asset: str) -> None:
        """Reset specific algorithm instance"""
        # FAIL FAST validations
        if not isinstance(algorithm_name, str) or not algorithm_name.strip():
            raise ValueError(f"FAIL FAST: algorithm_name must be non-empty string")
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError(f"FAIL FAST: asset must be non-empty string")
        
        instance_key = f"{asset}_{algorithm_name}"
        
        # Remove instance
        if instance_key in self.algorithm_instances:
            del self.algorithm_instances[instance_key]
        
        # Reset phase status
        if asset in self.phase_status and algorithm_name in self.phase_status[asset]:
            del self.phase_status[asset][algorithm_name]
        
        print(f"🔄 Reset {algorithm_name} for {asset}")
    
    def get_algorithm_stats(self, algorithm_name: str, asset: str) -> Dict[str, Any]:
        """Get statistics from algorithm instance"""
        instance = self.get_algorithm_instance(algorithm_name, asset)
        if instance is None:
            raise RuntimeError(f"FAIL FAST: No algorithm instance found for {algorithm_name}/{asset}")
        
        if hasattr(instance, 'get_algorithm_stats'):
            return instance.get_algorithm_stats()
        else:
            return {"error": "Algorithm does not support stats"}


# Factory function
def create_three_phase_handler(data_path: str = "./analyzer_data") -> ThreePhaseHandler:
    """Factory function per creare ThreePhaseHandler - FAIL FAST se parametri invalidi"""
    return ThreePhaseHandler(data_path)


# Export
__all__ = [
    'ThreePhaseHandler',
    'create_three_phase_handler'
]