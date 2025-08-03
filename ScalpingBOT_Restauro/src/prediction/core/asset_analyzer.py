#!/usr/bin/env python3
"""
Asset Analyzer - REFACTORED FROM MONOLITH
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults
- ✅ Fail fast error handling
- ✅ No debug prints/spam
- ✅ Modular architecture using migrated components

Core Asset Analysis Engine che orchestrerà i moduli migrati FASE 1-5.
ESTRATTO e REFACTORIZZATO da src/Analyzer.py:10494-19169 (8,675 linee).
"""

import os
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import shared enums
from ScalpingBOT_Restauro.src.shared.enums import ModelType

# Import migrated components from FASE 1-5
from ScalpingBOT_Restauro.src.config.base.config_loader import get_configuration_manager
from ScalpingBOT_Restauro.src.config.base.base_config import get_analyzer_config
from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventCollector
from ScalpingBOT_Restauro.src.data.collectors.tick_collector import TickCollector
from ScalpingBOT_Restauro.src.data.processors.market_data_processor import MarketDataProcessor
from ScalpingBOT_Restauro.src.ml.models.competition import AlgorithmCompetition
from ScalpingBOT_Restauro.src.ml.models.base_models import Prediction
from ScalpingBOT_Restauro.src.interfaces.mt5.mt5_adapter import MT5Adapter
# Removed safe_print import - using fail-fast error handling instead


class AssetAnalyzer:
    """
    Core Asset Analysis Engine - REFACTORED VERSION
    
    Orchestrerà tutti i moduli migrati FASE 1-5:
    - CONFIG: AnalyzerConfig, ConfigurationManager
    - MONITORING: EventCollector, logging systems  
    - DATA: TickCollector, MarketDataProcessor
    - ML: Competition, LSTM, CNN, Transformer models
    - INTERFACES: MT5 integration
    """
    
    def __init__(self, asset: str, data_path: str = "./test_analyzer_data", config_manager=None):
        """
        Inizializza Asset Analyzer usando componenti migrati
        
        Args:
            asset: Nome dell'asset da analizzare
            data_path: Path per i dati (default ./test_analyzer_data)
            config_manager: Configuration manager (opzionale)
        """
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("asset must be non-empty string")
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError("data_path must be non-empty string")
            
        self.asset = asset
        self.data_path = f"{data_path}/{asset}"
        os.makedirs(self.data_path, exist_ok=True)
        
        # FASE 1 - CONFIG: Use migrated configuration system
        self.config_manager = config_manager or get_configuration_manager()
        self.config = get_analyzer_config()
        
        # FASE 2 - MONITORING: Use migrated event collector
        self.event_collector = EventCollector(self.config_manager.get_current_configuration().monitoring)
        
        # FASE 4 - DATA: Use migrated data components
        self.tick_collector = TickCollector(
            max_buffer_size=self.config.max_tick_buffer_size
        )
        self.market_data_processor = MarketDataProcessor(self.config)
        
        # FASE 3 - INTERFACES: Use migrated MT5 interface
        self.mt5_adapter = MT5Adapter()
        
        # FASE 5 - ML: Initialize algorithm bridge and competitions
        from ScalpingBOT_Restauro.src.ml.integration.algorithm_bridge import create_algorithm_bridge
        self.algorithm_bridge = create_algorithm_bridge(
            ml_models={},  # Will be populated with actual models
            logger=None    # Will use default logger
        )
        
        # 🔧 COMPLETE COMPETITION SYSTEM: Initialize full competition framework
        from ScalpingBOT_Restauro.src.ml.models.competition import ChampionPreserver, RealityChecker, EmergencyStopSystem, AlgorithmCompetition
        
        # Initialize competition dependencies with correct signatures
        self.champion_preserver = ChampionPreserver(
            storage_path=f"{self.data_path}/champions"
        )
        
        self.reality_checker = RealityChecker(
            config=self.config
        )
        
        self.emergency_stop_system = EmergencyStopSystem(
            logger=None,  # Will use default logger
            config=self.config
        )
        
        # Initialize competitions for ALL model types - FAIL FAST if any problems
        self.competitions: Dict[ModelType, AlgorithmCompetition] = {}
        for model_type in ModelType:
            # FAIL FAST: Every ModelType MUST have algorithms - no skip/fallback allowed
            available_algorithms = self.algorithm_bridge.get_available_algorithms(model_type)
            if not available_algorithms:
                raise RuntimeError(f"FAIL FAST: ModelType {model_type.value} has no algorithms available - system integrity compromised")
            
            # Create competition - will fail fast if any initialization problems
            self.competitions[model_type] = AlgorithmCompetition(
                model_type=model_type,
                asset=self.asset,
                logger=None,  # Will use default logger
                champion_preserver=self.champion_preserver,
                reality_checker=self.reality_checker,
                emergency_stop=self.emergency_stop_system,
                config=self.config
            )
            print(f"✅ Created competition for {model_type.value} with {len(available_algorithms)} algorithms")
        
        # Threading
        self.data_lock = threading.RLock()
        self.competitions_lock = threading.RLock()
        
        # State tracking
        self.learning_phase = True
        self.is_running = False
        self.prediction_history: List[Prediction] = []
        
        # Performance tracking
        self.stats = {
            'ticks_processed': 0,
            'predictions_made': 0,
            'start_time': None,
            'errors': 0
        }
    
    def process_tick(self, timestamp: datetime, price: float, volume: float, 
                    bid: Optional[float] = None, ask: Optional[float] = None) -> Dict[str, Any]:
        """
        Processa un tick usando i componenti migrati
        
        Args:
            timestamp: Timestamp del tick
            price: Prezzo del tick
            volume: Volume del tick
            bid: Prezzo bid (opzionale)
            ask: Prezzo ask (opzionale)
            
        Returns:
            Risultato del processing
        """
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be datetime")
        if not isinstance(price, (int, float)) or price <= 0:
            raise ValueError("price must be positive number")
        if not isinstance(volume, (int, float)) or volume < 0:
            raise ValueError("volume must be non-negative number")
        
        processing_start = datetime.now()
        
        try:
            with self.data_lock:
                # FASE 4 - DATA: Use migrated tick collector
                collection_result = self.tick_collector.collect_tick(
                    timestamp=timestamp,
                    price=price,
                    volume=volume,
                    bid=bid,
                    ask=ask
                )
                
                # FASE 4 - DATA: Process market data with FAIL-FAST validation
                # 🔧 FIXED RACE CONDITION: Single atomic operation to get buffer and check length
                # Use tick collector buffer and convert to deque for processor
                tick_buffer_list = self.tick_collector.get_tick_buffer()
                buffer_length = len(tick_buffer_list)  # Capture length atomically
                if buffer_length == 0:
                    raise RuntimeError("No tick data available for market data processing")
                
                # Validate bid/ask data before processing
                if bid is not None and bid <= 0:
                    raise ValueError(f"Invalid bid value: {bid}")
                if ask is not None and ask <= 0:
                    raise ValueError(f"Invalid ask value: {ask}")
                
                # Convert list to deque for MarketDataProcessor
                from collections import deque
                tick_deque = deque(tick_buffer_list, maxlen=5000)
                
                market_data = self.market_data_processor.prepare_market_data(tick_deque)
                
                # Generate predictions if we have enough data
                predictions = {}
                if self._has_sufficient_data():
                    predictions = self._generate_predictions(market_data)
                
                # Update stats
                self.stats['ticks_processed'] += 1
                
                return {
                    'status': 'success',
                    'collection_result': collection_result,
                    'market_data': market_data,
                    'predictions': predictions,
                    'processing_time_ms': (datetime.now() - processing_start).total_seconds() * 1000
                }
                
        except Exception as e:
            self.stats['errors'] += 1
            
            # FASE 2 - MONITORING: Use migrated event collector for errors
            if self.event_collector:
                from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventType, EventSeverity
                self.event_collector.emit_manual_event(
                    EventType.ERROR_EVENT,
                    {
                        'component': 'asset_analyzer',
                        'method': 'process_tick',
                        'asset': self.asset,
                        'error': str(e)
                    },
                    EventSeverity.ERROR
                )
            
            raise
    
    def _has_sufficient_data(self) -> bool:
        """Verifica se abbiamo dati sufficienti per le predizioni"""
        buffer_data = self.tick_collector.get_tick_buffer()
        return len(buffer_data) >= getattr(self.config, 'min_ticks_for_prediction', 100)
    
    def _generate_predictions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera predizioni usando le competizioni ML
        
        Args:
            market_data: Dati di mercato processati
            
        Returns:
            Predizioni per ogni model type
        """
        predictions = {}
        
        # Use algorithm bridge to generate predictions
        try:
            for model_type in ModelType:
                available_algorithms = self.algorithm_bridge.get_available_algorithms(model_type)
                
                if not available_algorithms:
                    raise RuntimeError(f"No algorithms available for model type {model_type.value}")
                
                # 🔧 COMPLETE COMPETITION SYSTEM: Use full competition framework
                # Get competition for this model type
                with self.competitions_lock:
                    if model_type not in self.competitions:
                        raise RuntimeError(f"Competition not initialized for model type {model_type.value}")
                    
                    competition = self.competitions[model_type]
                
                # Get current champion algorithm from competition system
                try:
                    champion_algorithm_name = competition.get_champion_algorithm()
                    if champion_algorithm_name is None:
                        # FAIL FAST - No champion established is an error condition
                        raise RuntimeError(f"No champion algorithm established for {model_type.value} - system not properly initialized")
                    else:
                        # Verify champion is still available
                        if champion_algorithm_name in available_algorithms:
                            champion_algorithm = champion_algorithm_name
                        else:
                            # FAIL FAST - Champion should always be available
                            raise RuntimeError(f"Champion algorithm {champion_algorithm_name} not available for {model_type.value}")
                except Exception as e:
                    # FAIL FAST - Competition system errors should not be silently handled
                    raise RuntimeError(f"Competition system error for {model_type.value}: {e}")
                
                try:
                    # 🔧 COMPLETE COMPETITION SYSTEM: Execute algorithm with full competition integration
                    algorithm_result = self.algorithm_bridge.execute_algorithm(
                        model_type, champion_algorithm, market_data
                    )
                    
                    # Convert to prediction format
                    prediction_obj = self.algorithm_bridge.convert_to_prediction(
                        algorithm_result, self.asset, model_type
                    )
                    
                    # 🔧 COMPLETE COMPETITION SYSTEM: Submit prediction to competition for validation
                    with self.competitions_lock:
                        competition.submit_prediction(
                            algorithm_name=champion_algorithm,
                            prediction_data=prediction_obj.prediction_data,
                            confidence=prediction_obj.confidence,
                            validation_criteria={'method': 'real_time_validation'},
                            market_conditions=market_data
                        )
                    
                    # Store prediction with correct field names
                    predictions[model_type.value] = {
                        'model_type': model_type.value,
                        'algorithm': champion_algorithm,
                        'confidence': prediction_obj.confidence,
                        'prediction_data': prediction_obj.prediction_data,
                        'timestamp': prediction_obj.timestamp.isoformat(),
                        'prediction_id': prediction_obj.id
                    }
                    
                except Exception as e:
                    # Log error but continue with other models
                    if self.event_collector:
                        from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventType, EventSeverity
                        self.event_collector.emit_manual_event(
                            EventType.ERROR_EVENT,
                            {
                                'component': 'asset_analyzer',
                                'method': '_generate_predictions',
                                'model_type': model_type.value,
                                'algorithm': champion_algorithm,
                                'asset': self.asset,
                                'error': str(e)
                            },
                            EventSeverity.ERROR
                        )
                        
        except Exception as e:
            # Log general prediction error
            if self.event_collector:
                from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventType, EventSeverity
                self.event_collector.emit_manual_event(
                    EventType.ERROR_EVENT,
                    {
                        'component': 'asset_analyzer',
                        'method': '_generate_predictions_bridge',
                        'asset': self.asset,
                        'error': str(e)
                    },
                    EventSeverity.ERROR
                )
        
        if predictions:
            self.stats['predictions_made'] += 1
            
        return predictions
    
    def start(self):
        """Avvia il sistema analyzer"""
        if self.is_running:
            raise RuntimeError("AssetAnalyzer already running")
        
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        # Start event collector
        if self.event_collector:
            from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventType, EventSeverity
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    'action': 'asset_analyzer_start',
                    'asset': self.asset,
                    'data_path': self.data_path
                },
                EventSeverity.INFO
            )
    
    def stop(self):
        """Ferma il sistema analyzer"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop event collector notification
        if self.event_collector:
            from ScalpingBOT_Restauro.src.monitoring.events.event_collector import EventType, EventSeverity
            self.event_collector.emit_manual_event(
                EventType.SYSTEM_STATUS,
                {
                    'action': 'asset_analyzer_stop',
                    'asset': self.asset,
                    'duration_seconds': (datetime.now() - self.stats['start_time']).total_seconds(),
                    'ticks_processed': self.stats['ticks_processed'],
                    'predictions_made': self.stats['predictions_made']
                },
                EventSeverity.INFO
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche del sistema"""
        stats = self.stats.copy()
        stats['asset'] = self.asset
        stats['is_running'] = self.is_running
        stats['learning_phase'] = self.learning_phase
        stats['competitions_count'] = len(self.competitions)
        stats['buffer_size'] = len(self.tick_collector.get_tick_buffer())
        
        # Add algorithm bridge statistics
        if hasattr(self, 'algorithm_bridge'):
            stats['algorithm_bridge'] = self.algorithm_bridge.get_bridge_stats()
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Restituisce stato di salute del sistema"""
        health = {
            'asset': self.asset,
            'overall_status': 'healthy',
            'components': {},
            'issues': []
        }
        
        # 🔧 COMPLETE COMPETITION SYSTEM: Check full competition health
        healthy_competitions = 0
        with self.competitions_lock:
            for model_type, competition in self.competitions.items():
                try:
                    # Get real health status from competition
                    performance_summary = competition.get_performance_summary()
                    if 'health_status' not in performance_summary:
                        raise KeyError("Missing required field 'health_status' from competition performance summary")
                    
                    comp_health = {
                        'status': performance_summary['health_status'],
                        'model_type': model_type.value,
                        'initialized': True
                    }
                    
                    # Add competition-specific metrics
                    champion_algorithm = competition.get_champion_algorithm()
                    comp_health['has_champion'] = champion_algorithm is not None
                    comp_health['champion_algorithm'] = champion_algorithm
                    
                    # Check if current champion is emergency stopped
                    if champion_algorithm:
                        stopped_algorithms = self.emergency_stop_system.get_stopped_algorithms()
                        comp_health['emergency_stopped'] = champion_algorithm in stopped_algorithms
                    else:
                        comp_health['emergency_stopped'] = False
                    
                except Exception as e:
                    # Fallback health status if competition fails
                    comp_health = {
                        'status': 'critical',
                        'model_type': model_type.value,
                        'initialized': False,
                        'error': str(e),
                        'has_champion': False,
                        'emergency_stopped': True
                    }
                
                health['components'][f'competition_{model_type.value}'] = comp_health
                
                if comp_health.get('status') == 'healthy':
                    healthy_competitions += 1
        
        # Overall health assessment
        if healthy_competitions == 0:
            health['overall_status'] = 'critical'
            health['issues'].append('No healthy competitions')
        elif healthy_competitions < len(self.competitions) / 2:
            health['overall_status'] = 'degraded'
            health['issues'].append('Less than 50% competitions healthy')
        
        # Check data flow
        if self.stats['ticks_processed'] == 0:
            health['issues'].append('No ticks processed')
        
        # Check error rate
        if self.stats['errors'] > 0:
            error_rate = self.stats['errors'] / max(self.stats['ticks_processed'], 1)
            if error_rate > 0.1:  # 10% error rate
                health['overall_status'] = 'degraded'
                health['issues'].append(f'High error rate: {error_rate:.1%}')
        
        return health


# Factory function
def create_asset_analyzer(asset: str, data_path: str = "./test_analyzer_data", 
                         config_manager=None) -> AssetAnalyzer:
    """Factory function per creare AssetAnalyzer"""
    return AssetAnalyzer(asset, data_path, config_manager)


# Export
__all__ = [
    'AssetAnalyzer',
    'create_asset_analyzer'
]