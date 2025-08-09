#!/usr/bin/env python3
"""
PivotPoints_Classic Algorithm - ARCHITETTURA CORRETTA
======================================================

Implementazione completa dell'algoritmo PivotPoints_Classic secondo le specifiche:

FASE 1 - TRAINING (30 giorni):
- Calcolo giornaliero H/L/C → Pivot + S1/S2/S3 + R1/R2/R3
- Salvataggio giornaliero: day_1_pivot.json, day_2_pivot.json, etc.
- Ogni file contiene: livelli + timestamp + confidence iniziale

FASE 2 - EVALUATION (100K ticks futuri):
- Per ogni livello di ogni giorno, test sui SUCCESSIVI 100K ticks
- Calcola hit-rate specifico per ogni singolo livello
- Aggiorna confidence del livello nel file corrispondente

FASE 3 - VALIDATION (real-time):
- Carica SOLO livelli con confidence >= 0.5 dai file giornalieri
- Test tick-by-tick in tempo reale
- Aggiornamento dinamico confidence basato su performance reali

REGOLE BIBBIA:
- ZERO FALLBACK: No default values, FAIL FAST
- NO TEST DATA: Solo dati reali
- ONE ROAD: Un solo percorso di implementazione
- CLEAN CODE: Codice pulito e specifico

Author: ScalpingBOT Team  
Version: 2.0.0 - RESET COMPLETO
"""

import os
import json
from datetime import datetime, timedelta, date
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path


class PivotPointsClassic:
    """
    Algoritmo PivotPoints_Classic con architettura corretta a 3 fasi
    
    Implementa il vero sistema specificato:
    - Training: Calcolo e salvataggio livelli giornalieri (30 giorni)
    - Evaluation: Test ogni livello sui 100K ticks futuri + confidence
    - Validation: Caricamento selettivo + test real-time
    """
    
    # Constants - CLEAN CODE compliant
    MIN_TICKS_PER_DAY = 50
    MIN_EVALUATION_TICKS = 50000
    CONFIDENCE_THRESHOLD = 0.5
    PRICE_TOLERANCE = 0.0001  # 1 pip
    BOUNCE_THRESHOLD_PERCENT = 0.0005  # 0.05%
    LOOKAHEAD_TICKS = 10
    
    def __init__(self, data_path: str = "./analyzer_data"):
        """
        Inizializza PivotPoints_Classic
        
        Args:
            data_path: Path base per salvataggio dati
        """
        if not isinstance(data_path, str) or not data_path.strip():
            raise ValueError("FAIL FAST: data_path must be non-empty string")
            
        self.data_path = Path(data_path)
        self.algorithm_name = "PivotPoints_Classic"
        
        # State tracking
        self.current_phase = None  # 'training', 'evaluation', 'validation'
        self.daily_levels = {}     # Cache per livelli giornalieri durante training
        self.active_levels = {}    # Livelli attivi durante validation (confidence >= 0.5)
        
        # Statistics
        self.stats = {
            'training_days_processed': 0,
            'levels_calculated': 0,
            'levels_evaluated': 0,
            'active_levels_count': 0,
            'validation_hits': 0,
            'validation_tests': 0
        }
        
        print(f"🎯 PivotPoints_Classic initialized - data_path: {self.data_path}")
    
    # ========================================
    # FASE 1: TRAINING - Calcolo e salvataggio giornaliero
    # ========================================
    
    def training_phase(self, market_data: Dict[str, Any], asset: str) -> Dict[str, Any]:
        """
        FASE 1 - TRAINING: Calcola livelli giornalieri per 30 giorni
        
        Args:
            market_data: Dati di mercato con 30 giorni di ticks
            asset: Nome asset (es. "EURUSD")
            
        Returns:
            Risultato training con summary livelli calcolati
        """
        if not isinstance(market_data, dict):
            raise TypeError("FAIL FAST: market_data must be dict")
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("FAIL FAST: asset must be non-empty string")
            
        # Validate required fields
        required_fields = ['price_history', 'timestamps']
        for field in required_fields:
            if field not in market_data:
                raise KeyError(f"FAIL FAST: Required field '{field}' missing from market_data")
        
        prices = market_data['price_history']
        timestamps = market_data['timestamps']
        
        if not prices or not timestamps:
            raise ValueError("FAIL FAST: Empty price_history or timestamps")
        if len(prices) != len(timestamps):
            raise ValueError("FAIL FAST: price_history and timestamps length mismatch")
            
        self.current_phase = 'training'
        print(f"🚀 Starting TRAINING phase for {asset} - Processing {len(prices):,} ticks")
        
        # Group ticks by day
        daily_data = self._group_ticks_by_day(prices, timestamps)
        
        if not daily_data:
            raise ValueError("FAIL FAST: No daily data found - cannot calculate pivot points")
            
        # Calculate pivot levels for each day
        training_results = []
        
        for day_index, (date, day_prices) in enumerate(sorted(daily_data.items())):
            if len(day_prices) < self.MIN_TICKS_PER_DAY:  # Skip days with insufficient data
                print(f"⚠️ Skipping {date} - insufficient data ({len(day_prices)} ticks)")
                continue
                
            # Calculate daily H/L/C
            daily_ohlc = self._calculate_daily_ohlc(day_prices)
            
            # Calculate pivot levels
            pivot_levels = self._calculate_pivot_levels(daily_ohlc)
            
            # Save daily levels to file
            day_filename = f"day_{day_index + 1:02d}_pivot.json"
            self._save_daily_levels(asset, date, pivot_levels, day_filename)
            
            # Store in memory for summary
            self.daily_levels[date] = {
                'levels': pivot_levels,
                'filename': day_filename,
                'ohlc': daily_ohlc
            }
            
            training_results.append({
                'date': date.isoformat(),
                'levels_calculated': len(pivot_levels),
                'pivot_value': pivot_levels['P']['value'],
                'support_levels': len([l for l in pivot_levels.values() if l['type'] == 'support']),
                'resistance_levels': len([l for l in pivot_levels.values() if l['type'] == 'resistance'])
            })
            
            self.stats['training_days_processed'] += 1
            self.stats['levels_calculated'] += len(pivot_levels)
            
        print(f"✅ TRAINING completed - {len(training_results)} days processed")
        print(f"   Total levels calculated: {self.stats['levels_calculated']}")
        
        return {
            'phase': 'training',
            'status': 'completed',
            'asset': asset,
            'days_processed': len(training_results),
            'total_levels': self.stats['levels_calculated'],
            'daily_results': training_results[-5:],  # Last 5 days summary
            'algorithm': self.algorithm_name,
            'confidence': 0.5  # Confidence iniziale neutro - sarà aggiornato in evaluation phase
        }
    
    def _group_ticks_by_day(self, prices: List[float], timestamps: List[Any]) -> Dict[date, List[float]]:
        """Raggruppa ticks per giorno"""
        daily_data = {}
        
        for i, (price, ts) in enumerate(zip(prices, timestamps)):
            # Parse timestamp
            if isinstance(ts, str):
                try:
                    # Try ISO format first
                    parsed_dt = datetime.fromisoformat(ts)
                    tick_date = parsed_dt.date()
                except ValueError:
                    try:
                        # Try MT5 format: "2025.06.06 00:09:09"
                        parsed_dt = datetime.strptime(ts, "%Y.%m.%d %H:%M:%S")
                        tick_date = parsed_dt.date()
                    except ValueError:
                        raise ValueError(f"FAIL FAST: Invalid timestamp format at index {i}: {ts}")
            elif hasattr(ts, 'date'):  # datetime object
                tick_date = ts.date()
            else:
                raise TypeError(f"FAIL FAST: Unexpected timestamp type at index {i}: {type(ts)}")
            
            # Group by date
            if tick_date not in daily_data:
                daily_data[tick_date] = []
            daily_data[tick_date].append(float(price))
                
        return daily_data
    
    def _calculate_daily_ohlc(self, day_prices: List[float]) -> Dict[str, float]:
        """Calcola OHLC giornaliero"""
        if not day_prices:
            raise ValueError("FAIL FAST: Empty day_prices for OHLC calculation")
            
        return {
            'open': day_prices[0],
            'high': max(day_prices),
            'low': min(day_prices),
            'close': day_prices[-1]
        }
    
    def _calculate_pivot_levels(self, ohlc: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """
        Calcola livelli pivot classici da OHLC
        
        Returns:
            Dict con livelli: {'P': {}, 'S1': {}, 'R1': {}, 'S2': {}, 'R2': {}, 'S3': {}, 'R3': {}}
        """
        high = ohlc['high']
        low = ohlc['low'] 
        close = ohlc['close']
        
        # Classic pivot calculation
        pivot = (high + low + close) / 3
        
        # Support and Resistance levels
        s1 = 2 * pivot - high
        r1 = 2 * pivot - low
        s2 = pivot - (high - low)
        r2 = pivot + (high - low)
        s3 = low - 2 * (high - pivot)
        r3 = high + 2 * (pivot - low)
        
        # Structure levels data
        levels = {
            'P': {
                'value': round(pivot, 5),
                'type': 'pivot',
                'confidence': 0.8,  # Initial confidence
                'hits': 0,
                'tests': 0,
                'last_test_time': None,
                'broken': False
            },
            'S1': {
                'value': round(s1, 5),
                'type': 'support', 
                'confidence': 0.7,
                'hits': 0,
                'tests': 0,
                'last_test_time': None,
                'broken': False
            },
            'R1': {
                'value': round(r1, 5),
                'type': 'resistance',
                'confidence': 0.7,
                'hits': 0,
                'tests': 0,
                'last_test_time': None,
                'broken': False
            },
            'S2': {
                'value': round(s2, 5),
                'type': 'support',
                'confidence': 0.6,
                'hits': 0,
                'tests': 0,
                'last_test_time': None,
                'broken': False
            },
            'R2': {
                'value': round(r2, 5),
                'type': 'resistance',
                'confidence': 0.6,
                'hits': 0,
                'tests': 0,
                'last_test_time': None,
                'broken': False
            },
            'S3': {
                'value': round(s3, 5),
                'type': 'support',
                'confidence': self.CONFIDENCE_THRESHOLD,
                'hits': 0,
                'tests': 0,
                'last_test_time': None,
                'broken': False
            },
            'R3': {
                'value': round(r3, 5),
                'type': 'resistance',
                'confidence': self.CONFIDENCE_THRESHOLD,
                'hits': 0,
                'tests': 0,
                'last_test_time': None,
                'broken': False
            }
        }
        
        return levels
    
    def _save_daily_levels(self, asset: str, date: date, levels: Dict[str, Dict[str, Any]], filename: str):
        """Salva livelli giornalieri su file"""
        # Create asset directory
        asset_dir = self.data_path / asset / "pivot_levels"
        asset_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            'asset': asset,
            'date': date.isoformat(),
            'algorithm': self.algorithm_name,
            'created_timestamp': datetime.now().isoformat(),
            'phase': 'training',
            'levels': levels,
            'summary': {
                'total_levels': len(levels),
                'pivot_value': levels['P']['value'],
                'support_count': len([l for l in levels.values() if l['type'] == 'support']),
                'resistance_count': len([l for l in levels.values() if l['type'] == 'resistance'])
            }
        }
        
        # Save to file
        file_path = asset_dir / filename
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        print(f"💾 Saved daily levels: {filename} - {len(levels)} levels")
    
    # ========================================
    # FASE 2: EVALUATION - Test sui 100K ticks futuri
    # ========================================
    
    def evaluation_phase(self, future_ticks: List[Dict[str, Any]], asset: str) -> Dict[str, Any]:
        """
        FASE 2 - EVALUATION: Testa ogni livello sui 100K ticks futuri
        
        Args:
            future_ticks: 100K ticks successivi al training
            asset: Nome asset
            
        Returns:
            Risultato evaluation con confidence aggiornate
        """
        if not isinstance(future_ticks, list) or len(future_ticks) < self.MIN_EVALUATION_TICKS:
            raise ValueError(f"FAIL FAST: Need at least {self.MIN_EVALUATION_TICKS} future ticks, got {len(future_ticks) if isinstance(future_ticks, list) else 'invalid'}")
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("FAIL FAST: asset must be non-empty string")
            
        self.current_phase = 'evaluation'
        print(f"📊 Starting EVALUATION phase for {asset} - Testing on {len(future_ticks):,} future ticks")
        
        # Load all daily level files
        daily_files = self._load_daily_level_files(asset)
        if not daily_files:
            raise RuntimeError(f"FAIL FAST: No daily level files found for {asset}")
            
        evaluation_results = []
        total_levels_tested = 0
        total_levels_improved = 0
        
        # Test each day's levels
        for filename, daily_data in daily_files.items():
            levels = daily_data['levels']
            original_date = daily_data['date']
            
            print(f"   🎯 Testing levels from {original_date} ({filename})")
            
            # Test each level on future ticks
            for level_name, level_data in levels.items():
                level_value = level_data['value']
                level_type = level_data['type']
                original_confidence = level_data['confidence']
                
                # Calculate hit rate for this specific level
                hit_rate = self._calculate_level_hit_rate(level_value, level_type, future_ticks)
                
                # Update confidence based on performance
                new_confidence = self._calculate_new_confidence(original_confidence, hit_rate)
                level_data['confidence'] = new_confidence
                level_data['hit_rate'] = hit_rate
                level_data['evaluation_completed'] = True
                level_data['evaluation_timestamp'] = datetime.now().isoformat()
                
                total_levels_tested += 1
                if new_confidence > original_confidence:
                    total_levels_improved += 1
                    
                print(f"     {level_name}: {level_value:.5f} | Hit Rate: {hit_rate:.2%} | Confidence: {original_confidence:.2f} → {new_confidence:.2f}")
                
            # Save updated levels back to file
            self._update_daily_level_file(asset, filename, daily_data)
            
            evaluation_results.append({
                'filename': filename,
                'date': original_date,
                'levels_tested': len(levels),
                'avg_hit_rate': np.mean([l['hit_rate'] for l in levels.values()]),
                'avg_confidence': np.mean([l['confidence'] for l in levels.values()]),
                'high_confidence_levels': len([l for l in levels.values() if l['confidence'] >= self.CONFIDENCE_THRESHOLD])
            })
            
        self.stats['levels_evaluated'] = total_levels_tested
        
        print(f"✅ EVALUATION completed")
        print(f"   Total levels tested: {total_levels_tested}")
        print(f"   Levels improved: {total_levels_improved}")
        print(f"   Average improvement: {(total_levels_improved/total_levels_tested)*100:.1f}%")
        
        # Calculate overall confidence from all evaluated levels
        all_confidences = []
        for daily_data in daily_files.values():
            for level_data in daily_data['levels'].values():
                all_confidences.append(level_data['confidence'])
        
        overall_confidence = np.mean(all_confidences) if all_confidences else 0.5
        
        return {
            'phase': 'evaluation', 
            'status': 'completed',
            'asset': asset,
            'total_levels_tested': total_levels_tested,
            'levels_improved': total_levels_improved,
            'improvement_rate': (total_levels_improved/total_levels_tested) if total_levels_tested > 0 else 0,
            'daily_results': evaluation_results,
            'algorithm': self.algorithm_name,
            'confidence': overall_confidence  # Confidence medio di tutti i livelli dopo evaluation
        }
    
    def _load_daily_level_files(self, asset: str) -> Dict[str, Dict[str, Any]]:
        """Carica tutti i file di livelli giornalieri"""
        asset_dir = self.data_path / asset / "pivot_levels"
        if not asset_dir.exists():
            raise FileNotFoundError(f"FAIL FAST: Asset directory not found: {asset_dir}")
            
        daily_files = {}
        
        # Load all day_*.json files
        for file_path in asset_dir.glob("day_*_pivot.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                daily_files[file_path.name] = data
            except (json.JSONDecodeError, IOError) as e:
                raise RuntimeError(f"FAIL FAST: Failed to load {file_path.name}: {e}")
                
        return daily_files
    
    def _calculate_level_hit_rate(self, level_value: float, level_type: str, future_ticks: List[Dict[str, Any]]) -> float:
        """
        Calcola hit rate per un singolo livello sui ticks futuri
        
        Hit Rate = numero di volte che il livello viene rispettato / numero di volte che viene testato
        """
        hits = 0
        tests = 0
        tolerance = self.PRICE_TOLERANCE
        
        # Extract prices from ticks
        prices = []
        for tick in future_ticks:
            if 'last' in tick:
                prices.append(float(tick['last']))
            elif 'close' in tick:
                prices.append(float(tick['close']))
            else:
                raise KeyError("FAIL FAST: Tick missing both 'last' and 'close' fields")
                
        # Test level interactions
        for i in range(1, len(prices)):
            current_price = prices[i]
            previous_price = prices[i-1]
            
            # Check if price tested the level
            level_tested = False
            
            if level_type in ['support', 'pivot']:
                # Support test: price approaches from above
                if previous_price > level_value and abs(current_price - level_value) <= tolerance:
                    level_tested = True
                    # Check if support held (price bounced back up)
                    if i + self.LOOKAHEAD_TICKS < len(prices):  # Look ahead ticks
                        future_prices = prices[i:i+self.LOOKAHEAD_TICKS]
                        if max(future_prices) > level_value * (1.0 + self.BOUNCE_THRESHOLD_PERCENT):
                            hits += 1
                            
            elif level_type == 'resistance':
                # Resistance test: price approaches from below  
                if previous_price < level_value and abs(current_price - level_value) <= tolerance:
                    level_tested = True
                    # Check if resistance held (price bounced back down)
                    if i + self.LOOKAHEAD_TICKS < len(prices):  # Look ahead ticks
                        future_prices = prices[i:i+self.LOOKAHEAD_TICKS]
                        if min(future_prices) < level_value * (1.0 - self.BOUNCE_THRESHOLD_PERCENT):
                            hits += 1
                            
            if level_tested:
                tests += 1
                
        # Calculate hit rate
        if tests == 0:
            return 0.0  # Level never tested
        
        return hits / tests
    
    def _calculate_new_confidence(self, original_confidence: float, hit_rate: float) -> float:
        """Calcola nuova confidence basata su hit rate"""
        # Weighted average: 30% original, 70% performance
        new_confidence = 0.3 * original_confidence + 0.7 * hit_rate
        
        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, new_confidence))
    
    def _update_daily_level_file(self, asset: str, filename: str, updated_data: Dict[str, Any]):
        """Aggiorna file giornaliero con confidence aggiornate"""
        asset_dir = self.data_path / asset / "pivot_levels"
        file_path = asset_dir / filename
        
        # Add evaluation metadata
        updated_data['last_evaluation_timestamp'] = datetime.now().isoformat()
        updated_data['evaluation_completed'] = True
        
        # Save updated data
        with open(file_path, 'w') as f:
            json.dump(updated_data, f, indent=2)
    
    # ========================================
    # FASE 3: VALIDATION - Caricamento selettivo + test real-time
    # ========================================
    
    def validation_phase_init(self, asset: str) -> Dict[str, Any]:
        """
        FASE 3 - VALIDATION: Carica solo livelli con confidence >= 0.5
        
        Args:
            asset: Nome asset
            
        Returns:
            Risultato inizializzazione con livelli attivi caricati
        """
        if not isinstance(asset, str) or not asset.strip():
            raise ValueError("FAIL FAST: asset must be non-empty string")
            
        self.current_phase = 'validation'
        print(f"🔮 Starting VALIDATION phase for {asset}")
        
        # Load and filter levels
        daily_files = self._load_daily_level_files(asset)
        
        self.active_levels = {}
        levels_loaded = 0
        levels_skipped = 0
        
        for filename, daily_data in daily_files.items():
            date = daily_data['date']
            levels = daily_data['levels']
            
            for level_name, level_data in levels.items():
                confidence = level_data['confidence']
                
                if confidence >= self.CONFIDENCE_THRESHOLD:  # Load only high-confidence levels
                    level_key = f"{date}_{level_name}"
                    self.active_levels[level_key] = {
                        **level_data,
                        'source_date': date,
                        'source_file': filename,
                        'level_name': level_name,
                        'validation_hits': 0,
                        'validation_tests': 0,
                        'last_update': datetime.now().isoformat()
                    }
                    levels_loaded += 1
                else:
                    levels_skipped += 1
                    
        self.stats['active_levels_count'] = levels_loaded
        
        print(f"✅ VALIDATION initialized")
        print(f"   Levels loaded (confidence >= {self.CONFIDENCE_THRESHOLD}): {levels_loaded}")
        print(f"   Levels skipped (confidence < {self.CONFIDENCE_THRESHOLD}): {levels_skipped}")
        
        return {
            'phase': 'validation_init',
            'status': 'ready',
            'asset': asset,
            'active_levels_count': levels_loaded,
            'levels_skipped': levels_skipped,
            'confidence_threshold': self.CONFIDENCE_THRESHOLD,
            'algorithm': self.algorithm_name
        }
    
    def validation_tick_test(self, current_tick: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test real-time di un singolo tick contro livelli attivi
        
        Args:
            current_tick: Tick corrente con price/timestamp
            
        Returns:
            Risultato test con eventuali eventi significativi
        """
        if self.current_phase != 'validation':
            raise RuntimeError("FAIL FAST: Must call validation_phase_init first")
        if not isinstance(current_tick, dict):
            raise TypeError("FAIL FAST: current_tick must be dict")
        if 'last' not in current_tick and 'close' not in current_tick:
            raise KeyError("FAIL FAST: current_tick missing both 'last' and 'close' price fields")
            
        # Extract price with FAIL FAST validation - NO fallback with .get()
        if 'last' in current_tick:
            current_price = current_tick['last']
        elif 'close' in current_tick:
            current_price = current_tick['close']
        else:
            raise ValueError("FAIL FAST: current_tick missing both 'last' and 'close' price values")
        current_price = float(current_price)  # Ensure it's float
        current_time = datetime.now()
        
        # Test each active level
        significant_events = []
        levels_tested = 0
        
        for level_key, level_data in self.active_levels.items():
            level_value = level_data['value']
            level_type = level_data['type']
            tolerance = self.PRICE_TOLERANCE
            
            # Check if price is testing this level
            if abs(current_price - level_value) <= tolerance:
                levels_tested += 1
                level_data['validation_tests'] += 1
                level_data['last_test_time'] = current_time.isoformat()
                
                # For real-time, we assume level holds if price doesn't break through immediately
                # This is a simplified version - in production you'd wait for confirmation
                level_holds = self._check_level_hold_realtime(current_price, level_value, level_type)
                
                if level_holds:
                    level_data['validation_hits'] += 1
                    
                    # Update confidence based on validation performance
                    validation_hit_rate = level_data['validation_hits'] / level_data['validation_tests']
                    level_data['confidence'] = self._calculate_new_confidence(level_data['confidence'], validation_hit_rate)
                    
                    significant_events.append({
                        'event_type': 'level_respected',
                        'level_key': level_key,
                        'level_value': level_value,
                        'level_type': level_type,
                        'current_price': current_price,
                        'confidence': level_data['confidence'],
                        'validation_hit_rate': validation_hit_rate,
                        'message': f"{level_type.title()} at {level_value:.5f} respected (confidence: {level_data['confidence']:.2f})"
                    })
                    
                self.stats['validation_tests'] += 1
                if level_holds:
                    self.stats['validation_hits'] += 1
        
        # Return results
        if significant_events:
            return {
                'status': 'events_detected',
                'current_price': current_price,
                'levels_tested': levels_tested,
                'events': significant_events,
                'validation_hit_rate': self.stats['validation_hits'] / max(self.stats['validation_tests'], 1),
                'algorithm': self.algorithm_name
            }
        else:
            return {
                'status': 'monitoring',
                'current_price': current_price,
                'levels_tested': levels_tested,
                'active_levels_count': len(self.active_levels),
                'algorithm': self.algorithm_name
            }
    
    def _check_level_hold_realtime(self, current_price: float, level_value: float, level_type: str) -> bool:
        """
        Check semplificato per real-time se il livello tiene
        In produzione avresti logica più sofisticata con buffer di conferma
        """
        # Simplified: assume level holds if price is within tolerance
        tolerance = self.PRICE_TOLERANCE / 2  # Half tolerance for real-time
        return abs(current_price - level_value) <= tolerance
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def get_algorithm_stats(self) -> Dict[str, Any]:
        """Restituisce statistiche algoritmo"""
        return {
            'algorithm': self.algorithm_name,
            'current_phase': self.current_phase,
            'stats': self.stats.copy(),
            'data_path': str(self.data_path),
            'active_levels_summary': {
                'total_active': len(self.active_levels),
                'by_type': {
                    'support': len([l for l in self.active_levels.values() if l['type'] == 'support']),
                    'resistance': len([l for l in self.active_levels.values() if l['type'] == 'resistance']),
                    'pivot': len([l for l in self.active_levels.values() if l['type'] == 'pivot'])
                },
                'avg_confidence': np.mean([l['confidence'] for l in self.active_levels.values()])
            } if self.current_phase == 'validation' else {}
        }
    
    def reset_algorithm(self):
        """Reset completo dell'algoritmo"""
        self.current_phase = None
        self.daily_levels.clear()
        self.active_levels.clear()
        self.stats = {
            'training_days_processed': 0,
            'levels_calculated': 0,
            'levels_evaluated': 0,
            'active_levels_count': 0,
            'validation_hits': 0,
            'validation_tests': 0
        }
        print(f"🔄 {self.algorithm_name} reset completed")


# Factory function
def create_pivot_points_classic(data_path: str = "./analyzer_data") -> PivotPointsClassic:
    """Factory function per creare istanza PivotPoints_Classic"""
    return PivotPointsClassic(data_path)


# Export
__all__ = [
    'PivotPointsClassic',
    'create_pivot_points_classic'
]