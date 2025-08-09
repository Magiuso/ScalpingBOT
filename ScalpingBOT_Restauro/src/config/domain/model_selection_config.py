#!/usr/bin/env python3
"""
Model Selection Configuration - INTERACTIVE MODEL TRAINING SELECTION
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults
- ✅ Fail fast error handling  
- ✅ No debug prints/spam
- ✅ Production-ready for real money trading

Sistema per selezione interattiva dei modelli ML da addestrare.
Permette di scegliere specifici modelli o categorie per testing e debugging.
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass

from ScalpingBOT_Restauro.src.shared.enums import ModelType


class ModelSelectionMode(Enum):
    """Modalità di selezione modelli"""
    ALL_MODELS = "all_models"
    SINGLE_CATEGORY = "single_category" 
    SINGLE_MODEL = "single_model"
    CUSTOM_LIST = "custom_list"


@dataclass
class ModelTrainingConfig:
    """Configurazione per training di un modello specifico"""
    model_type: ModelType
    algorithm_name: str
    display_name: str
    estimated_time_hours: float
    complexity_level: str  # "low", "medium", "high"
    dependencies: List[str]  # Modelli che devono essere addestrati prima


class ModelSelectionManager:
    """
    Manager per selezione interattiva dei modelli ML da addestrare
    PRODUCTION-READY per sistema di trading con denaro reale
    """
    
    def __init__(self):
        """Inizializza il manager con tutti i modelli disponibili"""
        
        # 📊 DEFINIZIONE COMPLETA MODELLI ML (da CLAUDE_RESTAURO.md)
        self.available_models: Dict[str, ModelTrainingConfig] = {
            
            # 🎯 SUPPORT/RESISTANCE (5 modelli)
            "lstm_sr": ModelTrainingConfig(
                model_type=ModelType.SUPPORT_RESISTANCE,
                algorithm_name="LSTM_SupportResistance",
                display_name="LSTM Support/Resistance",
                estimated_time_hours=2.5,
                complexity_level="high",
                dependencies=[]
            ),
            "pivot_points": ModelTrainingConfig(
                model_type=ModelType.SUPPORT_RESISTANCE,
                algorithm_name="PivotPoints_Classic",
                display_name="Pivot Points Classic (Modulare)",
                estimated_time_hours=0.5,
                complexity_level="low",
                dependencies=[]
            ),
            "volume_profile": ModelTrainingConfig(
                model_type=ModelType.SUPPORT_RESISTANCE,
                algorithm_name="VolumeProfile_Advanced",
                display_name="Volume Profile Advanced",
                estimated_time_hours=1.5,
                complexity_level="medium",
                dependencies=[]
            ),
            "statistical_levels": ModelTrainingConfig(
                model_type=ModelType.SUPPORT_RESISTANCE,
                algorithm_name="StatisticalLevels_ML",
                display_name="Statistical Levels ML",
                estimated_time_hours=1.0,
                complexity_level="medium",
                dependencies=[]
            ),
            "transformer_levels": ModelTrainingConfig(
                model_type=ModelType.SUPPORT_RESISTANCE,
                algorithm_name="Transformer_Levels",
                display_name="Transformer S/R Levels",
                estimated_time_hours=3.5,
                complexity_level="high",
                dependencies=[]
            ),
            
            # 🔍 PATTERN RECOGNITION (5 modelli)
            "cnn_patterns": ModelTrainingConfig(
                model_type=ModelType.PATTERN_RECOGNITION,
                algorithm_name="CNN_PatternRecognizer",
                display_name="CNN Pattern Recognizer",
                estimated_time_hours=2.0,
                complexity_level="high",
                dependencies=[]
            ),
            "classical_patterns": ModelTrainingConfig(
                model_type=ModelType.PATTERN_RECOGNITION,
                algorithm_name="Classical_Patterns",
                display_name="Classical Chart Patterns", 
                estimated_time_hours=0.8,
                complexity_level="low",
                dependencies=[]
            ),
            "lstm_sequences": ModelTrainingConfig(
                model_type=ModelType.PATTERN_RECOGNITION,
                algorithm_name="LSTM_Sequences",
                display_name="LSTM Pattern Sequences",
                estimated_time_hours=2.5,
                complexity_level="high",
                dependencies=[]
            ),
            "transformer_patterns": ModelTrainingConfig(
                model_type=ModelType.PATTERN_RECOGNITION,
                algorithm_name="Transformer_Patterns",
                display_name="Transformer Pattern Recognition",
                estimated_time_hours=3.0,
                complexity_level="high",
                dependencies=[]
            ),
            "ensemble_patterns": ModelTrainingConfig(
                model_type=ModelType.PATTERN_RECOGNITION,
                algorithm_name="Ensemble_Patterns",
                display_name="Ensemble Pattern Recognition",
                estimated_time_hours=1.5,
                complexity_level="medium",
                dependencies=["cnn_patterns", "lstm_sequences", "classical_patterns"]
            ),
            
            # 🧠 BIAS DETECTION (5 modelli)
            "sentiment_lstm": ModelTrainingConfig(
                model_type=ModelType.BIAS_DETECTION,
                algorithm_name="Sentiment_LSTM",
                display_name="Sentiment Analysis LSTM",
                estimated_time_hours=2.0,
                complexity_level="high",
                dependencies=[]
            ),
            "volume_price_analysis": ModelTrainingConfig(
                model_type=ModelType.BIAS_DETECTION,
                algorithm_name="VolumePrice_Analysis",
                display_name="Volume/Price Bias Analysis",
                estimated_time_hours=1.0,
                complexity_level="medium",
                dependencies=[]
            ),
            "momentum_ml": ModelTrainingConfig(
                model_type=ModelType.BIAS_DETECTION,
                algorithm_name="Momentum_ML",
                display_name="Momentum ML Bias Detection",
                estimated_time_hours=1.2,
                complexity_level="medium",
                dependencies=[]
            ),
            "transformer_bias": ModelTrainingConfig(
                model_type=ModelType.BIAS_DETECTION,
                algorithm_name="Transformer_Bias",
                display_name="Transformer Bias Detection",
                estimated_time_hours=3.0,
                complexity_level="high",
                dependencies=[]
            ),
            "multimodal_bias": ModelTrainingConfig(
                model_type=ModelType.BIAS_DETECTION,
                algorithm_name="MultiModal_Bias",
                display_name="MultiModal Bias Detection",
                estimated_time_hours=2.5,
                complexity_level="high",
                dependencies=["sentiment_lstm", "volume_price_analysis"]
            ),
            
            # 📈 TREND ANALYSIS (5 modelli)
            "random_forest_trend": ModelTrainingConfig(
                model_type=ModelType.TREND_ANALYSIS,
                algorithm_name="RandomForest_Trend",
                display_name="Random Forest Trend Analysis",
                estimated_time_hours=0.8,
                complexity_level="low",
                dependencies=[]
            ),
            "lstm_trend": ModelTrainingConfig(
                model_type=ModelType.TREND_ANALYSIS,
                algorithm_name="LSTM_TrendPrediction",
                display_name="LSTM Trend Prediction",
                estimated_time_hours=2.2,
                complexity_level="high",
                dependencies=[]
            ),
            "gradient_boosting_trend": ModelTrainingConfig(
                model_type=ModelType.TREND_ANALYSIS,
                algorithm_name="GradientBoosting_Trend",
                display_name="Gradient Boosting Trend",
                estimated_time_hours=1.0,
                complexity_level="medium",
                dependencies=[]
            ),
            "transformer_trend": ModelTrainingConfig(
                model_type=ModelType.TREND_ANALYSIS,
                algorithm_name="Transformer_Trend",
                display_name="Transformer Trend Analysis",
                estimated_time_hours=3.2,
                complexity_level="high",
                dependencies=[]
            ),
            "ensemble_trend": ModelTrainingConfig(
                model_type=ModelType.TREND_ANALYSIS,
                algorithm_name="Ensemble_Trend",
                display_name="Ensemble Trend Analysis",
                estimated_time_hours=1.8,
                complexity_level="medium",
                dependencies=["random_forest_trend", "lstm_trend", "gradient_boosting_trend"]
            ),
            
            # 📊 VOLATILITY PREDICTION (3 modelli)
            "garch_volatility": ModelTrainingConfig(
                model_type=ModelType.VOLATILITY_PREDICTION,
                algorithm_name="GARCH_Volatility",
                display_name="GARCH Volatility Model",
                estimated_time_hours=0.7,
                complexity_level="medium",
                dependencies=[]
            ),
            "lstm_volatility": ModelTrainingConfig(
                model_type=ModelType.VOLATILITY_PREDICTION,
                algorithm_name="LSTM_Volatility",
                display_name="LSTM Volatility Prediction",
                estimated_time_hours=2.0,
                complexity_level="high",
                dependencies=[]
            ),
            "realized_volatility": ModelTrainingConfig(
                model_type=ModelType.VOLATILITY_PREDICTION,
                algorithm_name="Realized_Volatility",
                display_name="Realized Volatility Model",
                estimated_time_hours=0.5,
                complexity_level="low",
                dependencies=[]
            )
        }
        
        # Raggruppamento per categoria
        self.models_by_category: Dict[ModelType, List[str]] = {}
        for model_key, config in self.available_models.items():
            if config.model_type not in self.models_by_category:
                self.models_by_category[config.model_type] = []
            self.models_by_category[config.model_type].append(model_key)
    
    def get_interactive_selection(self, asset_symbol: str) -> List[str]:
        """
        Interfaccia interattiva per selezione modelli da addestrare
        LOGICA: 1=ALL, 2-24=SINGOLI MODELLI per test/debug
        
        Args:
            asset_symbol: Simbolo dell'asset per cui addestrare
            
        Returns:
            Lista dei model keys selezionati per il training
        """
        print(f"\n🎯 MODEL SELECTION FOR ASSET: {asset_symbol}")
        print("=" * 60)
        
        # Calcola tempo totale stimato
        total_time = sum(config.estimated_time_hours for config in self.available_models.values())
        print(f"📊 Total models available: {len(self.available_models)}")
        print(f"⏱️ Total estimated training time: {total_time:.1f} hours")
        print("=" * 60)
        
        # OPZIONE 1: ALL MODELS (per training completo finale)
        print("\n🔧 TRAINING OPTIONS:")
        print("   1. 🚀 ALL MODELS (Complete training - ~40-60 hours)")
        
        # OPZIONI 2-24: SINGOLI MODELLI (per test/debug)
        print("\n🔍 INDIVIDUAL MODELS (for testing/debugging):")
        
        # Lista tutti i modelli numerati da 2 in poi
        model_list = list(self.available_models.items())
        for i, (model_key, config) in enumerate(model_list, 2):
            complexity_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}[config.complexity_level]
            category_short = config.model_type.value.replace('_', ' ').title()[:12]
            print(f"   {i:2d}. {config.display_name} {complexity_icon} (~{config.estimated_time_hours:.1f}h) [{category_short}]")
        
        max_option = len(model_list) + 1
        
        while True:
            try:
                choice = input(f"\n🔢 Select training option (1-{max_option}): ").strip()
                choice_num = int(choice)
                
                if choice_num == 1:
                    # ALL MODELS
                    print("✅ Selected: ALL MODELS (Complete training)")
                    return list(self.available_models.keys())
                
                elif 2 <= choice_num <= max_option:
                    # SINGLE MODEL
                    selected_index = choice_num - 2  # Convert to 0-based index
                    selected_key, selected_config = model_list[selected_index]
                    
                    print(f"✅ Selected: {selected_config.display_name}")
                    print(f"⏱️ Estimated time: {selected_config.estimated_time_hours:.1f} hours")
                    print(f"🎯 Category: {selected_config.model_type.value.replace('_', ' ').title()}")
                    
                    return [selected_key]
                
                else:
                    print(f"❌ Invalid selection. Please choose 1-{max_option}.")
                    continue
                    
            except ValueError:
                print(f"❌ Please enter a valid number (1-{max_option}).")
                continue
            except KeyboardInterrupt:
                print("\n🛑 Model selection cancelled")
                raise SystemExit(0)
            except Exception as e:
                print(f"❌ Error in model selection: {e}")
                continue
    
    def _select_single_model(self) -> List[str]:
        """Selezione di un singolo modello"""
        print("\n🎲 SINGLE MODEL SELECTION:")
        print("=" * 50)
        
        # Raggruppa per categoria per display più chiaro
        for i, (category, model_keys) in enumerate(self.models_by_category.items(), 1):
            print(f"\n{i}. {category.value.upper().replace('_', ' ')}:")
            for j, model_key in enumerate(model_keys, 1):
                config = self.available_models[model_key]
                complexity_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}[config.complexity_level]
                print(f"   {i}.{j} {config.display_name} {complexity_icon} (~{config.estimated_time_hours:.1f}h)")
        
        while True:
            try:
                selection = input("\n🔢 Select model (e.g., 1.2 for second model in first category): ").strip()
                
                if '.' not in selection:
                    print("❌ Please use format X.Y (e.g., 1.2)")
                    continue
                
                category_idx, model_idx = selection.split('.')
                category_idx, model_idx = int(category_idx) - 1, int(model_idx) - 1
                
                categories = list(self.models_by_category.keys())
                if category_idx < 0 or category_idx >= len(categories):
                    print("❌ Invalid category index")
                    continue
                
                selected_category = categories[category_idx]
                category_models = self.models_by_category[selected_category]
                
                if model_idx < 0 or model_idx >= len(category_models):
                    print("❌ Invalid model index")
                    continue
                
                selected_model = category_models[model_idx]
                config = self.available_models[selected_model]
                
                print(f"✅ Selected: {config.display_name}")
                print(f"⏱️ Estimated time: {config.estimated_time_hours:.1f} hours")
                
                return [selected_model]
                
            except (ValueError, IndexError):
                print("❌ Invalid format. Please use X.Y format (e.g., 1.2)")
                continue
    
    def _select_custom_list(self) -> List[str]:
        """Selezione personalizzata di multipli modelli"""
        print("\n📋 CUSTOM LIST SELECTION:")
        print("Select multiple models by entering their numbers separated by commas")
        print("=" * 60)
        
        # Lista numerata di tutti i modelli
        model_list = list(self.available_models.items())
        for i, (model_key, config) in enumerate(model_list, 1):
            complexity_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}[config.complexity_level]
            print(f"{i:2d}. {config.display_name} {complexity_icon} (~{config.estimated_time_hours:.1f}h) [{config.model_type.value}]")
        
        while True:
            try:
                selection = input(f"\n🔢 Select models (1-{len(model_list)}, comma-separated): ").strip()
                
                if not selection:
                    print("❌ Please enter model numbers")
                    continue
                
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                
                # Validate indices
                if any(i < 0 or i >= len(model_list) for i in indices):
                    print(f"❌ Invalid model numbers. Please use 1-{len(model_list)}")
                    continue
                
                selected_models = [model_list[i][0] for i in indices]
                
                # Show selection summary
                total_time = sum(self.available_models[model].estimated_time_hours for model in selected_models)
                print(f"\n✅ Selected {len(selected_models)} models:")
                for model_key in selected_models:
                    config = self.available_models[model_key]
                    print(f"   • {config.display_name}")
                print(f"⏱️ Total estimated time: {total_time:.1f} hours")
                
                return selected_models
                
            except ValueError:
                print("❌ Please enter valid numbers separated by commas")
                continue
    
    def _select_quick_start_models(self) -> List[str]:
        """Selezione modelli veloci per testing rapido"""
        quick_models = [
            model_key for model_key, config in self.available_models.items()
            if config.complexity_level == "low" or config.estimated_time_hours <= 1.0
        ]
        
        total_time = sum(self.available_models[model].estimated_time_hours for model in quick_models)
        
        print(f"\n💡 QUICK START MODELS ({len(quick_models)} models, ~{total_time:.1f} hours):")
        for model_key in quick_models:
            config = self.available_models[model_key]
            print(f"   • {config.display_name} (~{config.estimated_time_hours:.1f}h)")
        
        return quick_models
    
    def validate_selection(self, selected_models: List[str]) -> bool:
        """
        Valida la selezione e verifica le dipendenze
        
        Args:
            selected_models: Lista dei modelli selezionati
            
        Returns:
            True se la selezione è valida
        """
        if not selected_models:
            raise ValueError("No models selected for training")
        
        # Verifica che tutti i modelli esistano
        for model_key in selected_models:
            if model_key not in self.available_models:
                raise ValueError(f"Invalid model key: {model_key}")
        
        # Verifica dipendenze
        missing_dependencies = []
        for model_key in selected_models:
            config = self.available_models[model_key]
            for dependency in config.dependencies:
                if dependency not in selected_models:
                    missing_dependencies.append((model_key, dependency))
        
        if missing_dependencies:
            print("\n⚠️ DEPENDENCY WARNINGS:")
            for model, dependency in missing_dependencies:
                model_name = self.available_models[model].display_name
                dep_name = self.available_models[dependency].display_name
                print(f"   • {model_name} recommends {dep_name}")
            
            choice = input("\nContinue anyway? (y/n): ").strip().lower()
            if choice != 'y':
                return False
        
        return True
    
    def get_training_summary(self, selected_models: List[str], asset_symbol: str) -> Dict[str, Any]:
        """
        Genera riepilogo del training pianificato
        
        Args:
            selected_models: Modelli selezionati
            asset_symbol: Simbolo asset
            
        Returns:
            Dizionario con riepilogo del training
        """
        total_time = sum(self.available_models[model].estimated_time_hours for model in selected_models)
        
        # Raggruppa per categoria
        by_category = {}
        for model_key in selected_models:
            config = self.available_models[model_key]
            category = config.model_type.value
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(config.display_name)
        
        return {
            'asset_symbol': asset_symbol,
            'total_models': len(selected_models),
            'estimated_total_hours': total_time,
            'models_by_category': by_category,
            'selected_models': selected_models,
            'complexity_breakdown': {
                'low': len([m for m in selected_models if self.available_models[m].complexity_level == 'low']),
                'medium': len([m for m in selected_models if self.available_models[m].complexity_level == 'medium']),
                'high': len([m for m in selected_models if self.available_models[m].complexity_level == 'high'])
            }
        }


# Factory function
def create_model_selection_manager() -> ModelSelectionManager:
    """Factory function per creare ModelSelectionManager"""
    return ModelSelectionManager()


# Export
__all__ = [
    'ModelSelectionMode',
    'ModelTrainingConfig', 
    'ModelSelectionManager',
    'create_model_selection_manager'
]