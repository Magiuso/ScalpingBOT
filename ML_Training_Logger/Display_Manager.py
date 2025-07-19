#!/usr/bin/env python3
"""
MLTrainingLogger - Display Manager (Progress Tree Layout)
========================================================

Gestisce la visualizzazione real-time degli eventi ML training nel terminale.
Layout ad albero con progress indicators e informazioni gerarchiche.

Author: ScalpingBOT Team
Version: 3.0.0
"""

import os
import sys
import time
import threading
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum

# Import configuration and events
from .Config_Manager import (
    MLTrainingLoggerConfig, TerminalMode, DisplaySettings
)
from .Event_Collector import MLEvent, EventType, EventSource, EventSeverity


class ColorCode(Enum):
    """ANSI color codes per output colorato"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"


@dataclass
class TerminalCapabilities:
    """Capabilities del terminale"""
    supports_ansi: bool = False
    supports_colors: bool = False
    width: int = 80
    height: int = 24
    supports_cursor_control: bool = False


@dataclass
class ModelProgress:
    """Progress di un singolo modello"""
    name: str
    progress: float = 0.0
    accuracy: float = 0.0
    status: str = "Training"
    predictions: int = 0
    is_champion: bool = False
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class DisplayMetrics:
    """Metriche per il display con modelli individuali"""
    learning_progress: float = 0.0
    duration_seconds: int = 0
    ticks_processed: int = 0
    processing_rate: float = 0.0
    champions_active: int = 0
    
    # Individual model progress
    models: Dict[str, ModelProgress] = field(default_factory=dict)
    
    # Performance metrics
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    health_score: float = 0.0
    
    # Recent events for log
    recent_events: List[MLEvent] = field(default_factory=list)
    
    # System info
    asset_symbol: str = "UNKNOWN"
    system_status: str = "RUNNING"


class TreeSymbols:
    """Simboli per layout ad albero"""
    BRANCH = "├─"
    LAST_BRANCH = "└─"
    CONTINUATION = "│  "
    SPACE = "   "
    
    # Progress symbols
    PROGRESS_FULL = "█"
    PROGRESS_EMPTY = "░"
    PROGRESS_PARTIAL = "▓"
    
    # Status indicators
    RUNNING = "🔄"
    SUCCESS = "✅"
    WARNING = "⚠️"
    ERROR = "❌"
    CHAMPION = "🏆"
    TRAINING = "🧠"
    DATA = "📊"
    PERFORMANCE = "⚡"
    HEALTH = "💚"
    TIME = "⏱️"


class ProgressBar:
    """Barra di progresso ASCII con simboli Unicode"""
    
    def __init__(self, width: int = 20):
        self.width = width
    
    def render(self, progress: float, show_percentage: bool = True) -> str:
        """Renderizza barra di progresso con colori"""
        progress = max(0.0, min(1.0, progress))
        filled = int(self.width * progress)
        empty = self.width - filled
        
        # Choose color based on progress
        if progress >= 0.8:
            color = ColorCode.BRIGHT_GREEN.value
        elif progress >= 0.5:
            color = ColorCode.BRIGHT_YELLOW.value
        else:
            color = ColorCode.BRIGHT_RED.value
        
        bar = (TreeSymbols.PROGRESS_FULL * filled + 
               TreeSymbols.PROGRESS_EMPTY * empty)
        
        if show_percentage:
            percentage = f"{progress * 100:.1f}%"
            return f"{color}{bar}{ColorCode.RESET.value} {percentage}"
        else:
            return f"{color}{bar}{ColorCode.RESET.value}"


class TreeProgressRenderer:
    """Renderer principale con layout ad albero"""
    
    def __init__(self, config: DisplaySettings, capabilities: TerminalCapabilities):
        self.config = config
        self.capabilities = capabilities
        self.progress_bar = ProgressBar(width=25)
        self.last_render = datetime.now()
        
        # Event formatting settings
        self.max_events_display = 8  # Numero massimo di eventi da mostrare
        self.event_timeout = timedelta(seconds=30)  # Timeout per eventi vecchi
    
    def render_full_status(self, metrics: DisplayMetrics) -> str:
        """Renderizza status completo con layout ad albero"""
        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Header principale
        header = self._render_header(metrics, timestamp)
        lines.extend(header)
        
        # Main progress section
        progress_section = self._render_progress_section(metrics)
        lines.extend(progress_section)
        
        # Models section
        if metrics.models:
            models_section = self._render_models_section(metrics)
            lines.extend(models_section)
        
        # System stats section
        system_section = self._render_system_section(metrics)
        lines.extend(system_section)
        
        # Recent events section
        if metrics.recent_events:
            events_section = self._render_events_section(metrics)
            lines.extend(events_section)
        
        # Footer
        lines.append("")
        
        return "\n".join(lines)
    
    def _render_header(self, metrics: DisplayMetrics, timestamp: str) -> List[str]:
        """Renderizza header principale"""
        lines = []
        
        # Status icon based on system health
        if metrics.health_score >= 80:
            status_icon = TreeSymbols.SUCCESS
            status_color = ColorCode.BRIGHT_GREEN.value
        elif metrics.health_score >= 60:
            status_icon = TreeSymbols.WARNING
            status_color = ColorCode.BRIGHT_YELLOW.value
        else:
            status_icon = TreeSymbols.ERROR
            status_color = ColorCode.BRIGHT_RED.value
        
        # Main title with asset and status
        if self.config.color_enabled:
            title = (f"[{timestamp}] {status_icon} "
                    f"{ColorCode.BOLD.value}{ColorCode.BRIGHT_CYAN.value}ML Training{ColorCode.RESET.value} "
                    f"{ColorCode.BRIGHT_WHITE.value}{metrics.asset_symbol}{ColorCode.RESET.value}")
        else:
            title = f"[{timestamp}] {status_icon} ML Training {metrics.asset_symbol}"
        
        lines.append(title)
        
        return lines
    
    def _render_progress_section(self, metrics: DisplayMetrics) -> List[str]:
        """Renderizza sezione progresso principale"""
        lines = []
        
        # Overall progress
        duration_str = self._format_duration(metrics.duration_seconds)
        progress_bar = self.progress_bar.render(metrics.learning_progress / 100.0)
        
        if self.config.color_enabled:
            lines.append(f"{TreeSymbols.BRANCH} {TreeSymbols.DATA} "
                        f"{ColorCode.BOLD.value}Progress:{ColorCode.RESET.value} {progress_bar} ({duration_str})")
        else:
            lines.append(f"{TreeSymbols.BRANCH} {TreeSymbols.DATA} Progress: {progress_bar} ({duration_str})")
        
        # Champions and health on same level
        champions_icon = TreeSymbols.CHAMPION if metrics.champions_active > 0 else TreeSymbols.WARNING
        health_icon = TreeSymbols.HEALTH if metrics.health_score >= 70 else TreeSymbols.WARNING
        
        if self.config.color_enabled:
            champions_color = ColorCode.BRIGHT_YELLOW.value if metrics.champions_active > 0 else ColorCode.DIM.value
            health_color = ColorCode.BRIGHT_GREEN.value if metrics.health_score >= 70 else ColorCode.BRIGHT_YELLOW.value
            
            lines.append(f"{TreeSymbols.BRANCH} {champions_icon} "
                        f"{champions_color}Champions: {metrics.champions_active}{ColorCode.RESET.value} | "
                        f"{health_icon} {health_color}Health: {metrics.health_score:.0f}%{ColorCode.RESET.value} | "
                        f"{TreeSymbols.DATA} Accuracy: {self._calculate_avg_accuracy(metrics):.1f}%")
        else:
            lines.append(f"{TreeSymbols.BRANCH} {champions_icon} Champions: {metrics.champions_active} | "
                        f"{health_icon} Health: {metrics.health_score:.0f}% | "
                        f"{TreeSymbols.DATA} Accuracy: {self._calculate_avg_accuracy(metrics):.1f}%")
        
        return lines
    
    def _render_models_section(self, metrics: DisplayMetrics) -> List[str]:
        """Renderizza sezione modelli - VERSIONE MIGLIORATA"""
        lines = []
        
        # Ensure all 6 model types are represented
        expected_models = ["support_resistance", "pattern_recognition", "bias_detection", 
                          "trend_analysis", "volatility_prediction", "momentum_analysis"]
        
        if self.config.color_enabled:
            models_title = (f"{TreeSymbols.BRANCH} {TreeSymbols.TRAINING} "
                           f"{ColorCode.BOLD.value}Models ({len(metrics.models)}/6):{ColorCode.RESET.value}")
        else:
            models_title = f"{TreeSymbols.BRANCH} {TreeSymbols.TRAINING} Models ({len(metrics.models)}/6):"
        
        lines.append(models_title)
        
        # Sort models: champions first, then by accuracy
        sorted_models = sorted(
            metrics.models.items(),
            key=lambda x: (not x[1].is_champion, -x[1].accuracy)
        )
        
        for i, (model_name, model_data) in enumerate(sorted_models):
            is_last = i == len(sorted_models) - 1
            prefix = TreeSymbols.CONTINUATION + (TreeSymbols.LAST_BRANCH if is_last else TreeSymbols.BRANCH)
            
            # Model status icon
            if model_data.is_champion:
                status_icon = TreeSymbols.CHAMPION
                status_color = ColorCode.BRIGHT_YELLOW.value if self.config.color_enabled else ""
            elif model_data.status == "Training":
                status_icon = TreeSymbols.RUNNING
                status_color = ColorCode.BRIGHT_BLUE.value if self.config.color_enabled else ""
            elif model_data.status == "Complete":
                status_icon = TreeSymbols.SUCCESS
                status_color = ColorCode.BRIGHT_GREEN.value if self.config.color_enabled else ""
            else:
                status_icon = TreeSymbols.WARNING
                status_color = ColorCode.BRIGHT_RED.value if self.config.color_enabled else ""
            
            # Extract model type from full name (e.g., "USTEC_support_resistance" -> "support_resistance")
            model_type = model_name.split('_', 1)[1] if '_' in model_name else model_name
            
            # More compact and functional display
            if model_data.is_champion:
                champion_indicator = "🏆"
            else:
                champion_indicator = "  "
                
            # Show status more clearly
            if model_data.status == "Training":
                status_indicator = "🔄"
            elif model_data.status == "Complete":
                status_indicator = "✅"
            elif model_data.status == "Collecting Data":
                status_indicator = "📊"
            else:
                status_indicator = "⚠️"
            
            # Compact display with essential info
            if self.config.color_enabled:
                model_line = (f"{prefix} {champion_indicator} {status_indicator} "
                             f"{status_color}{model_type:<20}{ColorCode.RESET.value} "
                             f"Acc: {model_data.accuracy:4.1f}% | Pred: {model_data.predictions:4d}")
            else:
                model_line = (f"{prefix} {champion_indicator} {status_indicator} "
                             f"{model_type:<20} Acc: {model_data.accuracy:4.1f}% | Pred: {model_data.predictions:4d}")
            
            lines.append(model_line)
        
        return lines
    
    def _render_system_section(self, metrics: DisplayMetrics) -> List[str]:
        """Renderizza sezione sistema"""
        lines = []
        
        if self.config.color_enabled:
            system_title = (f"{TreeSymbols.BRANCH} {TreeSymbols.PERFORMANCE} "
                           f"{ColorCode.BOLD.value}System:{ColorCode.RESET.value}")
        else:
            system_title = f"{TreeSymbols.BRANCH} {TreeSymbols.PERFORMANCE} System:"
        
        lines.append(system_title)
        
        # Ticks and rate
        rate_str = f"{metrics.processing_rate:.0f}/sec" if metrics.processing_rate > 0 else "N/A"
        lines.append(f"{TreeSymbols.CONTINUATION}{TreeSymbols.BRANCH} {TreeSymbols.DATA} "
                    f"Ticks: {metrics.ticks_processed:,} | Rate: {rate_str}")
        
        # Performance metrics if available
        if metrics.memory_usage > 0 or metrics.cpu_usage > 0:
            perf_parts = []
            if metrics.memory_usage > 0:
                mem_color = (ColorCode.BRIGHT_RED.value if metrics.memory_usage > 80 else 
                           ColorCode.BRIGHT_YELLOW.value if metrics.memory_usage > 60 else 
                           ColorCode.BRIGHT_GREEN.value) if self.config.color_enabled else ""
                reset = ColorCode.RESET.value if self.config.color_enabled else ""
                perf_parts.append(f"Mem: {mem_color}{metrics.memory_usage:.1f}%{reset}")
            
            if metrics.cpu_usage > 0:
                cpu_color = (ColorCode.BRIGHT_RED.value if metrics.cpu_usage > 80 else 
                           ColorCode.BRIGHT_YELLOW.value if metrics.cpu_usage > 60 else 
                           ColorCode.BRIGHT_GREEN.value) if self.config.color_enabled else ""
                reset = ColorCode.RESET.value if self.config.color_enabled else ""
                perf_parts.append(f"CPU: {cpu_color}{metrics.cpu_usage:.1f}%{reset}")
            
            if perf_parts:
                lines.append(f"{TreeSymbols.CONTINUATION}{TreeSymbols.LAST_BRANCH} {TreeSymbols.PERFORMANCE} "
                           f"{' | '.join(perf_parts)}")
        else:
            # Make the data line the last one
            lines[-1] = lines[-1].replace(TreeSymbols.BRANCH, TreeSymbols.LAST_BRANCH)
        
        return lines
    
    def _render_events_section(self, metrics: DisplayMetrics) -> List[str]:
        """Renderizza sezione eventi recenti"""
        lines = []
        
        # Filter and limit recent events
        now = datetime.now()
        recent_events = [
            event for event in metrics.recent_events
            if now - event.timestamp <= self.event_timeout
        ][-self.max_events_display:]
        
        if not recent_events:
            return lines
        
        if self.config.color_enabled:
            events_title = (f"{TreeSymbols.LAST_BRANCH} {TreeSymbols.RUNNING} "
                           f"{ColorCode.BOLD.value}Recent Events:{ColorCode.RESET.value}")
        else:
            events_title = f"{TreeSymbols.LAST_BRANCH} {TreeSymbols.RUNNING} Recent Events:"
        
        lines.append(events_title)
        
        for i, event in enumerate(recent_events):
            is_last = i == len(recent_events) - 1
            prefix = TreeSymbols.SPACE + (TreeSymbols.LAST_BRANCH if is_last else TreeSymbols.BRANCH)
            
            event_line = self._format_event_line(event, prefix)
            lines.append(event_line)
        
        return lines
    
    def _format_event_line(self, event: MLEvent, prefix: str) -> str:
        """Formatta singola riga evento"""
        timestamp = event.timestamp.strftime("%H:%M:%S")
        
        # Event type icon and color
        event_icons = {
            "learning_progress": TreeSymbols.DATA,
            "champion_change": TreeSymbols.CHAMPION,
            "model_training": TreeSymbols.TRAINING,
            "emergency_stop": TreeSymbols.ERROR,
            "performance_metrics": TreeSymbols.PERFORMANCE,
            "validation_complete": TreeSymbols.SUCCESS,
            "diagnostic": TreeSymbols.RUNNING,
        }
        
        severity_colors = {
            EventSeverity.DEBUG: ColorCode.DIM.value,
            EventSeverity.INFO: ColorCode.WHITE.value,
            EventSeverity.WARNING: ColorCode.BRIGHT_YELLOW.value,
            EventSeverity.ERROR: ColorCode.BRIGHT_RED.value,
            EventSeverity.CRITICAL: ColorCode.BRIGHT_RED.value + ColorCode.BOLD.value
        } if self.config.color_enabled else {}
        
        # Get event type string
        if isinstance(event.event_type, EventType):
            event_type = event.event_type.value
        else:
            event_type = str(event.event_type)
        
        event_icon = event_icons.get(event_type, TreeSymbols.RUNNING)
        event_color = severity_colors.get(event.severity, ColorCode.WHITE.value if self.config.color_enabled else "")
        reset = ColorCode.RESET.value if self.config.color_enabled else ""
        
        # Format event content
        content = self._format_event_content(event)
        
        if self.config.color_enabled:
            return f"{prefix} {event_icon} {ColorCode.DIM.value}{timestamp}{reset} {event_color}{content}{reset}"
        else:
            return f"{prefix} {event_icon} {timestamp} {content}"
    
    def _format_event_content(self, event: MLEvent) -> str:
        """Formatta contenuto evento in modo conciso"""
        if isinstance(event.event_type, EventType):
            event_type = event.event_type.value
        else:
            event_type = str(event.event_type)
        
        if event_type == "learning_progress":
            progress = event.data.get('progress_percent', 0)
            return f"Learning Progress: {progress:.1f}%"
        elif event_type == "champion_change":
            new_champion = event.data.get('new_champion', 'Unknown')
            model_type = event.data.get('model_type', 'Unknown')
            score = event.data.get('score', 0)
            return f"New Champion [{model_type}]: {new_champion} (score: {score:.1f})"
        elif event_type == "model_training":
            model_name = event.data.get('model_name', 'Unknown')
            status = event.data.get('status', 'Unknown')
            accuracy = event.data.get('accuracy', 0)
            return f"Training [{model_name}]: {status} (acc: {accuracy:.1f}%)"
        elif event_type == "emergency_stop":
            reason = event.data.get('reason', 'Unknown')
            return f"Emergency Stop: {reason}"
        elif event_type == "performance_metrics":
            memory = event.data.get('memory_usage', 0)
            cpu = event.data.get('cpu_usage', 0)
            return f"Performance: Mem {memory:.1f}%, CPU {cpu:.1f}%"
        elif event_type == "validation_complete":
            model = event.data.get('model_name', 'Unknown')
            score = event.data.get('score', 0)
            return f"Validation [{model}]: {score:.2f}"
        elif event_type == "diagnostic":
            message = event.data.get('message', event_type.replace('_', ' ').title())
            return message
        else:
            # Try to get message from data
            message = event.data.get('message', event_type.replace('_', ' ').title())
            return message
    
    def _calculate_avg_accuracy(self, metrics: DisplayMetrics) -> float:
        """Calcola accuratezza media dei modelli"""
        if not metrics.models:
            return 0.0
        
        total_accuracy = sum(model.accuracy for model in metrics.models.values())
        return total_accuracy / len(metrics.models)
    
    def _format_duration(self, seconds: int) -> str:
        """Formatta durata in formato human-readable"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m"
    
    def render_event(self, event: MLEvent) -> str:
        """Renderizza singolo evento per output scroll"""
        timestamp = event.timestamp.strftime("%H:%M:%S")
        content = self._format_event_content(event)
        
        # Event icon
        event_type = event.event_type.value if isinstance(event.event_type, EventType) else str(event.event_type)
        event_icons = {
            "learning_progress": TreeSymbols.DATA,
            "champion_change": TreeSymbols.CHAMPION,
            "model_training": TreeSymbols.TRAINING,
            "emergency_stop": TreeSymbols.ERROR,
            "performance_metrics": TreeSymbols.PERFORMANCE,
            "validation_complete": TreeSymbols.SUCCESS,
            "diagnostic": TreeSymbols.RUNNING,
        }
        
        event_icon = event_icons.get(event_type, TreeSymbols.RUNNING)
        
        if self.config.color_enabled:
            severity_color = {
                EventSeverity.DEBUG: ColorCode.DIM.value,
                EventSeverity.INFO: ColorCode.WHITE.value,
                EventSeverity.WARNING: ColorCode.BRIGHT_YELLOW.value,
                EventSeverity.ERROR: ColorCode.BRIGHT_RED.value,
                EventSeverity.CRITICAL: ColorCode.BRIGHT_RED.value + ColorCode.BOLD.value
            }.get(event.severity, ColorCode.WHITE.value)
            
            return f"[{ColorCode.DIM.value}{timestamp}{ColorCode.RESET.value}] {event_icon} {severity_color}{content}{ColorCode.RESET.value}"
        else:
            return f"[{timestamp}] {event_icon} {content}"


class DisplayManager:
    """
    Manager principale per display degli eventi ML con layout ad albero
    """
    
    def __init__(self, config: MLTrainingLoggerConfig):
        self.config = config
        self.display_config = config.get_display_config()
        
        # Detect terminal capabilities
        self.capabilities = self._detect_terminal_capabilities()
        
        # Resolve AUTO mode to appropriate mode based on capabilities
        if self.display_config.terminal_mode == TerminalMode.AUTO:
            if self.capabilities.supports_cursor_control and self.capabilities.supports_ansi:
                self.display_config.terminal_mode = TerminalMode.DASHBOARD
            else:
                self.display_config.terminal_mode = TerminalMode.SCROLL
        
        # Initialize renderer
        self.renderer = TreeProgressRenderer(self.display_config, self.capabilities)
        
        # Display state
        self.current_metrics = DisplayMetrics()
        self.is_running = False
        
        # Event buffer
        self.recent_events = deque(maxlen=50)  # Keep last 50 events
        
        # Lock for thread safety
        self.display_lock = threading.RLock()
        
        # Timing for periodic updates
        self.last_full_display = datetime.now()
        self.full_display_interval = timedelta(seconds=5)  # More frequent updates for fixed display
        
        # Performance tracking
        self.update_count = 0
        
        # Fixed display state
        self.table_lines_count = 0  # Number of lines used by fixed table
        self.log_area_start = 0     # Line where log area starts
        self.initial_display_shown = False
    
    def _detect_terminal_capabilities(self) -> TerminalCapabilities:
        """Rileva capabilities del terminale"""
        capabilities = TerminalCapabilities()
        
        try:
            size = shutil.get_terminal_size()
            capabilities.width = size.columns
            capabilities.height = size.lines
        except:
            capabilities.width = 100
            capabilities.height = 30
        
        if os.name == 'nt':  # Windows
            capabilities.supports_ansi = os.environ.get('WT_SESSION') is not None or \
                                       os.environ.get('TERM_PROGRAM') == 'vscode' or \
                                       'ANSICON' in os.environ
            capabilities.supports_colors = capabilities.supports_ansi
        else:  # Unix-like systems
            capabilities.supports_ansi = sys.stdout.isatty() and os.environ.get('TERM', '').lower() != 'dumb'
            capabilities.supports_colors = capabilities.supports_ansi and \
                                         os.environ.get('TERM', '').find('color') != -1
        
        capabilities.supports_cursor_control = capabilities.supports_ansi
        
        return capabilities
    
    def _clear_screen(self):
        """Pulisce lo schermo - DISABILITATO per mantenere log persistenti"""
        # NOTA: Comando di clear disabilitato per evitare cancellazione log
        # if self.capabilities.supports_cursor_control:
        #     print("\033[2J\033[H", end="")  # Clear screen and move to top
        #     sys.stdout.flush()
        pass  # Non pulire lo schermo per mantenere i log visibili
    
    def _move_cursor_to(self, line: int, column: int = 1):
        """Muove cursore a posizione specifica"""
        if self.capabilities.supports_cursor_control:
            print(f"\033[{line};{column}H", end="")
            sys.stdout.flush()
    
    def _save_cursor_position(self):
        """Salva posizione corrente del cursore"""
        if self.capabilities.supports_cursor_control:
            print("\033[s", end="")
            sys.stdout.flush()
    
    def _restore_cursor_position(self):
        """Ripristina posizione salvata del cursore"""
        if self.capabilities.supports_cursor_control:
            print("\033[u", end="")
            sys.stdout.flush()
    
    def _clear_from_cursor_to_end(self):
        """Pulisce da cursore alla fine dello schermo"""
        if self.capabilities.supports_cursor_control:
            print("\033[0J", end="")
            sys.stdout.flush()
    
    def _show_initial_fixed_display(self):
        """Mostra display fisso iniziale"""
        with self.display_lock:
            # Render the fixed table
            status_output = self.renderer.render_full_status(self.current_metrics)
            status_lines = status_output.split('\n')
            
            # Print the fixed table
            print(status_output)
            print("=" * 60)
            print("Event Log (live updates):")
            print("-" * 60)
            
            # Calculate where the log area starts
            self.table_lines_count = len(status_lines) + 3  # +3 for separators and log header
            self.log_area_start = self.table_lines_count + 1
            self.initial_display_shown = True
            
            sys.stdout.flush()
    
    def _update_fixed_table(self):
        """Aggiorna solo la tabella fissa senza toccare i log"""
        if not self.capabilities.supports_cursor_control or not self.initial_display_shown:
            return
        
        with self.display_lock:
            # Save current cursor position
            self._save_cursor_position()
            
            # Move to top and render table
            self._move_cursor_to(1, 1)
            status_output = self.renderer.render_full_status(self.current_metrics)
            status_lines = status_output.split('\n')
            
            # Print table lines, clearing any old content
            for i, line in enumerate(status_lines):
                self._move_cursor_to(i + 1, 1)
                print(f"\033[K{line}")  # Clear line and print
            
            # Update separators if needed - DISABLED to fix BIAS model display
            # separator_line = self.table_lines_count - 2
            # self._move_cursor_to(separator_line, 1)
            # print(f"\033[K{'=' * 60}")
            
            # Restore cursor position to continue with logs
            self._restore_cursor_position()
            sys.stdout.flush()
    
    def _handle_terminal_resize(self):
        """Gestisce ridimensionamento del terminale"""
        if not self.capabilities.supports_cursor_control:
            return
        
        try:
            size = shutil.get_terminal_size()
            old_width = self.capabilities.width
            old_height = self.capabilities.height
            
            self.capabilities.width = size.columns
            self.capabilities.height = size.lines
            
            # If size changed significantly, refresh the display
            if (abs(size.columns - old_width) > 5 or 
                abs(size.lines - old_height) > 3):
                self._clear_screen()
                self._show_initial_fixed_display()
        except:
            pass  # Ignore resize errors
    
    def start(self):
        """Avvia display manager"""
        if self.is_running or self.display_config.terminal_mode == TerminalMode.SILENT:
            return
        
        self.is_running = True
        
        # Clear screen and setup fixed display if terminal supports it and in DASHBOARD mode
        if (self.capabilities.supports_cursor_control and 
            self.display_config.terminal_mode == TerminalMode.DASHBOARD):
            self._clear_screen()
            self._show_initial_fixed_display()
        else:
            print(f"{TreeSymbols.SUCCESS} ML Training Logger Display Started")
            print("=" * 60)
    
    def stop(self):
        """Ferma display manager"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Clean shutdown message
        if self.capabilities.supports_cursor_control and self.initial_display_shown:
            print("\n" + "=" * 60)
            print(f"{TreeSymbols.SUCCESS} ML Training Logger Display Stopped")
        else:
            print("\n" + "=" * 60)
            print(f"{TreeSymbols.SUCCESS} ML Training Logger Display Stopped")
    
    def update_metrics(self, **kwargs):
        """Aggiorna metriche display"""
        with self.display_lock:
            for key, value in kwargs.items():
                if hasattr(self.current_metrics, key):
                    setattr(self.current_metrics, key, value)
            
            self.current_metrics.recent_events = list(self.recent_events)
            self.update_count += 1
            
            # Update fixed table if supported, otherwise fallback to periodic full status
            now = datetime.now()
            if self.capabilities.supports_cursor_control and self.initial_display_shown:
                # Update fixed table frequently for real-time display
                if (now - self.last_full_display >= self.full_display_interval or 
                    self.update_count % 1 == 0):  # Real-time updates for tick counter
                    self._update_fixed_table()
                    self.last_full_display = now
            else:
                # Fallback to old behavior for terminals without cursor control
                if (now - self.last_full_display >= self.full_display_interval or 
                    self.update_count % 1000 == 0):
                    self._display_full_status()
                    self.last_full_display = now
    
    def display_event(self, event: MLEvent):
        """Visualizza nuovo evento"""
        if self.display_config.terminal_mode == TerminalMode.SILENT:
            return
        
        with self.display_lock:
            self.recent_events.append(event)
            
            # Format event for display
            event_line = self.renderer.render_event(event)
            
            if self.capabilities.supports_cursor_control and self.initial_display_shown:
                # For fixed display, events scroll normally in the log area
                # No cursor manipulation needed - just print and it will scroll naturally
                print(event_line)
            else:
                # Fallback to traditional display
                print(event_line)
            
            sys.stdout.flush()
    
    def _display_full_status(self):
        """Visualizza status completo"""
        if self.display_config.terminal_mode == TerminalMode.SILENT:
            return
        
        with self.display_lock:
            status_output = self.renderer.render_full_status(self.current_metrics)
            print("\n" + "="*60)
            print(status_output)
            print("="*60)
            sys.stdout.flush()
    
    def force_refresh(self):
        """Forza refresh immediato del display"""
        if self.capabilities.supports_cursor_control and self.initial_display_shown:
            self._update_fixed_table()
        else:
            self._display_full_status()
    
    # Convenience methods for updating specific metrics
    def set_learning_progress(self, progress_percent: float):
        """Aggiorna progresso apprendimento"""
        self.update_metrics(learning_progress=progress_percent)
    
    def set_duration(self, duration_seconds: int):
        """Aggiorna durata sessione"""
        self.update_metrics(duration_seconds=duration_seconds)
    
    def set_ticks_processed(self, tick_count: int):
        """Aggiorna conteggio tick processati"""
        self.update_metrics(ticks_processed=tick_count)
    
    def set_processing_rate(self, rate: float):
        """Aggiorna rate di processing"""
        self.update_metrics(processing_rate=rate)
    
    def set_asset_symbol(self, symbol: str):
        """Aggiorna simbolo asset"""
        self.update_metrics(asset_symbol=symbol)
    
    def set_health_score(self, health_score: float):
        """Aggiorna health score del sistema"""
        self.update_metrics(health_score=health_score)
    
    def update_model_progress(self, model_name: str, progress: Optional[float] = None, 
                            accuracy: Optional[float] = None, status: Optional[str] = None, 
                            predictions: Optional[int] = None, is_champion: Optional[bool] = None):
        """Aggiorna progress di un modello specifico"""
        with self.display_lock:
            if model_name not in self.current_metrics.models:
                self.current_metrics.models[model_name] = ModelProgress(name=model_name)
            
            model = self.current_metrics.models[model_name]
            if progress is not None:
                model.progress = progress
            if accuracy is not None:
                model.accuracy = accuracy
            if status is not None:
                model.status = status
            if predictions is not None:
                model.predictions = predictions
            if is_champion is not None:
                model.is_champion = is_champion
                # Update champions count
                self.current_metrics.champions_active = sum(
                    1 for m in self.current_metrics.models.values() if m.is_champion
                )
            
            model.last_update = datetime.now()
    
    def set_performance_metrics(self, memory_usage: Optional[float] = None, cpu_usage: Optional[float] = None):
        """Aggiorna metriche performance"""
        kwargs = {}
        if memory_usage is not None:
            kwargs['memory_usage'] = memory_usage
        if cpu_usage is not None:
            kwargs['cpu_usage'] = cpu_usage
        
        if kwargs:
            self.update_metrics(**kwargs)
    
    def get_display_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche del display"""
        with self.display_lock:
            return {
                'display_mode': self.display_config.terminal_mode.value,
                'renderer_type': type(self.renderer).__name__,
                'is_running': self.is_running,
                'capabilities': {
                    'supports_ansi': self.capabilities.supports_ansi,
                    'supports_colors': self.capabilities.supports_colors,
                    'terminal_size': f"{self.capabilities.width}x{self.capabilities.height}"
                },
                'metrics': {
                    'update_count': self.update_count,
                    'recent_events_count': len(self.recent_events),
                    'models_tracked': len(self.current_metrics.models),
                    'champions_active': self.current_metrics.champions_active,
                    'learning_progress': self.current_metrics.learning_progress,
                    'ticks_processed': self.current_metrics.ticks_processed
                },
                'fixed_display': {
                    'enabled': self.capabilities.supports_cursor_control,
                    'initial_display_shown': self.initial_display_shown,
                    'table_lines_count': self.table_lines_count,
                    'log_area_start': self.log_area_start
                }
            }
    
    def get_display_mode_info(self) -> str:
        """Ottieni informazioni sulla modalità di display"""
        if self.capabilities.supports_cursor_control and self.initial_display_shown:
            return (f"Fixed Display Mode: Table at top (lines 1-{self.table_lines_count}), "
                   f"scrollable logs from line {self.log_area_start}")
        else:
            return "Scroll Display Mode: Traditional scrolling output"


# Factory functions
def create_tree_display(config: MLTrainingLoggerConfig) -> DisplayManager:
    """Crea display manager con layout ad albero"""
    return DisplayManager(config)


def create_silent_display(config: MLTrainingLoggerConfig) -> DisplayManager:
    """Crea display manager in modalità silent"""
    display_config = config.get_display_config()
    display_config.terminal_mode = TerminalMode.SILENT
    return DisplayManager(config)


# Export main classes
__all__ = [
    'DisplayManager',
    'TreeProgressRenderer',
    'TerminalCapabilities',
    'DisplayMetrics',
    'ModelProgress',
    'TreeSymbols',
    'ProgressBar',
    'ColorCode',
    'create_tree_display',
    'create_silent_display'
]