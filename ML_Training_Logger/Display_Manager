#!/usr/bin/env python3
"""
MLTrainingLogger - Display Manager
==================================

Gestisce la visualizzazione real-time degli eventi ML training nel terminale.
Supporta modalità dashboard (update in-place) e scroll tradizionale.

Author: ScalpingBOT Team
Version: 1.0.0
"""

import os
import sys
import time
import threading
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
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
class DisplayMetrics:
    """Metriche per il display"""
    learning_progress: float = 0.0
    duration_seconds: int = 0
    ticks_processed: int = 0
    champions_active: int = 0
    champions_status: Optional[Dict[str, str]] = None
    performance_metrics: Optional[Dict[str, float]] = None
    recent_events: Optional[List[MLEvent]] = None
    custom_metrics: Optional[Dict[str, Any]] = None  # AGGIUNGI QUESTA RIGA
    
    def __post_init__(self):
        if self.champions_status is None:
            self.champions_status = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.recent_events is None:
            self.recent_events = []
        if self.custom_metrics is None:  # AGGIUNGI QUESTA RIGA
            self.custom_metrics = {}


class ANSIController:
    """Controllo ANSI per manipolazione terminale"""
    
    @staticmethod
    def clear_screen():
        """Pulisce schermo"""
        return "\033[2J"
    
    @staticmethod
    def cursor_home():
        """Sposta cursore in home (0,0)"""
        return "\033[H"
    
    @staticmethod
    def cursor_to(row: int, col: int):
        """Sposta cursore a posizione specifica"""
        return f"\033[{row};{col}H"
    
    @staticmethod
    def clear_line():
        """Pulisce riga corrente"""
        return "\033[K"
    
    @staticmethod
    def clear_to_end():
        """Pulisce da cursore a fine schermo"""
        return "\033[J"
    
    @staticmethod
    def hide_cursor():
        """Nasconde cursore"""
        return "\033[?25l"
    
    @staticmethod
    def show_cursor():
        """Mostra cursore"""
        return "\033[?25h"
    
    @staticmethod
    def save_cursor():
        """Salva posizione cursore"""
        return "\033[s"
    
    @staticmethod
    def restore_cursor():
        """Ripristina posizione cursore"""
        return "\033[u"


class ProgressBar:
    """Barra di progresso ASCII"""
    
    def __init__(self, width: int = 40, fill_char: str = "█", empty_char: str = "░"):
        self.width = width
        self.fill_char = fill_char
        self.empty_char = empty_char
    
    def render(self, progress: float, show_percentage: bool = True) -> str:
        """
        Renderizza barra di progresso
        
        Args:
            progress: Progresso 0.0-1.0
            show_percentage: Mostra percentuale
            
        Returns:
            str: Barra renderizzata
        """
        progress = max(0.0, min(1.0, progress))
        filled = int(self.width * progress)
        empty = self.width - filled
        
        bar = self.fill_char * filled + self.empty_char * empty
        
        if show_percentage:
            percentage = f"{progress * 100:.1f}%"
            return f"[{bar}] {percentage}"
        else:
            return f"[{bar}]"


class DashboardRenderer:
    """Renderer per modalità dashboard"""
    
    def __init__(self, config: DisplaySettings, capabilities: TerminalCapabilities):
        self.config = config
        self.capabilities = capabilities
        self.progress_bar = ProgressBar(width=min(50, capabilities.width - 20))
        
        # Layout configuration
        self.sections = {
            'header': {'start_row': 1, 'height': 2},
            'progress': {'start_row': 4, 'height': self.config.progress_section_height},
            'champions': {'start_row': 8, 'height': self.config.champions_section_height},
            'metrics': {'start_row': 15, 'height': self.config.metrics_section_height},
            'events': {'start_row': 21, 'height': self.config.events_section_height}
        }
    
    def render_dashboard(self, metrics: DisplayMetrics) -> str:
        """
        Renderizza dashboard completo
        
        Args:
            metrics: Metriche da visualizzare
            
        Returns:
            str: Dashboard renderizzato
        """
        output = []
        
        # Clear screen and position cursor
        output.append(ANSIController.clear_screen())
        output.append(ANSIController.cursor_home())
        output.append(ANSIController.hide_cursor())
        
        # Render sections
        output.append(self._render_header(metrics))
        output.append(self._render_progress_section(metrics))
        output.append(self._render_champions_section(metrics))
        
        if self.config.show_performance_metrics:
            output.append(self._render_metrics_section(metrics))
        
        output.append(self._render_events_section(metrics))
        
        # Footer with controls info
        output.append(self._render_footer())
        
        return "".join(output)
    
    def _render_header(self, metrics: DisplayMetrics) -> str:
        """Renderizza header del dashboard"""
        output = []
        
        # Title line
        title = "ML Training Monitor Dashboard"
        if self.config.color_enabled:
            title = f"{ColorCode.BOLD.value}{ColorCode.BRIGHT_CYAN.value}{title}{ColorCode.RESET.value}"
        
        centered_title = title.center(self.capabilities.width)
        output.append(f"{ANSIController.cursor_to(1, 1)}{centered_title}")
        
        # Timestamp and duration line
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration = self._format_duration(metrics.duration_seconds)
        
        info_line = f"Time: {now} | Duration: {duration}"
        if self.config.color_enabled:
            info_line = f"{ColorCode.DIM.value}{info_line}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(2, 1)}{info_line}")
        
        # Separator line
        separator = "─" * self.capabilities.width
        if self.config.color_enabled:
            separator = f"{ColorCode.BLUE.value}{separator}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(3, 1)}{separator}")
        
        return "".join(output)
    
    def _render_progress_section(self, metrics: DisplayMetrics) -> str:
        """Renderizza sezione progresso"""
        output = []
        start_row = self.sections['progress']['start_row']
        
        # Section title
        title = "┌─ Learning Progress ─────────────────────────────────────────┐"
        if self.config.color_enabled:
            title = f"{ColorCode.GREEN.value}{title}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(start_row, 1)}{title}")
        
        # Progress bar
        if self.config.show_progress_bar:
            progress_line = f"│ Progress: {self.progress_bar.render(metrics.learning_progress / 100.0):<55} │"
            output.append(f"{ANSIController.cursor_to(start_row + 1, 1)}{progress_line}")
        
        # Statistics line
        stats_line = f"│ Ticks: {metrics.ticks_processed:,} | Duration: {self._format_duration(metrics.duration_seconds):<20} │"
        output.append(f"{ANSIController.cursor_to(start_row + 2, 1)}{stats_line}")
        
        # Bottom border
        bottom = "└─────────────────────────────────────────────────────────────┘"
        if self.config.color_enabled:
            bottom = f"{ColorCode.GREEN.value}{bottom}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(start_row + 3, 1)}{bottom}")
        
        return "".join(output)
    
    def _render_champions_section(self, metrics: DisplayMetrics) -> str:
        """Renderizza sezione champions"""
        output = []
        start_row = self.sections['champions']['start_row']
        
        # Section title
        title = "┌─ Champions Status ──────────────────────────────────────────┐"
        if self.config.color_enabled:
            title = f"{ColorCode.YELLOW.value}{title}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(start_row, 1)}{title}")
        
        # Champions info
        if metrics.champions_status:
            row = start_row + 1
            for model_type, status in metrics.champions_status.items():
                status_color = self._get_status_color(status)
                status_line = f"│ {model_type:<15}: {status_color}{status:<20}{ColorCode.RESET.value if self.config.color_enabled else ''} │"
                output.append(f"{ANSIController.cursor_to(row, 1)}{status_line}")
                row += 1
            
            # Fill remaining rows
            while row < start_row + self.sections['champions']['height']:
                empty_line = "│" + " " * 61 + "│"
                output.append(f"{ANSIController.cursor_to(row, 1)}{empty_line}")
                row += 1
        else:
            # No champions data
            no_data_line = f"│ {'No champions data available':<61} │"
            output.append(f"{ANSIController.cursor_to(start_row + 1, 1)}{no_data_line}")
            
            # Fill remaining rows
            for i in range(2, self.sections['champions']['height']):
                empty_line = "│" + " " * 61 + "│"
                output.append(f"{ANSIController.cursor_to(start_row + i, 1)}{empty_line}")
        
        # Bottom border
        bottom = "└─────────────────────────────────────────────────────────────┘"
        if self.config.color_enabled:
            bottom = f"{ColorCode.YELLOW.value}{bottom}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(start_row + self.sections['champions']['height'] - 1, 1)}{bottom}")
        
        return "".join(output)
    
    def _render_metrics_section(self, metrics: DisplayMetrics) -> str:
        """Renderizza sezione metriche performance"""
        output = []
        start_row = self.sections['metrics']['start_row']
        
        # Section title
        title = "┌─ Performance Metrics ───────────────────────────────────────┐"
        if self.config.color_enabled:
            title = f"{ColorCode.MAGENTA.value}{title}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(start_row, 1)}{title}")
        
        # Metrics display
        if metrics.performance_metrics:
            row = start_row + 1
            for metric_name, value in metrics.performance_metrics.items():
                formatted_value = self._format_metric_value(metric_name, value)
                metric_line = f"│ {metric_name:<20}: {formatted_value:<38} │"
                output.append(f"{ANSIController.cursor_to(row, 1)}{metric_line}")
                row += 1
            
            # Fill remaining rows
            while row < start_row + self.sections['metrics']['height'] - 1:
                empty_line = "│" + " " * 61 + "│"
                output.append(f"{ANSIController.cursor_to(row, 1)}{empty_line}")
                row += 1
        else:
            no_data_line = f"│ {'No performance data available':<61} │"
            output.append(f"{ANSIController.cursor_to(start_row + 1, 1)}{no_data_line}")
        
        # Bottom border
        bottom = "└─────────────────────────────────────────────────────────────┘"
        if self.config.color_enabled:
            bottom = f"{ColorCode.MAGENTA.value}{bottom}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(start_row + self.sections['metrics']['height'] - 1, 1)}{bottom}")
        
        return "".join(output)
    
    def _render_events_section(self, metrics: DisplayMetrics) -> str:
        """Renderizza sezione eventi recenti"""
        output = []
        start_row = self.sections['events']['start_row']
        
        # Section title
        title = "┌─ Recent Events ─────────────────────────────────────────────┐"
        if self.config.color_enabled:
            title = f"{ColorCode.CYAN.value}{title}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(start_row, 1)}{title}")
        
        # Events display
        if metrics.recent_events:
            row = start_row + 1
            # Show most recent events first
            recent_events = metrics.recent_events[-self.config.max_recent_events:]
            
            for event in reversed(recent_events):
                event_line = self._format_event_line(event)
                padded_line = f"│ {event_line:<61} │"
                output.append(f"{ANSIController.cursor_to(row, 1)}{padded_line}")
                row += 1
                
                if row >= start_row + self.sections['events']['height'] - 1:
                    break
            
            # Fill remaining rows
            while row < start_row + self.sections['events']['height'] - 1:
                empty_line = "│" + " " * 61 + "│"
                output.append(f"{ANSIController.cursor_to(row, 1)}{empty_line}")
                row += 1
        else:
            no_events_line = f"│ {'No recent events':<61} │"
            output.append(f"{ANSIController.cursor_to(start_row + 1, 1)}{no_events_line}")
        
        # Bottom border
        bottom = "└─────────────────────────────────────────────────────────────┘"
        if self.config.color_enabled:
            bottom = f"{ColorCode.CYAN.value}{bottom}{ColorCode.RESET.value}"
        
        output.append(f"{ANSIController.cursor_to(start_row + self.sections['events']['height'] - 1, 1)}{bottom}")
        
        return "".join(output)
    
    def _render_footer(self) -> str:
        """Renderizza footer con info controlli"""
        footer_row = self.capabilities.height
        
        footer_text = "Press Ctrl+C to stop monitoring"
        if self.config.color_enabled:
            footer_text = f"{ColorCode.DIM.value}{footer_text}{ColorCode.RESET.value}"
        
        centered_footer = footer_text.center(self.capabilities.width)
        return f"{ANSIController.cursor_to(footer_row, 1)}{centered_footer}"
    
    def _get_status_color(self, status: str) -> str:
        """Ottieni colore per status"""
        if not self.config.color_enabled:
            return ""
        
        status_lower = status.lower()
        if "champion" in status_lower or "active" in status_lower:
            return ColorCode.BRIGHT_GREEN.value
        elif "training" in status_lower or "learning" in status_lower:
            return ColorCode.BRIGHT_YELLOW.value
        elif "failed" in status_lower or "error" in status_lower:
            return ColorCode.BRIGHT_RED.value
        elif "validating" in status_lower or "testing" in status_lower:
            return ColorCode.BRIGHT_BLUE.value
        else:
            return ColorCode.WHITE.value
    
    def _format_duration(self, seconds: int) -> str:
        """Formatta durata in formato leggibile"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds}s"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{hours}h {remaining_minutes}m"
    
    def _format_metric_value(self, metric_name: str, value: float) -> str:
        """Formatta valore metrica"""
        if "percent" in metric_name.lower() or "%" in metric_name:
            return f"{value:.1f}%"
        elif "mb" in metric_name.lower():
            return f"{value:.1f} MB"
        elif "ms" in metric_name.lower():
            return f"{value:.2f} ms"
        elif "rate" in metric_name.lower():
            return f"{value:.2f}/s"
        else:
            return f"{value:.2f}"
    
    def _format_event_line(self, event: MLEvent) -> str:
        """Formatta riga evento per display"""
        timestamp = event.timestamp.strftime("%H:%M:%S")
        
        # Gestione sicura del tipo evento
        if isinstance(event.event_type, EventType):
            event_type = event.event_type.value
        else:
            event_type = str(event.event_type)
        
        # Severity icon
        severity_icon = self._get_severity_icon(event.severity)
        
        # Format based on event type
        if event_type == "learning_progress":
            progress = event.data.get('progress_percent', 0)
            content = f"Progress: {progress:.1f}%"
        elif event_type == "champion_change":
            new_champion = event.data.get('new_champion', 'Unknown')
            model_type = event.data.get('model_type', 'Unknown')
            content = f"Champion: {model_type} -> {new_champion}"
        elif event_type == "emergency_stop":
            reason = event.data.get('reason', 'Unknown')
            content = f"Emergency Stop: {reason}"
        else:
            content = event_type.replace('_', ' ').title()
        
        # Color based on severity
        if self.config.color_enabled:
            severity_color = self._get_severity_color(event.severity)
            return f"{timestamp} {severity_icon} {severity_color}{content[:45]}{ColorCode.RESET.value}"
        else:
            return f"{timestamp} {severity_icon} {content[:50]}"
    
    def _get_severity_icon(self, severity: EventSeverity) -> str:
        """Ottieni icona per severità"""
        icons = {
            EventSeverity.DEBUG: "🔍",
            EventSeverity.INFO: "ℹ️",
            EventSeverity.WARNING: "⚠️",
            EventSeverity.ERROR: "❌",
            EventSeverity.CRITICAL: "🚨"
        }
        return icons.get(severity, "•")
    
    def _get_severity_color(self, severity: EventSeverity) -> str:
        """Ottieni colore per severità"""
        if not self.config.color_enabled:
            return ""
        
        colors = {
            EventSeverity.DEBUG: ColorCode.DIM.value,
            EventSeverity.INFO: ColorCode.WHITE.value,
            EventSeverity.WARNING: ColorCode.BRIGHT_YELLOW.value,
            EventSeverity.ERROR: ColorCode.BRIGHT_RED.value,
            EventSeverity.CRITICAL: ColorCode.BG_RED.value + ColorCode.WHITE.value
        }
        return colors.get(severity, ColorCode.WHITE.value)


class ScrollRenderer:
    """Renderer per modalità scroll tradizionale"""
    
    def __init__(self, config: DisplaySettings, capabilities: TerminalCapabilities):
        self.config = config
        self.capabilities = capabilities
        self.last_metrics_display = datetime.now()
        self.metrics_display_interval = timedelta(seconds=30)  # Show metrics every 30s
    
    def render_event(self, event: MLEvent) -> str:
        """Renderizza singolo evento per output scroll"""
        timestamp = event.timestamp.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        
        # Gestione sicura del tipo evento
        if isinstance(event.event_type, EventType):
            event_type = event.event_type.value
        else:
            event_type = str(event.event_type)
        
        # Severity prefix
        severity_prefix = self._get_severity_prefix(event.severity)
        
        # Source prefix
        if isinstance(event.source, EventSource):
            source = event.source.value
        else:
            source = str(event.source)
        source_prefix = f"[{source[:10]}]"
        
        # Event content
        content = self._format_event_content(event)
        
        # Color formatting
        if self.config.color_enabled:
            severity_color = self._get_severity_color(event.severity)
            line = f"{ColorCode.DIM.value}{timestamp}{ColorCode.RESET.value} {severity_prefix} {ColorCode.BLUE.value}{source_prefix}{ColorCode.RESET.value} {severity_color}{content}{ColorCode.RESET.value}"
        else:
            line = f"{timestamp} {severity_prefix} {source_prefix} {content}"
        
        return line
    
    def should_display_metrics(self, metrics: DisplayMetrics) -> bool:
        """Verifica se è tempo di mostrare le metriche"""
        now = datetime.now()
        if now - self.last_metrics_display >= self.metrics_display_interval:
            self.last_metrics_display = now
            return True
        return False
    
    def render_metrics_summary(self, metrics: DisplayMetrics) -> str:
        """Renderizza riassunto metriche per modalità scroll"""
        lines = []
        
        # Header
        separator = "=" * 60
        if self.config.color_enabled:
            separator = f"{ColorCode.BRIGHT_BLUE.value}{separator}{ColorCode.RESET.value}"
        
        lines.append(separator)
        
        # Title
        title = "ML Training Status Summary"
        if self.config.color_enabled:
            title = f"{ColorCode.BOLD.value}{ColorCode.BRIGHT_CYAN.value}{title}{ColorCode.RESET.value}"
        
        lines.append(title)
        
        # Metrics
        lines.append(f"Progress: {metrics.learning_progress:.1f}% | Duration: {self._format_duration(metrics.duration_seconds)}")
        lines.append(f"Ticks Processed: {metrics.ticks_processed:,}")
        
        if metrics.champions_status:
            champions_line = "Champions: " + ", ".join([f"{k}={v}" for k, v in metrics.champions_status.items()])
            lines.append(champions_line)
        
        if metrics.performance_metrics:
            perf_items = []
            for name, value in list(metrics.performance_metrics.items())[:3]:  # Show top 3
                formatted_value = self._format_metric_value(name, value)
                perf_items.append(f"{name}={formatted_value}")
            if perf_items:
                lines.append("Performance: " + ", ".join(perf_items))
        
        lines.append(separator)
        
        return "\n".join(lines)
    
    def _format_event_content(self, event: MLEvent) -> str:
        """Formatta contenuto evento per scroll"""
        if isinstance(event.event_type, EventType):
            event_type = event.event_type.value
        else:
            event_type = str(event.event_type)
        
        if event_type == "learning_progress":
            progress = event.data.get('progress_percent', 0)
            return f"Learning Progress: {progress:.1f}%"
        elif event_type == "champion_change":
            old_champion = event.data.get('old_champion', 'Unknown')
            new_champion = event.data.get('new_champion', 'Unknown')
            model_type = event.data.get('model_type', 'Unknown')
            return f"Champion Change [{model_type}]: {old_champion} -> {new_champion}"
        elif event_type == "model_training":
            model_name = event.data.get('model_name', 'Unknown')
            status = event.data.get('status', 'Unknown')
            return f"Model Training [{model_name}]: {status}"
        elif event_type == "emergency_stop":
            reason = event.data.get('reason', 'Unknown')
            return f"Emergency Stop: {reason}"
        elif event_type == "performance_metrics":
            metrics_count = len(event.data)
            return f"Performance Update: {metrics_count} metrics"
        else:
            return event_type.replace('_', ' ').title()
    
    def _get_severity_prefix(self, severity: EventSeverity) -> str:
        """Ottieni prefix per severità"""
        prefixes = {
            EventSeverity.DEBUG: "[DEBUG]",
            EventSeverity.INFO: "[INFO ]",
            EventSeverity.WARNING: "[WARN ]",
            EventSeverity.ERROR: "[ERROR]",
            EventSeverity.CRITICAL: "[CRIT ]"
        }
        return prefixes.get(severity, "[INFO ]")
    
    def _get_severity_color(self, severity: EventSeverity) -> str:
        """Ottieni colore per severità"""
        if not self.config.color_enabled:
            return ""
        
        colors = {
            EventSeverity.DEBUG: ColorCode.DIM.value,
            EventSeverity.INFO: ColorCode.WHITE.value,
            EventSeverity.WARNING: ColorCode.BRIGHT_YELLOW.value,
            EventSeverity.ERROR: ColorCode.BRIGHT_RED.value,
            EventSeverity.CRITICAL: ColorCode.BRIGHT_RED.value + ColorCode.BOLD.value
        }
        return colors.get(severity, ColorCode.WHITE.value)
    
    def _format_duration(self, seconds: int) -> str:
        """Formatta durata"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        else:
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
    
    def _format_metric_value(self, metric_name: str, value: float) -> str:
        """Formatta valore metrica"""
        if "percent" in metric_name.lower():
            return f"{value:.1f}%"
        elif "mb" in metric_name.lower():
            return f"{value:.1f}MB"
        elif "ms" in metric_name.lower():
            return f"{value:.1f}ms"
        else:
            return f"{value:.2f}"


class DisplayManager:
    """
    Manager principale per display degli eventi ML
    """
    
    def __init__(self, config: MLTrainingLoggerConfig):
        self.config = config
        self.display_config = config.get_display_config()
        
        # Detect terminal capabilities
        self.capabilities = self._detect_terminal_capabilities()
        
        # Initialize renderer based on terminal mode
        self.renderer = self._create_renderer()
        
        # Display state
        self.current_metrics = DisplayMetrics()
        self.is_running = False
        self.display_thread = None
        self.stop_event = threading.Event()
        
        # Event buffer for display
        self.recent_events = deque(maxlen=self.display_config.max_recent_events)
        
        # Lock for thread safety
        self.display_lock = threading.RLock()
        
        # Performance tracking
        self.last_update = datetime.now()
        self.update_count = 0
    
    def _detect_terminal_capabilities(self) -> TerminalCapabilities:
        """Rileva capabilities del terminale"""
        capabilities = TerminalCapabilities()
        
        # Terminal size
        try:
            size = shutil.get_terminal_size()
            capabilities.width = size.columns
            capabilities.height = size.lines
        except:
            capabilities.width = 80
            capabilities.height = 24
        
        # ANSI support detection
        if os.name == 'nt':  # Windows
            # Check for Windows Terminal or other modern terminals
            capabilities.supports_ansi = os.environ.get('WT_SESSION') is not None or \
                                       os.environ.get('TERM_PROGRAM') == 'vscode' or \
                                       'ANSICON' in os.environ
            capabilities.supports_colors = capabilities.supports_ansi
        else:  # Unix-like systems
            capabilities.supports_ansi = sys.stdout.isatty() and os.environ.get('TERM', '').lower() != 'dumb'
            capabilities.supports_colors = capabilities.supports_ansi and \
                                         os.environ.get('TERM', '').find('color') != -1
        
        # Cursor control support (usually same as ANSI)
        capabilities.supports_cursor_control = capabilities.supports_ansi
        
        return capabilities
    
    def _create_renderer(self):
        """Crea renderer appropriato basato su configurazione e capabilities"""
        
        # Determine effective terminal mode
        effective_mode = self.display_config.terminal_mode
        
        if effective_mode == TerminalMode.AUTO:
            if self.capabilities.supports_ansi and self.capabilities.supports_cursor_control:
                effective_mode = TerminalMode.DASHBOARD
            else:
                effective_mode = TerminalMode.SCROLL
        
        # Override if terminal doesn't support required features
        if effective_mode == TerminalMode.DASHBOARD and not self.capabilities.supports_cursor_control:
            effective_mode = TerminalMode.SCROLL
        
        # Create appropriate renderer
        if effective_mode == TerminalMode.DASHBOARD:
            return DashboardRenderer(self.display_config, self.capabilities)
        elif effective_mode == TerminalMode.SCROLL:
            return ScrollRenderer(self.display_config, self.capabilities)
        else:  # SILENT mode
            return None
    
    def start(self):
        """Avvia display manager"""
        if self.is_running or self.display_config.terminal_mode == TerminalMode.SILENT:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Start display thread for dashboard mode
        if isinstance(self.renderer, DashboardRenderer):
            self.display_thread = threading.Thread(
                target=self._dashboard_update_loop,
                name="DisplayManager-Dashboard",
                daemon=True
            )
            self.display_thread.start()
        
        # Clear screen for dashboard mode
        if isinstance(self.renderer, DashboardRenderer):
            print(ANSIController.clear_screen(), end='')
            print(ANSIController.cursor_home(), end='')
    
    def stop(self):
        """Ferma display manager"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for display thread
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
        # Restore cursor and clear screen for dashboard mode
        if isinstance(self.renderer, DashboardRenderer):
            print(ANSIController.show_cursor(), end='')
            print(ANSIController.clear_screen(), end='')
            print("Display stopped.\n")
    
    def update_metrics(self, **kwargs):
        """
        Aggiorna metriche display
        
        Args:
            **kwargs: Metriche da aggiornare (learning_progress, duration_seconds, etc.)
        """
        with self.display_lock:
            for key, value in kwargs.items():
                if hasattr(self.current_metrics, key):
                    setattr(self.current_metrics, key, value)
            
            self.current_metrics.recent_events = list(self.recent_events)
            
            # Trigger immediate update for scroll mode
            if isinstance(self.renderer, ScrollRenderer) and self.renderer.should_display_metrics(self.current_metrics):
                self._display_scroll_metrics()
    
    def display_event(self, event: MLEvent):
        """
        Visualizza nuovo evento
        
        Args:
            event: Evento da visualizzare
        """
        if self.display_config.terminal_mode == TerminalMode.SILENT:
            return
        
        with self.display_lock:
            # Add to recent events
            self.recent_events.append(event)
            
            # Display based on mode
            if isinstance(self.renderer, ScrollRenderer):
                event_line = self.renderer.render_event(event)
                print(event_line)
                sys.stdout.flush()
            
            # Dashboard mode updates are handled by the update loop
    
    def _dashboard_update_loop(self):
        """Loop di aggiornamento per modalità dashboard"""
        
        while self.is_running and not self.stop_event.is_set():
            try:
                with self.display_lock:
                    if isinstance(self.renderer, DashboardRenderer):
                        dashboard_output = self.renderer.render_dashboard(self.current_metrics)
                        print(dashboard_output, end='')
                        sys.stdout.flush()
                
                self.update_count += 1
                
                # Wait for next update
                self.stop_event.wait(self.display_config.refresh_rate_seconds)
                
            except Exception as e:
                print(f"\nError in dashboard update loop: {e}")
                time.sleep(1.0)
    
    def _display_scroll_metrics(self):
        """Visualizza metriche in modalità scroll"""
        if isinstance(self.renderer, ScrollRenderer):
            metrics_summary = self.renderer.render_metrics_summary(self.current_metrics)
            print(metrics_summary)
            sys.stdout.flush()
    
    def force_refresh(self):
        """Forza refresh immediato del display"""
        if self.display_config.terminal_mode == TerminalMode.SILENT:
            return
        
        with self.display_lock:
            if isinstance(self.renderer, DashboardRenderer):
                dashboard_output = self.renderer.render_dashboard(self.current_metrics)
                print(dashboard_output, end='')
                sys.stdout.flush()
            elif isinstance(self.renderer, ScrollRenderer):
                self._display_scroll_metrics()
    
    def get_display_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche del display"""
        uptime = (datetime.now() - self.last_update).total_seconds()
        
        return {
            'display_mode': self.display_config.terminal_mode.value,
            'is_running': self.is_running,
            'capabilities': {
                'supports_ansi': self.capabilities.supports_ansi,
                'supports_colors': self.capabilities.supports_colors,
                'terminal_size': f"{self.capabilities.width}x{self.capabilities.height}"
            },
            'metrics': {
                'update_count': self.update_count,
                'uptime_seconds': uptime,
                'recent_events_count': len(self.recent_events),
                'refresh_rate': self.display_config.refresh_rate_seconds
            },
            'current_metrics': {
                'learning_progress': self.current_metrics.learning_progress,
                'duration_seconds': self.current_metrics.duration_seconds,
                'ticks_processed': self.current_metrics.ticks_processed,
                'champions_active': self.current_metrics.champions_active
            }
        }
    
    def set_learning_progress(self, progress_percent: float):
        """Aggiorna progresso apprendimento"""
        self.update_metrics(learning_progress=progress_percent)
    
    def set_duration(self, duration_seconds: int):
        """Aggiorna durata sessione"""
        self.update_metrics(duration_seconds=duration_seconds)
    
    def set_ticks_processed(self, tick_count: int):
        """Aggiorna conteggio tick processati"""
        self.update_metrics(ticks_processed=tick_count)
    
    def set_champions_status(self, champions_status: Dict[str, str]):
        """Aggiorna status champions"""
        self.update_metrics(champions_status=champions_status)
    
    def set_performance_metrics(self, metrics: Dict[str, float]):
        """Aggiorna metriche performance"""
        self.update_metrics(performance_metrics=metrics)
    
    def add_custom_metric(self, name: str, value: Any):
        """Aggiunge metrica personalizzata"""
        # Assicurati che custom_metrics esista
        if not hasattr(self.current_metrics, 'custom_metrics') or self.current_metrics.custom_metrics is None:
            self.current_metrics.custom_metrics = {}
        
        self.current_metrics.custom_metrics[name] = value
    
    def clear_events(self):
        """Pulisce eventi recenti"""
        with self.display_lock:
            self.recent_events.clear()
    
    def resize_terminal(self, width: int, height: int):
        """Gestisce ridimensionamento terminale"""
        self.capabilities.width = width
        self.capabilities.height = height
        
        # Recreate renderer with new dimensions
        if isinstance(self.renderer, DashboardRenderer):
            self.renderer = DashboardRenderer(self.display_config, self.capabilities)


class DisplayEventAdapter:
    """
    Adapter per convertire eventi del sistema esistente in display updates
    """
    
    def __init__(self, display_manager: DisplayManager):
        self.display_manager = display_manager
        self.start_time = datetime.now()
        self.last_tick_count = 0
    
    def handle_analyzer_event(self, event_data: Dict[str, Any]):
        """Gestisce eventi da AdvancedMarketAnalyzer"""
        
        if 'learning_progress' in event_data:
            progress = event_data['learning_progress']
            self.display_manager.set_learning_progress(progress * 100)  # Convert to percentage
        
        if 'tick_count' in event_data:
            self.display_manager.set_ticks_processed(event_data['tick_count'])
            self.last_tick_count = event_data['tick_count']
        
        if 'champions' in event_data:
            champions_status = {}
            for model_type, champion_info in event_data['champions'].items():
                if isinstance(champion_info, dict):
                    status = champion_info.get('status', 'Unknown')
                    champion_name = champion_info.get('name', 'None')
                    champions_status[model_type] = f"{champion_name} ({status})"
                else:
                    champions_status[model_type] = str(champion_info)
            
            self.display_manager.set_champions_status(champions_status)
        
        # Update duration
        duration = (datetime.now() - self.start_time).total_seconds()
        self.display_manager.set_duration(int(duration))
    
    def handle_unified_system_event(self, event_data: Dict[str, Any]):
        """Gestisce eventi da UnifiedAnalyzerSystem"""
        
        if 'performance_metrics' in event_data:
            metrics = event_data['performance_metrics']
            formatted_metrics = {}
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    formatted_metrics[key] = float(value)
            
            self.display_manager.set_performance_metrics(formatted_metrics)
        
        if 'system_status' in event_data:
            status = event_data['system_status']
            
            # Extract useful info from system status
            if 'stats' in status:
                stats = status['stats']
                if 'total_ticks_processed' in stats:
                    self.display_manager.set_ticks_processed(stats['total_ticks_processed'])
    
    def handle_manual_metrics_update(self, **metrics):
        """Gestisce aggiornamento manuale metriche"""
        self.display_manager.update_metrics(**metrics)


# Factory functions per configurazioni comuni
def create_dashboard_display(config: MLTrainingLoggerConfig) -> DisplayManager:
    """Crea display manager in modalità dashboard"""
    display_config = config.get_display_config()
    display_config.terminal_mode = TerminalMode.DASHBOARD
    return DisplayManager(config)


def create_scroll_display(config: MLTrainingLoggerConfig) -> DisplayManager:
    """Crea display manager in modalità scroll"""
    display_config = config.get_display_config()
    display_config.terminal_mode = TerminalMode.SCROLL
    return DisplayManager(config)


def create_silent_display(config: MLTrainingLoggerConfig) -> DisplayManager:
    """Crea display manager in modalità silent"""
    display_config = config.get_display_config()
    display_config.terminal_mode = TerminalMode.SILENT
    return DisplayManager(config)


# Export main classes
__all__ = [
    'DisplayManager',
    'DisplayEventAdapter',
    'DashboardRenderer',
    'ScrollRenderer',
    'TerminalCapabilities',
    'DisplayMetrics',
    'ANSIController',
    'ProgressBar',
    'ColorCode',
    'create_dashboard_display',
    'create_scroll_display',
    'create_silent_display'
]


# Example usage
if __name__ == "__main__":
    from .Config_Manager import create_standard_config
    from .Event_Collector import create_learning_progress_event, create_champion_change_event
    
    # Test display system
    print("Testing ML Training Logger Display System...")
    
    # Create config and display manager
    config = create_standard_config()
    display_manager = DisplayManager(config)
    
    print(f"Terminal capabilities: {display_manager.capabilities.__dict__}")
    print(f"Display mode: {display_manager.display_config.terminal_mode.value}")
    
    # Test event display
    if display_manager.display_config.terminal_mode != TerminalMode.SILENT:
        display_manager.start()
        
        try:
            # Simulate training progress
            for i in range(101):
                # Update metrics
                display_manager.set_learning_progress(i)
                display_manager.set_ticks_processed(i * 1000)
                display_manager.set_duration(i * 2)
                
                # Update champions status
                champions = {
                    'LSTM': 'Champion' if i > 20 else 'Training',
                    'RandomForest': 'Training' if i < 80 else 'Champion',
                    'SVM': 'Failed' if i > 50 else 'Training'
                }
                display_manager.set_champions_status(champions)
                
                # Add some events
                if i % 10 == 0:
                    event = create_learning_progress_event(i, "USTEC", {"iteration": i})
                    display_manager.display_event(event)
                
                if i % 25 == 0 and i > 0:
                    event = create_champion_change_event("OldChamp", "NewChamp", "LSTM", "USTEC")
                    display_manager.display_event(event)
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        
        finally:
            display_manager.stop()
    
    print("✓ Display system test completed")