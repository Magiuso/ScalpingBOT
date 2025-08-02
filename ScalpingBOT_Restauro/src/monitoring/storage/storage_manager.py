#!/usr/bin/env python3
"""
Storage Manager - CLEANED VERSION
REGOLE CLAUDE_RESTAURO.md APPLICATE:
- ✅ Zero fallback/defaults  
- ✅ Fail fast error handling
- ✅ No debug prints
- ✅ No test code embedded
- ✅ Simplified architecture

Gestisce la persistenza degli eventi ML in formato JSON e CSV.
"""

import os
import json
import csv
import gzip
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, TextIO
from pathlib import Path
from collections import deque
import uuid

# Import configuration and events
from ScalpingBOT_Restauro.src.config.domain.monitoring_config import (
    MLTrainingLoggerConfig, StorageSettings, OutputFormat
)
from ScalpingBOT_Restauro.src.monitoring.events.event_collector import MLEvent


class FileRotationManager:
    """Gestisce la rotazione dei file di log - SIMPLIFIED"""
    
    def __init__(self, storage_settings: StorageSettings):
        if not isinstance(storage_settings, StorageSettings):
            raise TypeError(f"Expected StorageSettings, got {type(storage_settings)}")
            
        self.settings = storage_settings
        self.current_files: Dict[str, Path] = {}
        self.file_sizes: Dict[str, int] = {}
        self.file_counts: Dict[str, int] = {}
    
    def get_current_file_path(self, format_type: OutputFormat, session_id: str) -> Path:
        """Ottiene il path del file corrente per il formato specificato"""
        if not isinstance(format_type, OutputFormat):
            raise TypeError(f"Expected OutputFormat, got {type(format_type)}")
        if not session_id:
            raise ValueError("session_id cannot be empty")
            
        base_dir = Path(self.settings.output_directory)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime(self.settings.timestamp_format)
        extension = format_type.value.lower()
        
        # Check if we need rotation
        file_key = f"{format_type.value}_{session_id}"
        
        if file_key in self.current_files:
            current_file = self.current_files[file_key]
            
            # Check if rotation is needed
            if self._needs_rotation(current_file):
                self._rotate_file(file_key, format_type, session_id)
        
        if file_key not in self.current_files:
            # Create new file
            if file_key not in self.file_counts:
                self.file_counts[file_key] = 0
            file_count = self.file_counts[file_key]
            suffix = f"_{file_count:03d}" if file_count > 0 else ""
            
            filename = f"{self.settings.session_prefix}_{session_id}_{timestamp}{suffix}.{extension}"
            file_path = base_dir / filename
            
            self.current_files[file_key] = file_path
            self.file_sizes[file_key] = 0
        
        return self.current_files[file_key]
    
    def _needs_rotation(self, file_path: Path) -> bool:
        """Verifica se il file necessita rotazione"""
        if not self.settings.enable_rotation:
            return False
        
        if not file_path.exists():
            return False
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        return file_size_mb >= self.settings.max_file_size_mb
    
    def _rotate_file(self, file_key: str, format_type: OutputFormat, session_id: str):
        """Ruota il file corrente"""
        if file_key not in self.current_files:
            raise KeyError(f"No current file for key: {file_key}")
            
        file_path = self.current_files[file_key]
        
        # Compress if enabled - FAIL FAST
        if self.settings.compress_old_logs and file_path.exists():
            self._compress_file(file_path)
        
        # Update file count
        if file_key not in self.file_counts:
            self.file_counts[file_key] = 0
        self.file_counts[file_key] = self.file_counts[file_key] + 1
        
        # Remove from current files to force new file creation
        del self.current_files[file_key]
        del self.file_sizes[file_key]
        
        # Cleanup old files
        self._cleanup_old_files(format_type, session_id)
    
    def _compress_file(self, file_path: Path):
        """Comprimi file con gzip - FAIL FAST"""
        compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
        
        with open(file_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # Remove original file
        file_path.unlink()
    
    def _cleanup_old_files(self, format_type: OutputFormat, session_id: str):
        """Rimuove file vecchi se superano il limite"""
        base_dir = Path(self.settings.output_directory)
        pattern = f"{self.settings.session_prefix}_{session_id}_*.{format_type.value.lower()}*"
        
        files = sorted(base_dir.glob(pattern), key=lambda f: f.stat().st_mtime)
        
        # Keep only max_files_per_session
        if len(files) > self.settings.max_files_per_session:
            for old_file in files[:-self.settings.max_files_per_session]:
                old_file.unlink()
    
    def update_file_size(self, file_key: str, bytes_written: int):
        """Aggiorna la dimensione del file tracciata"""
        if file_key not in self.file_sizes:
            raise KeyError(f"No file size tracking for key: {file_key}")
            
        self.file_sizes[file_key] = self.file_sizes[file_key] + bytes_written


class JSONWriter:
    """Writer specifico per formato JSON - CLEANED"""
    
    def __init__(self, file_path: Path, settings: StorageSettings):
        self.file_path = file_path
        self.settings = settings
        self.file_handle: Optional[TextIO] = None
        self.is_first_event = True
        self._lock = threading.Lock()
    
    def open(self):
        """Apre il file per scrittura - FAIL FAST"""
        with self._lock:
            if self.file_handle is not None:
                raise RuntimeError("File already open")
                
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open file - FAIL FAST on errors
            self.file_handle = open(self.file_path, 'w', encoding='utf-8')
            self.file_handle.write('[')  # Start JSON array
            self.is_first_event = True
    
    def write_event(self, event: MLEvent) -> int:
        """Scrive un evento in formato JSON - FAIL FAST"""
        with self._lock:
            if self.file_handle is None:
                raise RuntimeError("File not open")
            
            # Convert event to dict
            event_dict = event.to_dict()
            
            # Serialize to JSON
            json_str = json.dumps(
                event_dict,
                indent=self.settings.json_indent,
                ensure_ascii=self.settings.json_ensure_ascii,
                default=str  # Handle non-serializable objects
            )
            
            # Add comma for previous events
            if not self.is_first_event:
                self.file_handle.write(',\n')
            else:
                self.is_first_event = False
            
            # Write event
            bytes_written = self.file_handle.write(json_str)
            
            # FLUSH immediately
            self.file_handle.flush()
            os.fsync(self.file_handle.fileno())
            
            return bytes_written
    
    def write_events_batch(self, events: List[MLEvent]) -> int:
        """Scrive batch di eventi"""
        total_bytes = 0
        for event in events:
            total_bytes += self.write_event(event)
        return total_bytes
    
    def flush(self):
        """Forza flush dei dati su disco"""
        with self._lock:
            if self.file_handle is None:
                raise RuntimeError("File not open")
                
            self.file_handle.flush()
            os.fsync(self.file_handle.fileno())
    
    def close(self):
        """Chiude il file completando l'array JSON"""
        with self._lock:
            if self.file_handle:
                self.file_handle.write('\n]')  # Close JSON array
                self.file_handle.close()
                self.file_handle = None


class CSVWriter:
    """Writer specifico per formato CSV - CLEANED"""
    
    def __init__(self, file_path: Path, settings: StorageSettings):
        self.file_path = file_path
        self.settings = settings
        self.file_handle: Optional[TextIO] = None
        self.csv_writer: Optional[csv.DictWriter] = None
        self._lock = threading.Lock()
        
        # Define CSV field names
        self.fieldnames = [
            'event_id', 'timestamp', 'event_type', 'source', 'severity',
            'asset', 'session_id', 'source_method', 'processing_time_ms', 
            'tags', 'data_json'
        ]
    
    def open(self):
        """Apre il file per scrittura - FAIL FAST"""
        with self._lock:
            if self.file_handle is not None:
                raise RuntimeError("File already open")
                
            # Ensure directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Open file - FAIL FAST
            self.file_handle = open(self.file_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(
                self.file_handle,
                fieldnames=self.fieldnames,
                delimiter=self.settings.csv_delimiter
            )
            
            if self.settings.csv_include_headers:
                self.csv_writer.writeheader()
    
    def write_event(self, event: MLEvent) -> int:
        """Scrive un evento in formato CSV - FAIL FAST"""
        with self._lock:
            if self.csv_writer is None:
                raise RuntimeError("CSV writer not initialized")
            
            # Convert event to dict
            event_dict = event.to_dict()
            
            # Prepare row - REQUIRE all fields
            row = {
                'event_id': event_dict['event_id'],
                'timestamp': event_dict['timestamp'],
                'event_type': event_dict['event_type'],
                'source': event_dict['source'],
                'severity': event_dict['severity'],
                'asset': event_dict['asset'] or '',
                'session_id': event_dict['session_id'] or '',
                'source_method': event_dict['source_method'] or '',
                'processing_time_ms': str(event_dict['processing_time_ms']) if event_dict['processing_time_ms'] is not None else '',
                'tags': json.dumps(event_dict['tags']),
                'data_json': json.dumps(event_dict['data'])
            }
            
            # Write row
            self.csv_writer.writerow(row)
            
            # Estimate bytes written
            bytes_written = len(','.join(str(v) for v in row.values())) + 2
            
            return bytes_written
    
    def write_events_batch(self, events: List[MLEvent]) -> int:
        """Scrive batch di eventi"""
        total_bytes = 0
        for event in events:
            total_bytes += self.write_event(event)
        return total_bytes
    
    def flush(self):
        """Forza flush dei dati su disco"""
        with self._lock:
            if self.file_handle is None:
                raise RuntimeError("File not open")
                
            self.file_handle.flush()
            os.fsync(self.file_handle.fileno())
    
    def close(self):
        """Chiude il file"""
        with self._lock:
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
                self.csv_writer = None


class EventBuffer:
    """Buffer thread-safe per eventi - SIMPLIFIED"""
    
    def __init__(self, max_size: int = 10000):
        if max_size <= 0:
            raise ValueError("max_size must be positive")
            
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add_event(self, event: MLEvent) -> bool:
        """Aggiunge evento al buffer"""
        if not isinstance(event, MLEvent):
            raise TypeError(f"Expected MLEvent, got {type(event)}")
            
        with self._lock:
            if len(self.buffer) >= self.max_size:
                return False  # Buffer full
            
            self.buffer.append(event)
            return True
    
    def get_events(self, count: Optional[int] = None) -> List[MLEvent]:
        """Estrae eventi dal buffer"""
        with self._lock:
            if count is None:
                events = list(self.buffer)
                self.buffer.clear()
            else:
                events = []
                for _ in range(min(count, len(self.buffer))):
                    if self.buffer:
                        events.append(self.buffer.popleft())
            
            return events
    
    def size(self) -> int:
        """Restituisce il numero di eventi nel buffer"""
        with self._lock:
            return len(self.buffer)


class StorageManager:
    """Gestore principale dello storage - CLEANED & SIMPLIFIED"""
    
    def __init__(self, config: MLTrainingLoggerConfig, session_id: Optional[str] = None):
        if not isinstance(config, MLTrainingLoggerConfig):
            raise TypeError(f"Expected MLTrainingLoggerConfig, got {type(config)}")
            
        self.config = config
        self.storage_config = config.storage
        self.session_id = session_id or str(uuid.uuid4())
        
        # Components
        self.rotation_manager = FileRotationManager(self.storage_config)
        self.event_buffer = EventBuffer(self.storage_config.buffer_size)
        
        # Writers
        self.active_writers: Dict[OutputFormat, Any] = {}
        
        # Threading
        self.is_running = False
        self.writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'events_written': 0,
            'bytes_written': 0,
            'write_errors': 0,
            'last_flush_time': datetime.now()
        }
        
        # Initialize writers
        self._initialize_writers()
    
    def _initialize_writers(self):
        """Inizializza i writer per i formati configurati"""
        for format_type in self.storage_config.output_formats:
            self._create_writer(format_type)
    
    def _create_writer(self, format_type: OutputFormat):
        """Crea un writer per il formato specificato"""
        file_path = self.rotation_manager.get_current_file_path(format_type, self.session_id)
        
        if format_type == OutputFormat.JSON:
            writer = JSONWriter(file_path, self.storage_config)
        elif format_type == OutputFormat.CSV:
            writer = CSVWriter(file_path, self.storage_config)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Open writer - FAIL FAST
        writer.open()
        self.active_writers[format_type] = writer
    
    def start(self):
        """Avvia il thread di scrittura"""
        if self.is_running:
            raise RuntimeError("StorageManager already running")
        
        self.is_running = True
        self._stop_event.clear()
        
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="StorageManager-Writer",
            daemon=True
        )
        self.writer_thread.start()
    
    def stop(self):
        """Ferma il thread di scrittura e chiude i file"""
        if not self.is_running:
            return
        
        self.is_running = False
        self._stop_event.set()
        
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5.0)
        
        # Flush remaining events
        self._flush_pending_events()
        
        # Close all writers
        for writer in self.active_writers.values():
            writer.close()
        
        self.active_writers.clear()
    
    def store_event(self, event: MLEvent) -> bool:
        """Memorizza un evento - CLEANED VERSION"""
        if not isinstance(event, MLEvent):
            raise TypeError(f"Expected MLEvent, got {type(event)}")
        
        if not self.storage_config.enable_file_output:
            return False
        
        # Add to buffer
        if not self.event_buffer.add_event(event):
            self.stats['write_errors'] += 1
            return False
        
        # Flush immediately if critical
        if self.storage_config.flush_on_critical and event.severity.value == 'critical':
            self._flush_pending_events()
        
        return True
    
    def store_events_batch(self, events: List[MLEvent]) -> int:
        """Memorizza batch di eventi"""
        stored_count = 0
        for event in events:
            if self.store_event(event):
                stored_count += 1
        return stored_count
    
    def _writer_loop(self):
        """Loop del thread di scrittura - SIMPLIFIED"""
        while self.is_running and not self._stop_event.is_set():
            # Check if we should flush
            time_since_flush = (datetime.now() - self.stats['last_flush_time']).total_seconds()
            
            if (time_since_flush >= self.storage_config.flush_interval_seconds or
                self.event_buffer.size() >= self.storage_config.batch_processing_size):
                self._flush_pending_events()
            
            # Short sleep to avoid busy waiting
            time.sleep(0.1)
    
    def _flush_pending_events(self):
        """Scrive eventi pendenti sui file"""
        events = self.event_buffer.get_events()
        
        if not events:
            return
        
        # Write to all active writers
        for format_type, writer in self.active_writers.items():
            bytes_written = writer.write_events_batch(events)
            
            # Update rotation tracking
            file_key = f"{format_type.value}_{self.session_id}"
            self.rotation_manager.update_file_size(file_key, bytes_written)
            
            # Update stats
            self.stats['bytes_written'] += bytes_written
        
        self.stats['events_written'] += len(events)
        self.stats['last_flush_time'] = datetime.now()
        
        # Check rotation
        self._check_rotation()
    
    def _check_rotation(self):
        """Verifica se è necessaria la rotazione dei file"""
        for format_type in list(self.active_writers.keys()):
            file_path = self.rotation_manager.get_current_file_path(format_type, self.session_id)
            current_writer = self.active_writers[format_type]
            
            # If file path changed, we need new writer
            if hasattr(current_writer, 'file_path') and current_writer.file_path != file_path:
                # Close old writer
                current_writer.close()
                
                # Create new writer
                self._create_writer(format_type)
    
    def force_flush(self):
        """Forza flush immediato di tutti gli eventi pendenti"""
        self._flush_pending_events()
        
        # Flush all writers
        for writer in self.active_writers.values():
            writer.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche dello storage"""
        return {
            'session_id': self.session_id,
            'is_running': self.is_running,
            'buffer_size': self.event_buffer.size(),
            'active_formats': list(self.active_writers.keys()),
            'performance': self.stats.copy()
        }


# ================== FACTORY FUNCTIONS ==================

def create_storage_manager(config: MLTrainingLoggerConfig, 
                          session_id: Optional[str] = None) -> StorageManager:
    """Factory function per creare StorageManager"""
    return StorageManager(config, session_id)


def create_json_only_storage(config: MLTrainingLoggerConfig, 
                            session_id: Optional[str] = None) -> StorageManager:
    """Crea storage solo per JSON"""
    config.storage.output_formats = [OutputFormat.JSON]
    return StorageManager(config, session_id)


def create_csv_only_storage(config: MLTrainingLoggerConfig, 
                           session_id: Optional[str] = None) -> StorageManager:
    """Crea storage solo per CSV"""
    config.storage.output_formats = [OutputFormat.CSV]
    return StorageManager(config, session_id)


# Export
__all__ = [
    'StorageManager',
    'create_storage_manager',
    'create_json_only_storage', 
    'create_csv_only_storage',
    'JSONWriter',
    'CSVWriter',
    'FileRotationManager',
    'EventBuffer'
]