#!/usr/bin/env python3
"""
MLTrainingLogger - Storage Manager
==================================

Gestisce la persistenza degli eventi ML in formato JSON e CSV.
Supporta rotazione file, compression, e flush configurabile.

Author: ScalpingBOT Team
Version: 1.0.0
"""

import os
import json
import csv
import gzip
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, TextIO
from pathlib import Path
from collections import deque
import uuid

# Import configuration and events
from .Config_Manager import (
    MLTrainingLoggerConfig, StorageSettings, OutputFormat
)
from .Event_Collector import MLEvent, EventType, EventSource, EventSeverity


class FileRotationManager:
    """Gestisce la rotazione dei file di log"""
    
    def __init__(self, storage_settings: StorageSettings):
        self.settings = storage_settings
        self.current_files: Dict[str, Path] = {}
        self.file_sizes: Dict[str, int] = {}
        self.file_counts: Dict[str, int] = {}
    
    def get_current_file_path(self, format_type: OutputFormat, session_id: str) -> Path:
        """
        Ottiene il path del file corrente per il formato specificato
        
        Args:
            format_type: Formato file (JSON/CSV)
            session_id: ID sessione
            
        Returns:
            Path: Path del file corrente
        """
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
            file_count = self.file_counts.get(file_key, 0)
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
        if file_size_mb >= self.settings.max_file_size_mb:
            return True
        
        return False
    
    def _rotate_file(self, file_key: str, format_type: OutputFormat, session_id: str):
        """Ruota il file corrente"""
        if file_key not in self.current_files:
            return
        
        old_file = self.current_files[file_key]
        
        # Compress old file if enabled
        if self.settings.compress_old_logs and old_file.exists():
            self._compress_file(old_file)
        
        # Increment file count
        self.file_counts[file_key] = self.file_counts.get(file_key, 0) + 1
        
        # Remove from current files to force new file creation
        del self.current_files[file_key]
        self.file_sizes[file_key] = 0
        
        # Cleanup old files if limit exceeded
        self._cleanup_old_files(format_type, session_id)
    
    def _compress_file(self, file_path: Path):
        """Comprime un file usando gzip"""
        try:
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            file_path.unlink()
            
        except Exception as e:
            print(f"Failed to compress {file_path}: {e}")
    
    def _cleanup_old_files(self, format_type: OutputFormat, session_id: str):
        """Rimuove file vecchi se superano il limite"""
        if self.settings.max_files_per_session <= 0:
            return
        
        base_dir = Path(self.settings.output_directory)
        extension = format_type.value.lower()
        
        # Find all files for this session and format
        pattern = f"{self.settings.session_prefix}_{session_id}_*.{extension}*"
        files = list(base_dir.glob(pattern))
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda f: f.stat().st_mtime)
        
        # Remove excess files
        while len(files) > self.settings.max_files_per_session:
            old_file = files.pop(0)
            try:
                old_file.unlink()
            except Exception as e:
                print(f"Failed to remove old file {old_file}: {e}")
    
    def update_file_size(self, format_type: OutputFormat, session_id: str, bytes_written: int):
        """Aggiorna il conteggio delle dimensioni del file"""
        file_key = f"{format_type.value}_{session_id}"
        self.file_sizes[file_key] = self.file_sizes.get(file_key, 0) + bytes_written


class JSONWriter:
    """Writer specifico per formato JSON"""
    
    def __init__(self, file_path: Path, settings: StorageSettings):
        self.file_path = file_path
        self.settings = settings
        self.file_handle: Optional[Any] = None  # ← SOLUZIONE: usa Any invece di TextIO
        self.is_first_event = True
        self._lock = threading.Lock()
    
    def open(self):
        """Apre il file per scrittura"""
        with self._lock:
            if self.file_handle is None:
                try:
                    # Assicura che la directory esista
                    self.file_path.parent.mkdir(parents=True, exist_ok=True)
                    self.file_handle = open(self.file_path, 'w', encoding='utf-8')
                    self.file_handle.write('[\n')  # Start JSON array
                    self.is_first_event = True
                except Exception as e:
                    print(f"Failed to open JSON file {self.file_path}: {e}")
                    self.file_handle = None
    
    def write_event(self, event: MLEvent) -> int:
        """
        Scrive un evento in formato JSON
        
        Args:
            event: Evento da scrivere
            
        Returns:
            int: Numero di bytes scritti
        """
        with self._lock:
            if self.file_handle is None:
                self.open()
            
            # Controllo di sicurezza
            if self.file_handle is None:
                return 0
            
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
            
            # FLUSH immediato per garantire scrittura
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
            if self.file_handle:
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
    """Writer specifico per formato CSV"""
    
    def __init__(self, file_path: Path, settings: StorageSettings):
        self.file_path = file_path
        self.settings = settings
        self.file_handle: Optional[Any] = None  # ← SOLUZIONE
        self.csv_writer: Optional[Any] = None  # ← SOLUZIONE
        self.headers_written = False
        self.is_first_event = True
        self._lock = threading.Lock()
        
        # Define CSV field names
        self.fieldnames = [
            'event_id', 'timestamp', 'event_type', 'source', 'severity',
            'asset', 'session_id', 'source_method', 'source_object_id',
            'processing_time_ms', 'tags', 'data_json'
        ]
    
    def open(self):
        """Apre il file per scrittura"""
        with self._lock:
            if self.file_handle is None:
                try:
                    # Assicura che la directory esista
                    self.file_path.parent.mkdir(parents=True, exist_ok=True)
                    self.file_handle = open(self.file_path, 'w', newline='', encoding='utf-8')
                    self.csv_writer = csv.DictWriter(
                        self.file_handle,
                        fieldnames=self.fieldnames,
                        delimiter=self.settings.csv_delimiter
                    )
                    
                    if self.settings.csv_include_headers:
                        self.csv_writer.writeheader()
                        self.headers_written = True
                except Exception as e:
                    print(f"Failed to open CSV file {self.file_path}: {e}")
                    self.file_handle = None
                    self.csv_writer = None
    
    def write_event(self, event: MLEvent) -> int:
        """
        Scrive un evento in formato CSV
        
        Args:
            event: Evento da scrivere
            
        Returns:
            int: Numero di bytes scritti
        """
        with self._lock:
            if self.file_handle is None:
                self.open()
            
            # Controllo di sicurezza
            if self.file_handle is None or self.csv_writer is None:
                return 0
            
            # Convert event to dict
            event_dict = event.to_dict()
            
            # Prepare CSV row
            csv_row = {
                'event_id': event_dict.get('event_id', ''),
                'timestamp': event_dict.get('timestamp', ''),
                'event_type': event_dict.get('event_type', ''),
                'source': event_dict.get('source', ''),
                'severity': event_dict.get('severity', ''),
                'asset': event_dict.get('asset', ''),
                'session_id': event_dict.get('session_id', ''),
                'source_method': event_dict.get('source_method', ''),
                'source_object_id': event_dict.get('source_object_id', ''),
                'processing_time_ms': event_dict.get('processing_time_ms', ''),
                'tags': json.dumps(event_dict.get('tags', [])),
                'data_json': json.dumps(event_dict.get('data', {}))
            }
            
            # Write CSV row and calculate bytes written
            initial_pos = self.file_handle.tell()
            self.csv_writer.writerow(csv_row)
            final_pos = self.file_handle.tell()
            
            # FLUSH immediato per garantire scrittura
            self.file_handle.flush()
            os.fsync(self.file_handle.fileno())
            
            return final_pos - initial_pos
    
    def write_events_batch(self, events: List[MLEvent]) -> int:
        """Scrive batch di eventi"""
        total_bytes = 0
        for event in events:
            total_bytes += self.write_event(event)
        return total_bytes
    
    def flush(self):
        """Forza flush dei dati su disco"""
        with self._lock:
            if self.file_handle:
                self.file_handle.flush()
                os.fsync(self.file_handle.fileno())
    
    def close(self):
        """Chiude il file"""
        with self._lock:
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None


class EventBuffer:
    """Buffer thread-safe per eventi in attesa di scrittura"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer: deque = deque()
        self.lock = threading.RLock()
        self.overflow_count = 0
        self.total_events_buffered = 0
    
    def add_event(self, event: MLEvent) -> bool:
        """
        Aggiunge evento al buffer
        
        Args:
            event: Evento da bufferizzare
            
        Returns:
            bool: True se aggiunto, False se buffer pieno
        """
        with self.lock:
            if len(self.buffer) >= self.max_size:
                self.overflow_count += 1
                return False
            
            self.buffer.append(event)
            self.total_events_buffered += 1
            return True
    
    def get_events(self, max_count: Optional[int] = None) -> List[MLEvent]:
        """
        Estrae eventi dal buffer
        
        Args:
            max_count: Numero massimo di eventi da estrarre
            
        Returns:
            List[MLEvent]: Eventi estratti
        """
        with self.lock:
            if max_count is None:
                events = list(self.buffer)
                self.buffer.clear()
            else:
                events = []
                for _ in range(min(max_count, len(self.buffer))):
                    if self.buffer:
                        events.append(self.buffer.popleft())
            
            return events
    
    def peek_events(self, count: int = 10) -> List[MLEvent]:
        """Visualizza eventi senza rimuoverli"""
        with self.lock:
            return list(self.buffer)[:count]
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche del buffer"""
        with self.lock:
            return {
                'current_size': len(self.buffer),
                'max_size': self.max_size,
                'utilization_percent': (len(self.buffer) / self.max_size) * 100,
                'overflow_count': self.overflow_count,
                'total_events_buffered': self.total_events_buffered,
                'is_full': len(self.buffer) >= self.max_size
            }
    
    def clear(self):
        """Svuota il buffer"""
        with self.lock:
            self.buffer.clear()


class StorageManager:
    """
    Manager principale per storage degli eventi ML
    """
    
    def __init__(self, config: MLTrainingLoggerConfig, session_id: Optional[str] = None):
        self.config = config
        self.storage_config = config.get_storage_config()
        self.session_id = session_id or str(uuid.uuid4())
        
        # File management
        self.rotation_manager = FileRotationManager(self.storage_config)
        self.active_writers: Dict[OutputFormat, Union[JSONWriter, CSVWriter]] = {}
        
        # Buffering
        self.event_buffer = EventBuffer(self.storage_config.buffer_size)
        
        # Threading for async writing
        self.is_running = False
        self.writer_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_events_written': 0,
            'total_bytes_written': 0,
            'events_by_format': {fmt.value: 0 for fmt in OutputFormat},
            'bytes_by_format': {fmt.value: 0 for fmt in OutputFormat},
            'write_errors': 0,
            'last_flush_time': None,
            'storage_start_time': datetime.now(),
            'files_created': 0,
            'files_rotated': 0
        }
        
        # Last flush tracking
        self.last_flush_time = datetime.now()
        self.pending_events_count = 0
        
        # Initialize writers
        self._initialize_writers()
    
    def _initialize_writers(self):
        """Inizializza i writer per i formati abilitati"""
        if not self.storage_config.enable_file_output:
            return
        
        for format_type in self.storage_config.output_formats:
            file_path = self.rotation_manager.get_current_file_path(format_type, self.session_id)
            
            if format_type == OutputFormat.JSON:
                writer = JSONWriter(file_path, self.storage_config)
            elif format_type == OutputFormat.CSV:
                writer = CSVWriter(file_path, self.storage_config)
            else:
                continue
            
            self.active_writers[format_type] = writer
            self.stats['files_created'] += 1
    
    def start(self):
        """Avvia il storage manager"""
        if self.is_running or not self.storage_config.enable_file_output:
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        # Open all writers
        for writer in self.active_writers.values():
            try:
                writer.open()
            except Exception as e:
                print(f"Failed to open writer: {e}")
        
        # Start async writer thread
        self.writer_thread = threading.Thread(
            target=self._writer_loop,
            name="StorageManager-Writer",
            daemon=True
        )
        self.writer_thread.start()
    
    def stop(self):
        """Ferma il storage manager"""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for writer thread
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5.0)
        
        # Final flush
        self._flush_pending_events()
        
        # Close all writers
        for writer in self.active_writers.values():
            try:
                writer.close()
            except Exception as e:
                print(f"Error closing writer: {e}")
    
    def store_event(self, event: MLEvent) -> None:
        """
        Memorizza un evento
        
        Args:
            event: Evento da memorizzare
        """
        print(f"🔥 DEBUG: StorageManager.store_event called with event_type={event.event_type}, severity={event.severity}")
        print(f"🔥 DEBUG: enable_file_output={self.storage_config.enable_file_output}")
        
        if not self.storage_config.enable_file_output:
            print(f"🔥 DEBUG: File output disabled")
            return
        
        # Add to buffer
        print(f"🔥 DEBUG: Adding event to buffer...")
        if self.event_buffer.add_event(event):
            self.pending_events_count += 1
            print(f"🔥 DEBUG: Event added to buffer, pending_events_count={self.pending_events_count}")
            
            # Check for critical events that need immediate flush
            if (self.storage_config.flush_on_critical and 
                event.severity == EventSeverity.CRITICAL):
                print(f"🔥 DEBUG: Critical event detected, flushing immediately...")
                self._flush_pending_events()
            else:
                print(f"🔥 DEBUG: Non-critical event, will flush later...")
        else:
            print(f"🔥 DEBUG: Failed to add event to buffer!")
            self.stats['write_errors'] += 1
    
    def store_events_batch(self, events: List[MLEvent]) -> int:
        """
        Memorizza batch di eventi
        
        Args:
            events: Lista di eventi da memorizzare
            
        Returns:
            int: Numero di eventi memorizzati con successo
        """
        if not self.storage_config.enable_file_output:
            return len(events)
        
        stored_count = 0
        for event in events:
            if self.store_event(event):
                stored_count += 1
        
        return stored_count
    
    def _writer_loop(self):
        """Loop di scrittura asincrono"""
        print(f"🔥 DEBUG: Writer loop STARTED! flush_interval={self.storage_config.flush_interval_seconds}s, is_running={self.is_running}")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                # Check if it's time to flush
                now = datetime.now()
                time_since_flush = (now - self.last_flush_time).total_seconds()
                
                should_flush = (
                    time_since_flush >= self.storage_config.flush_interval_seconds or
                    self.pending_events_count >= self.storage_config.buffer_size // 2
                )
                
                if should_flush and self.pending_events_count > 0:
                    print(f"🔥 DEBUG: Should flush! time_since_flush={time_since_flush:.1f}s, pending_events={self.pending_events_count}")
                    self._flush_pending_events()
                
                # Sleep briefly to avoid busy waiting
                self.stop_event.wait(0.1)
                
            except Exception as e:
                print(f"Error in writer loop: {e}")
                self.stats['write_errors'] += 1
                time.sleep(1.0)
        
        print(f"🔥 DEBUG: Writer loop STOPPED! is_running={self.is_running}")

    def _flush_pending_events(self):
        """Flush eventi in attesa di scrittura"""
        print(f"🔥 DEBUG: _flush_pending_events called, pending_events_count={self.pending_events_count}")
        
        if self.pending_events_count == 0:
            print(f"🔥 DEBUG: No pending events to flush")
            return
        
        try:
            # Get events from buffer
            events = self.event_buffer.get_events()
            print(f"🔥 DEBUG: Got {len(events)} events from buffer")
            
            if not events:
                print(f"🔥 DEBUG: Event buffer returned empty list")
                return
            
            # Write to all active formats
            print(f"🔥 DEBUG: Writing to {len(self.active_writers)} active writers")
            for format_type, writer in self.active_writers.items():
                try:
                    print(f"🔥 DEBUG: Writing {len(events)} events to {format_type.value} writer")
                    bytes_written = writer.write_events_batch(events)
                    writer.flush()
                    print(f"🔥 DEBUG: Wrote {bytes_written} bytes to {format_type.value}")
                    
                    # Update statistics
                    self.stats['events_by_format'][format_type.value] += len(events)
                    self.stats['bytes_by_format'][format_type.value] += bytes_written
                    self.stats['total_bytes_written'] += bytes_written
                    
                    # Update rotation manager
                    self.rotation_manager.update_file_size(format_type, self.session_id, bytes_written)
                    
                except Exception as e:
                    print(f"❌ ERROR writing to {format_type.value}: {e}")
                    import traceback
                    traceback.print_exc()
                    self.stats['write_errors'] += 1
            
            # Update global statistics
            self.stats['total_events_written'] += len(events)
            self.last_flush_time = datetime.now()
            self.stats['last_flush_time'] = self.last_flush_time
            self.pending_events_count = 0
            
        except Exception as e:
            print(f"Error in flush pending events: {e}")
            self.stats['write_errors'] += 1
    
    def force_flush(self):
        """Forza flush immediato di tutti gli eventi in buffer"""
        if self.pending_events_count > 0:
            self._flush_pending_events()
    
    def rotate_files(self):
        """Forza rotazione di tutti i file attivi"""
        for format_type in self.active_writers.keys():
            self.rotation_manager._rotate_file(
                f"{format_type.value}_{self.session_id}",
                format_type,
                self.session_id
            )
            self.stats['files_rotated'] += 1
        
        # Reinitialize writers with new files
        self._reinitialize_writers()
    
    def _reinitialize_writers(self):
        """Reinizializza i writer dopo rotazione"""
        # Close current writers
        for writer in self.active_writers.values():
            try:
                writer.close()
            except Exception as e:
                print(f"Error closing writer during rotation: {e}")
        
        # Clear and reinitialize
        self.active_writers.clear()
        self._initialize_writers()
        
        # Reopen writers if running
        if self.is_running:
            for writer in self.active_writers.values():
                try:
                    writer.open()
                except Exception as e:
                    print(f"Error reopening writer after rotation: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche complete del storage"""
        uptime = (datetime.now() - self.stats['storage_start_time']).total_seconds()
        
        # Buffer stats
        buffer_stats = self.event_buffer.get_stats()
        
        # File stats
        file_stats = {}
        for format_type in self.storage_config.output_formats:
            file_path = self.rotation_manager.get_current_file_path(format_type, self.session_id)
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                file_stats[format_type.value] = {
                    'current_file': str(file_path),
                    'size_mb': round(file_size_mb, 2),
                    'exists': True
                }
            else:
                file_stats[format_type.value] = {
                    'current_file': str(file_path),
                    'size_mb': 0,
                    'exists': False
                }
        
        # Calculate rates
        events_per_second = self.stats['total_events_written'] / max(uptime, 1)
        mb_per_second = (self.stats['total_bytes_written'] / (1024 * 1024)) / max(uptime, 1)
        
        return {
            'storage_manager': {
                'is_running': self.is_running,
                'session_id': self.session_id,
                'uptime_seconds': uptime,
                'output_formats': [fmt.value for fmt in self.storage_config.output_formats],
                'pending_events': self.pending_events_count
            },
            'performance': {
                'events_per_second': round(events_per_second, 2),
                'mb_per_second': round(mb_per_second, 4),
                'events_written': self.stats['total_events_written'],
                'bytes_written': self.stats['total_bytes_written'],
                'write_errors': self.stats['write_errors']
            },
            'by_format': {
                format_name: {
                    'events_written': self.stats['events_by_format'][format_name],
                    'bytes_written': self.stats['bytes_by_format'][format_name]
                }
                for format_name in self.stats['events_by_format'].keys()
            },
            'files': file_stats,
            'buffer': buffer_stats,
            'rotation': {
                'files_created': self.stats['files_created'],
                'files_rotated': self.stats['files_rotated'],
                'rotation_enabled': self.storage_config.enable_rotation,
                'max_file_size_mb': self.storage_config.max_file_size_mb
            }
        }
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """Ottieni eventi recenti dal buffer"""
        recent_events = self.event_buffer.peek_events(count)
        return [event.to_dict() for event in recent_events]
    
    def clear_buffer(self):
        """Svuota il buffer degli eventi"""
        cleared_events = len(self.event_buffer.buffer)
        self.event_buffer.clear()
        self.pending_events_count = 0
        return cleared_events
    
    def change_output_directory(self, new_directory: str):
        """Cambia directory di output"""
        if self.is_running:
            raise RuntimeError("Cannot change directory while storage manager is running")
        
        self.storage_config.output_directory = new_directory
        self.rotation_manager = FileRotationManager(self.storage_config)
        self._initialize_writers()
    
    def add_output_format(self, format_type: OutputFormat):
        """Aggiunge nuovo formato di output"""
        if format_type not in self.storage_config.output_formats:
            self.storage_config.output_formats.append(format_type)
            
            # Initialize writer for new format
            file_path = self.rotation_manager.get_current_file_path(format_type, self.session_id)
            
            if format_type == OutputFormat.JSON:
                writer = JSONWriter(file_path, self.storage_config)
            elif format_type == OutputFormat.CSV:
                writer = CSVWriter(file_path, self.storage_config)
            else:
                return False
            
            self.active_writers[format_type] = writer
            self.stats['files_created'] += 1
            
            # Open writer if storage is running
            if self.is_running:
                try:
                    writer.open()
                except Exception as e:
                    print(f"Error opening new writer for {format_type.value}: {e}")
                    return False
            
            return True
        
        return False
    
    def remove_output_format(self, format_type: OutputFormat):
        """Rimuove formato di output"""
        if format_type in self.storage_config.output_formats:
            self.storage_config.output_formats.remove(format_type)
            
            # Close and remove writer
            if format_type in self.active_writers:
                try:
                    self.active_writers[format_type].close()
                except Exception as e:
                    print(f"Error closing writer for {format_type.value}: {e}")
                
                del self.active_writers[format_type]
            
            return True
        
        return False
    
    def create_session_summary(self) -> Dict[str, Any]:
        """Crea riassunto della sessione di storage"""
        stats = self.get_storage_stats()
        
        # Calculate session duration
        duration = (datetime.now() - self.stats['storage_start_time']).total_seconds()
        
        # Get file information
        files_info = []
        for format_type in self.storage_config.output_formats:
            file_path = self.rotation_manager.get_current_file_path(format_type, self.session_id)
            if file_path.exists():
                files_info.append({
                    'format': format_type.value,
                    'path': str(file_path),
                    'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                    'events_count': self.stats['events_by_format'][format_type.value]
                })
        
        return {
            'session_info': {
                'session_id': self.session_id,
                'start_time': self.stats['storage_start_time'].isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': round(duration, 2)
            },
            'summary_stats': {
                'total_events_stored': self.stats['total_events_written'],
                'total_size_mb': round(self.stats['total_bytes_written'] / (1024 * 1024), 2),
                'average_event_size_bytes': round(
                    self.stats['total_bytes_written'] / max(self.stats['total_events_written'], 1), 2
                ),
                'files_created': self.stats['files_created'],
                'files_rotated': self.stats['files_rotated'],
                'write_errors': self.stats['write_errors']
            },
            'files_created': files_info,
            'performance_summary': {
                'avg_events_per_second': round(self.stats['total_events_written'] / max(duration, 1), 2),
                'avg_mb_per_second': round(
                    (self.stats['total_bytes_written'] / (1024 * 1024)) / max(duration, 1), 4
                ),
                'buffer_overflow_count': self.event_buffer.overflow_count,
                'error_rate_percent': round(
                    (self.stats['write_errors'] / max(self.stats['total_events_written'], 1)) * 100, 2
                )
            }
        }


class StorageEventAdapter:
    """
    Adapter per integrazione con EventCollector
    """
    
    def __init__(self, storage_manager: StorageManager):
        self.storage_manager = storage_manager
        self.events_processed = 0
        self.events_failed = 0
    
    def handle_event(self, event: MLEvent):
        """
        Gestisce evento proveniente da EventCollector
        
        Args:
            event: Evento da memorizzare
        """
        try:
            if self.storage_manager.store_event(event):
                self.events_processed += 1
            else:
                self.events_failed += 1
        except Exception as e:
            print(f"Error handling event in storage adapter: {e}")
            self.events_failed += 1
    
    def handle_events_batch(self, events: List[MLEvent]):
        """Gestisce batch di eventi"""
        try:
            stored_count = self.storage_manager.store_events_batch(events)
            self.events_processed += stored_count
            self.events_failed += len(events) - stored_count
        except Exception as e:
            print(f"Error handling events batch in storage adapter: {e}")
            self.events_failed += len(events)
    
    def get_adapter_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche dell'adapter"""
        total_events = self.events_processed + self.events_failed
        success_rate = (self.events_processed / max(total_events, 1)) * 100
        
        return {
            'events_processed': self.events_processed,
            'events_failed': self.events_failed,
            'total_events': total_events,
            'success_rate_percent': round(success_rate, 2)
        }


# Factory functions per configurazioni comuni
def create_json_only_storage(config: MLTrainingLoggerConfig, session_id: Optional[str] = None) -> StorageManager:
    """Crea storage manager solo JSON"""
    storage_config = config.get_storage_config()
    storage_config.output_formats = [OutputFormat.JSON]
    return StorageManager(config, session_id)


def create_csv_only_storage(config: MLTrainingLoggerConfig, session_id: Optional[str] = None) -> StorageManager:
    """Crea storage manager solo CSV"""
    storage_config = config.get_storage_config()
    storage_config.output_formats = [OutputFormat.CSV]
    return StorageManager(config, session_id)


def create_dual_format_storage(config: MLTrainingLoggerConfig, session_id: Optional[str] = None) -> StorageManager:
    """Crea storage manager JSON + CSV"""
    storage_config = config.get_storage_config()
    storage_config.output_formats = [OutputFormat.JSON, OutputFormat.CSV]
    return StorageManager(config, session_id)


def create_high_performance_storage(config: MLTrainingLoggerConfig, session_id: Optional[str] = None) -> StorageManager:
    """Crea storage manager ottimizzato per performance"""
    storage_config = config.get_storage_config()
    storage_config.buffer_size = 5000
    storage_config.flush_interval_seconds = 10.0
    storage_config.enable_rotation = True
    storage_config.max_file_size_mb = 50
    storage_config.compress_old_logs = True
    return StorageManager(config, session_id)


# Export main classes
__all__ = [
    'StorageManager',
    'StorageEventAdapter',
    'FileRotationManager',
    'JSONWriter',
    'CSVWriter',
    'EventBuffer',
    'create_json_only_storage',
    'create_csv_only_storage',
    'create_dual_format_storage',
    'create_high_performance_storage'
]


# Example usage and testing
if __name__ == "__main__":
    from .Config_Manager import create_standard_config
    from .Event_Collector import create_learning_progress_event, create_champion_change_event
    
    # Test storage system
    print("Testing ML Training Logger Storage System...")
    
    # Create config and storage manager
    config = create_standard_config()
    storage_manager = StorageManager(config)
    
    print(f"Storage configuration: {config.get_storage_config().__dict__}")
    print(f"Output formats: {[fmt.value for fmt in config.get_storage_config().output_formats]}")
    
    # Test event storage
    storage_manager.start()
    
    try:
        # Create test events
        test_events = []
        
        for i in range(100):
            # Learning progress events
            if i % 10 == 0:
                event = create_learning_progress_event(i, "USTEC", {"iteration": i, "loss": 0.5 - i * 0.001})
                test_events.append(event)
            
            # Champion change events
            if i % 25 == 0 and i > 0:
                event = create_champion_change_event("OldChamp", f"NewChamp_{i}", "LSTM", "USTEC")
                test_events.append(event)
        
        # Store events
        print(f"Storing {len(test_events)} test events...")
        stored_count = storage_manager.store_events_batch(test_events)
        print(f"Stored {stored_count} events successfully")
        
        # Force flush
        storage_manager.force_flush()
        
        # Get statistics
        stats = storage_manager.get_storage_stats()
        print("\nStorage Statistics:")
        print(f"  Events written: {stats['performance']['events_written']}")
        print(f"  Bytes written: {stats['performance']['bytes_written']}")
        print(f"  Write errors: {stats['performance']['write_errors']}")
        
        # Show files created
        print("\nFiles created:")
        for format_name, file_info in stats['files'].items():
            if file_info['exists']:
                print(f"  {format_name}: {file_info['current_file']} ({file_info['size_mb']} MB)")
        
        # Test session summary
        session_summary = storage_manager.create_session_summary()
        print(f"\nSession Summary:")
        print(f"  Duration: {session_summary['session_info']['duration_seconds']} seconds")
        print(f"  Total events: {session_summary['summary_stats']['total_events_stored']}")
        print(f"  Total size: {session_summary['summary_stats']['total_size_mb']} MB")
        print(f"  Average event size: {session_summary['summary_stats']['average_event_size_bytes']} bytes")
        
    except Exception as e:
        print(f"Error during storage test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        storage_manager.stop()
    
    print("✓ Storage system test completed")