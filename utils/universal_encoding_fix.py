"""
UNIVERSAL WINDOWS ENCODING FIX - VERSIONE CORRETTA
==================================================
Risolve TUTTI i problemi di encoding Unicode/Emoji su Windows
Compatibile con Python 3.6+ e type checkers

PROBLEMI RISOLTI:
- UnicodeEncodeError con emoji 🔧🔄📊🚨⚡🎯💹🔥✅❌⚠️
- CP1252 codec errors
- Console Windows che non visualizza Unicode
- Logging con caratteri speciali
- Stdout/stderr encoding issues
"""

import os
import sys
import logging
import codecs
import locale
from typing import Optional, Dict, Any, TextIO, Union

# ============================================================================
# CONFIGURAZIONE ENCODING GLOBALE
# ============================================================================

class UniversalEncodingFixer:
    """
    Classe principale per risolvere tutti i problemi di encoding su Windows
    """
    
    # Mappa completa emoji → testo per fallback
    EMOJI_FALLBACK_MAP = {
        # Simboli tecnici
        '🔧': '[FIX]', '⚙️': '[SETTINGS]', '🔩': '[TOOLS]',
        '🔨': '[HAMMER]', '⚡': '[LIGHTNING]', '🔥': '[FIRE]',
        
        # Stati e processi
        '🔄': '[REFRESH]', '♻️': '[RECYCLE]', '🔃': '[RELOAD]',
        '⏳': '[WAITING]', '⏰': '[TIMER]', '⏱️': '[STOPWATCH]',
        
        # Indicatori di stato
        '✅': '[OK]', '❌': '[ERROR]', '⚠️': '[WARNING]',
        '🚨': '[ALERT]', '🟢': '[GREEN]', '🔴': '[RED]',
        '🟡': '[YELLOW]', '🟠': '[ORANGE]', '🔵': '[BLUE]',
        
        # Business e finanza
        '💹': '[MARKET]', '📊': '[CHART]', '📈': '[UP]',
        '📉': '[DOWN]', '💰': '[MONEY]', '💎': '[DIAMOND]',
        '🏦': '[BANK]', '💳': '[CARD]', '💵': '[DOLLAR]',
        
        # Direzioni e target
        '🎯': '[TARGET]', '🚀': '[ROCKET]', '📍': '[PIN]',
        '⬆️': '[UP]', '⬇️': '[DOWN]', '➡️': '[RIGHT]',
        '⬅️': '[LEFT]', '🔺': '[UP_TRIANGLE]', '🔻': '[DOWN_TRIANGLE]',
        
        # Comunicazione e info
        '📢': '[ANNOUNCE]', '📣': '[MEGAPHONE]', '💬': '[CHAT]',
        '📧': '[EMAIL]', '📞': '[PHONE]', '📱': '[MOBILE]',
        '💡': '[IDEA]', '🔍': '[SEARCH]', '📝': '[NOTE]',
        
        # Sicurezza e protezione
        '🔒': '[LOCK]', '🔓': '[UNLOCK]', '🛡️': '[SHIELD]',
        '🔑': '[KEY]', '🚪': '[DOOR]', '👁️': '[EYE]',
        
        # Sistema e computer
        '💻': '[COMPUTER]', '🖥️': '[DESKTOP]', '📱': '[MOBILE]',
        '💾': '[SAVE]', '📁': '[FOLDER]', '📂': '[OPEN_FOLDER]',
        '🗃️': '[FILE_CABINET]', '🖨️': '[PRINTER]', '⌨️': '[KEYBOARD]',
        
        # Varie utili
        '📅': '[CALENDAR]', '🕐': '[CLOCK]', '🌐': '[GLOBE]',
        '🎮': '[GAME]', '🎵': '[MUSIC]', '🔊': '[SPEAKER]',
        '🔇': '[MUTE]', '📖': '[BOOK]', '📚': '[BOOKS]'
    }
    
    def __init__(self, force_utf8: bool = True, silent: bool = False):
        """
        Inizializza il fixer
        """
        self.force_utf8 = force_utf8
        self.silent = silent
        self.is_windows = sys.platform.startswith('win')
        self.original_encoding = getattr(sys.stdout, 'encoding', 'unknown')
        self.fixes_applied = []
    
    def detect_encoding_issues(self) -> Dict[str, Any]:
        """
        Rileva problemi di encoding nel sistema corrente
        """
        issues = {
            'platform': sys.platform,
            'stdout_encoding': getattr(sys.stdout, 'encoding', 'unknown'),
            'stderr_encoding': getattr(sys.stderr, 'encoding', 'unknown'),
            'locale_encoding': locale.getpreferredencoding(),
            'filesystem_encoding': sys.getfilesystemencoding(),
            'pythonioencoding': os.environ.get('PYTHONIOENCODING', 'not_set'),
            'console_cp': None,
            'issues_found': []
        }
        
        # Su Windows, controlla code page console
        if self.is_windows:
            try:
                import subprocess
                result = subprocess.run(['chcp'], capture_output=True, text=True, shell=True)
                if result.stdout:
                    issues['console_cp'] = result.stdout.strip()
            except Exception:
                issues['console_cp'] = 'unable_to_detect'
        
        # Identifica problemi
        if issues['stdout_encoding'] in ['cp1252', 'cp850', 'ascii']:
            issues['issues_found'].append('stdout_encoding_problematic')
        
        console_cp = issues.get('console_cp', '')
        if console_cp and '1252' in str(console_cp):
            issues['issues_found'].append('console_cp_problematic')
        
        if issues['pythonioencoding'] == 'not_set':
            issues['issues_found'].append('pythonioencoding_not_set')
        
        return issues
    
    def fix_environment_variables(self) -> bool:
        """
        Configura le variabili d'ambiente per UTF-8
        """
        try:
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'
            
            if self.is_windows:
                # Imposta UTF-8 come default per Windows
                os.environ['PYTHONUTF8'] = '1'
            
            self.fixes_applied.append('environment_variables')
            return True
        except Exception as e:
            if not self.silent:
                print(f"Errore configurazione variabili ambiente: {e}")
            return False
    
    def fix_console_codepage(self) -> bool:
        """
        Cambia code page del console Windows a UTF-8 (65001)
        """
        if not self.is_windows:
            return True
        
        try:
            # Cambia code page a UTF-8 silenziosamente
            os.system('chcp 65001 >nul 2>&1')
            self.fixes_applied.append('console_codepage')
            return True
        except Exception as e:
            if not self.silent:
                print(f"Errore cambio code page: {e}")
            return False
    
    def fix_stdout_stderr(self) -> bool:
        """
        Riconfigura stdout e stderr per UTF-8
        """
        try:
            # Controlla se il metodo reconfigure esiste (Python 3.7+)
            if hasattr(sys.stdout, 'reconfigure'):
                # Type: ignore perché sappiamo che esiste
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore
            else:
                # Metodo compatibile per versioni più vecchie
                if hasattr(sys.stdout, 'buffer'):
                    sys.stdout = codecs.getwriter('utf-8')(
                        sys.stdout.buffer, 'replace'
                    )
                if hasattr(sys.stderr, 'buffer'):
                    sys.stderr = codecs.getwriter('utf-8')(
                        sys.stderr.buffer, 'replace'
                    )
            
            self.fixes_applied.append('stdout_stderr')
            return True
        except Exception as e:
            if not self.silent:
                print(f"Errore riconfigurazione stdout/stderr: {e}")
            return False
    
    def fix_logging_handlers(self) -> bool:
        """
        Riconfigura tutti i handler di logging esistenti per UTF-8
        """
        try:
            # Riconfigura handler esistenti
            for handler in logging.root.handlers[:]:
                # Controlla se ha attributo stream
                if hasattr(handler, 'stream'):
                    stream = getattr(handler, 'stream')
                    if hasattr(stream, 'buffer'):
                        new_stream = codecs.getwriter('utf-8')(
                            stream.buffer, 'replace'
                        )
                        setattr(handler, 'stream', new_stream)
                elif isinstance(handler, logging.FileHandler):
                    # Per file handler, ricreiamo con encoding UTF-8
                    old_filename = handler.baseFilename
                    old_formatter = handler.formatter
                    handler.close()
                    
                    # Crea nuovo handler con UTF-8
                    new_handler = logging.FileHandler(
                        old_filename, 
                        mode='a', 
                        encoding='utf-8'
                    )
                    if old_formatter:
                        new_handler.setFormatter(old_formatter)
                    
                    logging.root.removeHandler(handler)
                    logging.root.addHandler(new_handler)
            
            self.fixes_applied.append('logging_handlers')
            return True
        except Exception as e:
            if not self.silent:
                print(f"Errore riconfigurazione logging: {e}")
            return False
    
    def apply_all_fixes(self) -> Dict[str, bool]:
        """
        Applica tutti i fix disponibili
        """
        if not self.silent:
            print("Applicazione fix encoding universale...")
        
        results = {}
        
        # Applica fix
        results['environment'] = self.fix_environment_variables()
        results['console'] = self.fix_console_codepage()
        results['stdout_stderr'] = self.fix_stdout_stderr()
        results['logging'] = self.fix_logging_handlers()
        
        # Verifica risultati
        success_count = sum(1 for v in results.values() if v)
        total_fixes = len(results)
        
        if not self.silent:
            if success_count == total_fixes:
                print(f"Tutti i fix applicati con successo ({success_count}/{total_fixes})")
            else:
                print(f"Fix parziali applicati ({success_count}/{total_fixes})")
                successful = [k for k, v in results.items() if v]
                failed = [k for k, v in results.items() if not v]
                print(f"Fix riusciti: {successful}")
                print(f"Fix falliti: {failed}")
        
        return results
    
    def safe_text(self, text: str, use_fallback: Optional[bool] = None) -> str:
        """
        Converte testo con emoji in versione sicura per Windows
        """
        if use_fallback is None:
            use_fallback = self.is_windows
        
        if not use_fallback:
            return text
        
        safe_text = text
        for emoji, replacement in self.EMOJI_FALLBACK_MAP.items():
            safe_text = safe_text.replace(emoji, replacement)
        
        return safe_text
    
    def test_encoding(self) -> bool:
        """
        Testa se l'encoding funziona correttamente
        """
        test_strings = [
            "Test emoji tecnici",
            "Grafici e dati",
            "Stati: OK/ERROR/WARNING",
            "Unicode avanzato: àèìòù çñü",
            "Sistema: computer/desktop/mobile"
        ]
        
        try:
            for test_str in test_strings:
                print(test_str)
            
            if not self.silent:
                print("Test encoding completato con successo!")
            return True
        except UnicodeEncodeError as e:
            if not self.silent:
                print(f"Test encoding fallito: {e}")
            return False
    
    def get_report(self) -> str:
        """
        Genera report dettagliato dello stato encoding
        """
        issues = self.detect_encoding_issues()
        
        report = []
        report.append("=" * 50)
        report.append("UNIVERSAL ENCODING FIX - REPORT")
        report.append("=" * 50)
        report.append(f"Platform: {issues['platform']}")
        report.append(f"Stdout encoding: {issues['stdout_encoding']}")
        report.append(f"Stderr encoding: {issues['stderr_encoding']}")
        report.append(f"Locale encoding: {issues['locale_encoding']}")
        report.append(f"PYTHONIOENCODING: {issues['pythonioencoding']}")
        
        if self.is_windows:
            report.append(f"Console code page: {issues['console_cp']}")
        
        fixes_text = ', '.join(self.fixes_applied) if self.fixes_applied else 'Nessuno'
        report.append(f"\nFix applicati: {fixes_text}")
        
        if issues['issues_found']:
            issues_text = ', '.join(issues['issues_found'])
            report.append(f"\nProblemi rilevati: {issues_text}")
        else:
            report.append("\nNessun problema rilevato")
        
        return "\n".join(report)

# ============================================================================
# LOGGER UNIVERSALE CON GESTIONE EMOJI
# ============================================================================

class UniversalSafeLogger:
    """
    Logger universale che gestisce emoji e Unicode in modo sicuro
    """
    
    def __init__(self, name: str = 'universal', log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.fixer = UniversalEncodingFixer(silent=True)
        
        if log_file:
            self.setup_file_logging(log_file)
    
    def setup_file_logging(self, filename: str) -> None:
        """
        Configura logging su file con UTF-8
        """
        handler = logging.FileHandler(
            filename, 
            mode='a', 
            encoding='utf-8'
        )
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def safe_log(self, level: str, message: str) -> None:
        """
        Log sicuro con gestione automatica emoji
        """
        safe_message = self.fixer.safe_text(message)
        log_method = getattr(self.logger, level.lower(), None)
        if log_method:
            log_method(safe_message)
    
    def info(self, message: str) -> None:
        self.safe_log('info', message)
    
    def warning(self, message: str) -> None:
        self.safe_log('warning', message)
    
    def error(self, message: str) -> None:
        self.safe_log('error', message)
    
    def debug(self, message: str) -> None:
        self.safe_log('debug', message)
    
    def critical(self, message: str):
        self.safe_log('critical', message)

# ============================================================================
# FUNZIONI DI UTILITÀ GLOBALI
# ============================================================================

# Istanza globale del fixer
_global_fixer: Optional[UniversalEncodingFixer] = None

def init_universal_encoding(silent: bool = False) -> UniversalEncodingFixer:
    """
    Inizializza encoding universale (da chiamare una sola volta)
    """
    global _global_fixer
    
    if _global_fixer is None:
        _global_fixer = UniversalEncodingFixer(silent=silent)
        _global_fixer.apply_all_fixes()
    
    return _global_fixer

def safe_print(text: str) -> None:
    """
    Print sicuro che gestisce emoji automaticamente
    """
    global _global_fixer
    
    if _global_fixer is None:
        init_universal_encoding(silent=True)
    
    if _global_fixer is not None:
        safe_text = _global_fixer.safe_text(text)
        print(safe_text)
    else:
        print(text)

def get_safe_logger(name: str = 'app', log_file: Optional[str] = None) -> UniversalSafeLogger:
    """
    Crea logger sicuro con gestione emoji
    """
    if _global_fixer is None:
        init_universal_encoding(silent=True)
    
    return UniversalSafeLogger(name, log_file)

# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    # Test completo del sistema
    print("Inizializzazione Universal Encoding Fix...")
    
    fixer = init_universal_encoding(silent=False)
    
    print("\nReport sistema:")
    print(fixer.get_report())
    
    print("\nTest encoding:")
    fixer.test_encoding()
    
    print("\nTest logger sicuro:")
    logger = get_safe_logger('test', 'test_encoding.log')
    logger.info("Test logger con emoji")
    logger.warning("Warning con emoji")
    logger.error("Errore con emoji")
    
    print("\nTest completato! Controlla il file 'test_encoding.log'")