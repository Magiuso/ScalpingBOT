#!/usr/bin/env python3
"""
Script per verificare l'installazione delle librerie richieste dall'Analyzer
Esegui con: python library_check.py
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple

def check_python_version():
    """Verifica la versione di Python"""
    print("=" * 60)
    print("🐍 VERIFICA VERSIONE PYTHON")
    print("=" * 60)
    version = sys.version_info
    print(f"Versione Python: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERRORE: Python 3.8+ richiesto!")
        return False
    else:
        print("✅ Versione Python OK")
        return True

def get_pip_list():
    """Ottiene la lista dei pacchetti installati"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                              capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return ""

def check_library(library_name: str, import_name: str | None = None, version_attr: str | None = None) -> Tuple[bool, str, str]:
    """
    Verifica se una libreria è installata e importabile
    
    Args:
        library_name: Nome del pacchetto (per pip)
        import_name: Nome per l'import (se diverso)
        version_attr: Attributo per la versione
    
    Returns:
        (installed, version, error_message)
    """
    if import_name is None:
        import_name = library_name
    
    try:
        module = importlib.import_module(import_name)
        
        # Prova a ottenere la versione
        version = "Unknown"
        if version_attr and hasattr(module, version_attr):
            version = getattr(module, version_attr)
        elif hasattr(module, '__version__'):
            version = module.__version__
        elif hasattr(module, 'VERSION'):
            version = str(module.VERSION)
        elif hasattr(module, 'version'):
            version = str(module.version)
        
        return True, version, ""
        
    except ImportError as e:
        return False, "", str(e)
    except Exception as e:
        return False, "", f"Errore inaspettato: {str(e)}"

def main():
    """Funzione principale"""
    print("🔍 VERIFICA LIBRERIE ANALYZER")
    print("=" * 60)
    
    # Verifica versione Python
    if not check_python_version():
        return
    
    print("\n" + "=" * 60)
    print("📦 VERIFICA LIBRERIE INSTALLATE")
    print("=" * 60)
    
    # Lista delle librerie da verificare
    libraries = [
        # Core libraries (dovrebbero essere sempre presenti)
        ("numpy", "numpy", "__version__"),
        ("pandas", "pandas", "__version__"),
        
        # Machine Learning libraries
        ("scikit-learn", "sklearn", "__version__"),
        ("tensorflow", "tensorflow", "__version__"),
        ("torch", "torch", "__version__"),
        ("transformers", "transformers", "__version__"),
        
        # Financial libraries
        ("TA-Lib", "talib", "__version__"),
        ("MetaTrader5", "MetaTrader5", "__version__"),
        
        # Utility libraries
        ("pathlib", "pathlib", None),  # Built-in
        ("typing", "typing", None),    # Built-in
        ("dataclasses", "dataclasses", None),  # Built-in (Python 3.7+)
        ("collections", "collections", None),  # Built-in
        ("datetime", "datetime", None),        # Built-in
        ("json", "json", None),               # Built-in
        ("pickle", "pickle", None),           # Built-in
        ("threading", "threading", None),     # Built-in
        ("asyncio", "asyncio", None),         # Built-in
        ("enum", "enum", None),               # Built-in
        ("warnings", "warnings", None),       # Built-in
        ("csv", "csv", None),                 # Built-in
        ("logging", "logging", None),         # Built-in
        ("gzip", "gzip", None),               # Built-in
        ("shutil", "shutil", None),           # Built-in
    ]
    
    results = {}
    
    for lib_name, import_name, version_attr in libraries:
        print(f"\n📋 Verificando {lib_name}...")
        installed, version, error = check_library(lib_name, import_name, version_attr)
        
        results[lib_name] = {
            'installed': installed,
            'version': version,
            'error': error,
            'import_name': import_name
        }
        
        if installed:
            print(f"   ✅ {lib_name} ({import_name}) - v{version}")
        else:
            print(f"   ❌ {lib_name} ({import_name}) - {error}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RIASSUNTO")
    print("=" * 60)
    
    installed_count = sum(1 for r in results.values() if r['installed'])
    total_count = len(results)
    
    print(f"Librerie installate: {installed_count}/{total_count}")
    
    # Separazione per categoria
    core_libs = ["numpy", "pandas"]
    ml_libs = ["scikit-learn", "tensorflow", "torch", "transformers"]
    financial_libs = ["TA-Lib", "MetaTrader5"]
    
    print(f"\n🔧 CORE LIBRARIES:")
    for lib in core_libs:
        if lib in results:
            status = "✅" if results[lib]['installed'] else "❌"
            print(f"   {status} {lib}")
    
    print(f"\n🤖 MACHINE LEARNING LIBRARIES:")
    for lib in ml_libs:
        if lib in results:
            status = "✅" if results[lib]['installed'] else "❌"
            print(f"   {status} {lib}")
    
    print(f"\n💰 FINANCIAL LIBRARIES:")
    for lib in financial_libs:
        if lib in results:
            status = "✅" if results[lib]['installed'] else "❌"
            print(f"   {status} {lib}")
    
    # Comandi di installazione per librerie mancanti
    missing_libs = [lib for lib, data in results.items() if not data['installed']]
    
    if missing_libs:
        print(f"\n" + "=" * 60)
        print("🛠️  COMANDI DI INSTALLAZIONE")
        print("=" * 60)
        
        # Mapping dei nomi per pip install
        pip_names = {
            "scikit-learn": "scikit-learn",
            "tensorflow": "tensorflow",
            "torch": "torch torchvision torchaudio",
            "transformers": "transformers",
            "TA-Lib": "TA-Lib",
            "MetaTrader5": "MetaTrader5",
            "numpy": "numpy",
            "pandas": "pandas"
        }
        
        print("Esegui questi comandi per installare le librerie mancanti:\n")
        
        for lib in missing_libs:
            if lib in pip_names:
                print(f"pip install {pip_names[lib]}")
        
        # Note speciali
        if "TA-Lib" in missing_libs:
            print(f"\n📝 NOTA PER TA-LIB:")
            print("Su Windows, potresti aver bisogno di:")
            print("1. Scaricare il wheel da: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
            print("2. pip install TA_Lib-0.4.xx-cpxx-cpxx-win_amd64.whl")
            print("Oppure usando conda: conda install -c conda-forge ta-lib")
        
        if "torch" in missing_libs:
            print(f"\n📝 NOTA PER PYTORCH:")
            print("Per installazione ottimizzata, visita: https://pytorch.org/get-started/locally/")
            print("Es. con CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
        if "MetaTrader5" in missing_libs:
            print(f"\n📝 NOTA PER METATRADER5:")
            print("Assicurati di avere MT5 installato sul sistema")
            print("pip install MetaTrader5")
    
    # Verifica pip list
    print(f"\n" + "=" * 60)
    print("📦 PACCHETTI PIP INSTALLATI")
    print("=" * 60)
    
    pip_output = get_pip_list()
    if pip_output:
        # Mostra solo le librerie rilevanti
        relevant_packages = []
        for line in pip_output.split('\n'):
            line = line.strip()
            if any(lib.lower() in line.lower() for lib in ['numpy', 'pandas', 'sklearn', 'tensorflow', 'torch', 'transformers', 'talib', 'metatrader']):
                relevant_packages.append(line)
        
        if relevant_packages:
            for package in relevant_packages:
                print(f"   {package}")
        else:
            print("   Nessun pacchetto rilevante trovato")
    else:
        print("   ❌ Impossibile ottenere la lista pip")
    
    # Verifica environment Python
    print(f"\n" + "=" * 60)
    print("🔧 INFORMAZIONI ENVIRONMENT")
    print("=" * 60)
    print(f"Eseguibile Python: {sys.executable}")
    print(f"Percorso Python: {sys.path[0]}")
    print(f"Versione completa: {sys.version}")
    
    # Test import specifici per debugging
    print(f"\n" + "=" * 60)
    print("🧪 TEST IMPORT SPECIFICI")
    print("=" * 60)
    
    test_imports = [
        "from typing import Dict, List, Optional",
        "import numpy as np",
        "import pandas as pd",
        "from dataclasses import dataclass, field",
        "from collections import deque, defaultdict"
    ]
    
    for test_import in test_imports:
        try:
            exec(test_import)
            print(f"   ✅ {test_import}")
        except Exception as e:
            print(f"   ❌ {test_import} - {e}")
    
    print(f"\n" + "=" * 60)
    print("✅ VERIFICA COMPLETATA")
    print("=" * 60)

if __name__ == "__main__":
    main()