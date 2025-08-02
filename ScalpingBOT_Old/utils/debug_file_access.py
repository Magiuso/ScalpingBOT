#!/usr/bin/env python3
"""
🔍 Debug File Access - Verifica accesso ai file MT5
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime

def check_file_access():
    """Verifica completa accesso file"""
    
    print("="*60)
    print("🔍 DEBUG FILE ACCESS - MT5 ANALYZER")
    print("="*60)
    
    # Path da testare (MODIFICA QUESTO!)
    MT5_PATH = r"C:\Users\anton\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files"
    
    print(f"📁 Testing path: {MT5_PATH}")
    
    # 1. Verifica esistenza directory
    print("\n1️⃣ DIRECTORY CHECK:")
    if os.path.exists(MT5_PATH):
        print("✅ Directory exists")
        print(f"   Is directory: {os.path.isdir(MT5_PATH)}")
        print(f"   Readable: {os.access(MT5_PATH, os.R_OK)}")
        print(f"   Writable: {os.access(MT5_PATH, os.W_OK)}")
    else:
        print("❌ Directory does not exist!")
        print("   Check your MT5 installation path")
        return False
    
    # 2. Lista tutti i file
    print("\n2️⃣ ALL FILES:")
    try:
        all_files = os.listdir(MT5_PATH)
        print(f"   Total files: {len(all_files)}")
        
        # Filtra per tipo
        jsonl_files = [f for f in all_files if f.endswith('.jsonl')]
        txt_files = [f for f in all_files if f.endswith('.txt')]
        
        print(f"   .jsonl files: {len(jsonl_files)}")
        print(f"   .txt files: {len(txt_files)}")
        
        if jsonl_files:
            print("   📄 JSONL files found:")
            for f in jsonl_files:
                print(f"      - {f}")
        
        if txt_files:
            print("   📄 TXT files found:")
            for f in txt_files[:10]:  # Solo primi 10
                print(f"      - {f}")
            if len(txt_files) > 10:
                print(f"      ... and {len(txt_files)-10} more")
        
    except Exception as e:
        print(f"❌ Error listing files: {e}")
        return False
    
    # 3. Cerca file analyzer specifici
    print("\n3️⃣ ANALYZER FILES:")
    analyzer_files = [f for f in all_files if f.startswith('analyzer_')]
    
    if analyzer_files:
        print(f"   Found {len(analyzer_files)} analyzer files:")
        for f in analyzer_files:
            file_path = os.path.join(MT5_PATH, f)
            try:
                stat = os.stat(file_path)
                size_mb = stat.st_size / 1024 / 1024
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                print(f"      ✅ {f} ({size_mb:.2f} MB, modified: {mod_time})")
                
                # Test lettura
                test_read_file(file_path)
                
            except Exception as e:
                print(f"      ❌ {f} - Error: {e}")
    else:
        print("   ❌ No analyzer files found!")
        print("   Expected files like: analyzer_USTEC.jsonl")
        print("   Make sure MT5 EA is running and generating files")
    
    # 4. Test creazione file
    print("\n4️⃣ WRITE TEST:")
    test_file = os.path.join(MT5_PATH, "python_test.txt")
    try:
        with open(test_file, 'w') as f:
            f.write(f"Test from Python at {datetime.now()}")
        print("   ✅ Can write to directory")
        
        # Rimuovi file test
        os.remove(test_file)
        print("   ✅ Can delete files")
        
    except Exception as e:
        print(f"   ❌ Cannot write: {e}")
    
    return True

def test_read_file(file_path):
    """Test lettura specifico file"""
    print(f"\n   🔍 Testing read access: {os.path.basename(file_path)}")
    
    try:
        # Test 1: Apertura file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # Primi 1000 caratteri
        print(f"      ✅ File readable ({len(content)} chars read)")
        
        # Test 2: Parse JSON
        lines = content.strip().split('\n')
        valid_json_lines = 0
        
        for line in lines[:5]:  # Test primi 5 righe
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    valid_json_lines += 1
                    
                    # Mostra tipo messaggio
                    msg_type = data.get('type', 'unknown')
                    timestamp = data.get('timestamp', 'no-time')
                    print(f"         📄 Line: type={msg_type}, time={timestamp}")
                    
                except json.JSONDecodeError:
                    print(f"         ⚠️ Invalid JSON: {line[:50]}...")
        
        print(f"      ✅ Valid JSON lines: {valid_json_lines}")
        
        # Test 3: Monitoraggio modifiche
        stat_before = os.stat(file_path)
        print(f"      📊 File size: {stat_before.st_size} bytes")
        print(f"      🕒 Last modified: {datetime.fromtimestamp(stat_before.st_mtime)}")
        
        # Attendi modifiche
        print("      ⏳ Monitoring for changes (10 seconds)...")
        start_time = time.time()
        
        while time.time() - start_time < 10:
            try:
                stat_now = os.stat(file_path)
                if stat_now.st_mtime > stat_before.st_mtime:
                    print(f"      🔄 File modified! New size: {stat_now.st_size} bytes")
                    return
                time.sleep(1)
            except Exception as e:
                print(f"      ❌ Error monitoring: {e}")
                return
        
        print("      ⏰ No changes detected in 10 seconds")
        
    except PermissionError:
        print("      ❌ Permission denied - file may be locked by MT5")
    except FileNotFoundError:
        print("      ❌ File disappeared")
    except Exception as e:
        print(f"      ❌ Read error: {e}")

def monitor_directory():
    """Monitora directory per nuovi file"""
    MT5_PATH = r"C:\Users\anton\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\MQL5\Files"
    
    if not os.path.exists(MT5_PATH):
        print("❌ Directory not found")
        return
    
    print("\n" + "="*60)
    print("👁️ MONITORING DIRECTORY FOR CHANGES")
    print("="*60)
    print(f"📁 Watching: {MT5_PATH}")
    print("🔄 Looking for new analyzer files...")
    print("⏹️ Press Ctrl+C to stop")
    
    last_files = set()
    
    try:
        while True:
            current_files = set()
            
            try:
                all_files = os.listdir(MT5_PATH)
                analyzer_files = [f for f in all_files if f.startswith('analyzer_') and f.endswith('.jsonl')]
                current_files = set(analyzer_files)
                
                # Nuovi file
                new_files = current_files - last_files
                if new_files:
                    for new_file in new_files:
                        print(f"🆕 NEW FILE: {new_file}")
                        file_path = os.path.join(MT5_PATH, new_file)
                        test_read_file(file_path)
                
                # File rimossi
                removed_files = last_files - current_files
                if removed_files:
                    for removed_file in removed_files:
                        print(f"🗑️ REMOVED: {removed_file}")
                
                last_files = current_files
                
                # Status ogni 30 secondi
                if len(current_files) > 0:
                    print(f"📊 {datetime.now().strftime('%H:%M:%S')} - Monitoring {len(current_files)} files: {list(current_files)}")
                else:
                    print(f"⚠️ {datetime.now().strftime('%H:%M:%S')} - No analyzer files found")
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
            
            time.sleep(30)  # Check ogni 30 secondi
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        monitor_directory()
    else:
        if check_file_access():
            print("\n" + "="*60)
            print("✅ BASIC CHECKS COMPLETED")
            print("="*60)
            print("Next steps:")
            print("1. If no analyzer files found, check MT5 EA is running")
            print("2. Try: python debug_file_access.py --monitor")
            print("3. Check MT5 logs for file creation errors")
        else:
            print("\n❌ BASIC CHECKS FAILED")
            print("Fix directory access issues first")

if __name__ == "__main__":
    main()