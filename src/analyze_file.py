#!/usr/bin/env python3
"""
Script per analizzare un file di codice grande e generare una mappa strutturale
per suddividerlo in blocchi più piccoli e mantenibili.
"""

import ast
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
import json

class CodeAnalyzer:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.content = ""
        self.lines: List[str] = []
        self.analysis = {
            "file_info": {},
            "structure": {},
            "dependencies": {},
            "metrics": {},
            "suggested_splits": []
        }
    
    def read_file(self):
        """Legge il contenuto del file."""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            self.content = f.read()
            self.lines = self.content.splitlines()
        
        self.analysis["file_info"] = {
            "filename": os.path.basename(self.filepath),
            "total_lines": len(self.lines),
            "size_kb": len(self.content) / 1024
        }
    
    def analyze_python_structure(self):
        """Analizza la struttura del codice Python."""
        try:
            tree = ast.parse(self.content)
            
            # Analizza classi
            classes = []
            functions = []
            imports = []
            
            # Prima passata: raccogli tutte le classi per identificare i metodi
            class_nodes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_nodes.append(node)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Gestisce il caso in cui end_lineno potrebbe essere None
                    end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
                    
                    class_info = {
                        "name": node.name,
                        "line_start": node.lineno,
                        "line_end": end_lineno,
                        "methods": [],
                        "size": end_lineno - node.lineno + 1
                    }
                    
                    # Trova i metodi della classe
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            method_end_lineno = item.end_lineno if item.end_lineno is not None else item.lineno
                            class_info["methods"].append({
                                "name": item.name,
                                "line_start": item.lineno,
                                "line_end": method_end_lineno,
                                "size": method_end_lineno - item.lineno + 1
                            })
                    
                    classes.append(class_info)
                
                elif isinstance(node, ast.FunctionDef):
                    # Verifica se la funzione è dentro una classe
                    is_method = self._is_node_in_class(node, class_nodes)
                    
                    if not is_method:
                        # Solo funzioni top-level
                        func_end_lineno = node.end_lineno if node.end_lineno is not None else node.lineno
                        functions.append({
                            "name": node.name,
                            "line_start": node.lineno,
                            "line_end": func_end_lineno,
                            "size": func_end_lineno - node.lineno + 1
                        })
                
                elif isinstance(node, ast.Import):
                    # ast.Import non ha attributo module, ha names
                    for alias in node.names:
                        imports.append({
                            "line": node.lineno,
                            "module": alias.name,
                            "type": "import"
                        })
                
                elif isinstance(node, ast.ImportFrom):
                    # ast.ImportFrom ha l'attributo module
                    module_name = node.module if node.module is not None else "."
                    imports.append({
                        "line": node.lineno,
                        "module": module_name,
                        "type": "from_import"
                    })
            
            self.analysis["structure"] = {
                "classes": classes,
                "functions": functions,
                "imports": imports
            }
            
        except SyntaxError as e:
            print(f"Errore di sintassi nel file Python: {e}")
            self.analyze_generic_structure()
    
    def _is_node_in_class(self, func_node: ast.FunctionDef, class_nodes: List[ast.ClassDef]) -> bool:
        """Verifica se un nodo funzione è dentro una classe."""
        for class_node in class_nodes:
            if (func_node.lineno >= class_node.lineno and 
                func_node.lineno <= (class_node.end_lineno or class_node.lineno)):
                return True
        return False
    
    def analyze_generic_structure(self):
        """Analizza la struttura generica del file (non-Python)."""
        # Pattern per riconoscere blocchi di codice
        patterns = {
            "class": re.compile(r'^class\s+(\w+)', re.MULTILINE),
            "function": re.compile(r'^(def|function|func)\s+(\w+)', re.MULTILINE),
            "method": re.compile(r'^\s+(def|function|func)\s+(\w+)', re.MULTILINE),
            "section": re.compile(r'^#{2,}\s*(.+)$|^/\*{2,}\s*(.+)\s*\*{2,}/$', re.MULTILINE)
        }
        
        blocks = []
        
        # Trova sezioni/commenti strutturali
        for match in patterns["section"].finditer(self.content):
            line_num = self.content[:match.start()].count('\n') + 1
            section_name = match.group(1) or match.group(2) or "Unknown Section"
            blocks.append({
                "type": "section",
                "name": section_name,
                "line": line_num
            })
        
        self.analysis["structure"]["generic_blocks"] = blocks
    
    def analyze_complexity(self):
        """Analizza la complessità e le metriche del codice."""
        total_lines = len(self.lines)
        if total_lines == 0:
            metrics = {
                "avg_line_length": 0,
                "empty_lines": 0,
                "comment_lines": 0,
                "long_lines": 0,
                "very_long_lines": 0,
                "code_density": 0
            }
        else:
            metrics = {
                "avg_line_length": sum(len(line) for line in self.lines) / total_lines,
                "empty_lines": sum(1 for line in self.lines if not line.strip()),
                "comment_lines": sum(1 for line in self.lines if line.strip().startswith(('#', '//', '/*', '*'))),
                "long_lines": sum(1 for line in self.lines if len(line) > 120),
                "very_long_lines": sum(1 for line in self.lines if len(line) > 200)
            }
            
            # Calcola densità del codice
            code_lines = total_lines - metrics["empty_lines"] - metrics["comment_lines"]
            metrics["code_density"] = code_lines / total_lines
        
        self.analysis["metrics"] = metrics
    
    def find_logical_splits(self):
        """Suggerisce come dividere il file in blocchi logici."""
        suggestions = []
        target_lines = 500  # Dimensione target per ogni file
        
        if "classes" in self.analysis["structure"]:
            # Per file Python
            classes = self.analysis["structure"]["classes"]
            
            # Raggruppa classi correlate
            current_group = []
            current_size = 0
            
            for cls in sorted(classes, key=lambda x: x["line_start"]):
                if current_size + cls["size"] > target_lines and current_group:
                    base_name = self.analysis['file_info']['filename'].split('.')[0]
                    suggestions.append({
                        "type": "class_group",
                        "items": current_group.copy(),
                        "total_lines": current_size,
                        "suggested_filename": f"{base_name}_part_{len(suggestions)+1}.py"
                    })
                    current_group = [cls["name"]]
                    current_size = cls["size"]
                else:
                    current_group.append(cls["name"])
                    current_size += cls["size"]
            
            # Aggiungi l'ultimo gruppo
            if current_group:
                base_name = self.analysis['file_info']['filename'].split('.')[0]
                suggestions.append({
                    "type": "class_group",
                    "items": current_group,
                    "total_lines": current_size,
                    "suggested_filename": f"{base_name}_part_{len(suggestions)+1}.py"
                })
            
            # Suggerisci file separati per classi molto grandi
            for cls in classes:
                if cls["size"] > target_lines:
                    suggestions.append({
                        "type": "large_class",
                        "name": cls["name"],
                        "lines": cls["size"],
                        "suggested_filename": f"{cls['name'].lower()}.py",
                        "recommendation": "Considera di suddividere questa classe in più classi più piccole"
                    })
        
        # Suggerimenti basati su sezioni
        if not suggestions:
            # Dividi per numero di righe
            total_lines = len(self.lines)
            num_files = (total_lines + target_lines - 1) // target_lines
            base_name = self.analysis['file_info']['filename'].split('.')[0]
            
            for i in range(num_files):
                start_line = i * target_lines + 1
                end_line = min((i + 1) * target_lines, total_lines)
                suggestions.append({
                    "type": "line_based",
                    "start_line": start_line,
                    "end_line": end_line,
                    "total_lines": end_line - start_line + 1,
                    "suggested_filename": f"{base_name}_part_{i+1}.py"
                })
        
        self.analysis["suggested_splits"] = suggestions
    
    def generate_report(self) -> str:
        """Genera un report dettagliato dell'analisi."""
        report = []
        report.append("=" * 80)
        report.append(f"ANALISI FILE: {self.analysis['file_info']['filename']}")
        report.append("=" * 80)
        report.append(f"\nINFORMAZIONI GENERALI:")
        report.append(f"- Righe totali: {self.analysis['file_info']['total_lines']}")
        report.append(f"- Dimensione: {self.analysis['file_info']['size_kb']:.2f} KB")
        
        if "metrics" in self.analysis:
            report.append(f"\nMETRICHE:")
            report.append(f"- Lunghezza media riga: {self.analysis['metrics']['avg_line_length']:.1f}")
            report.append(f"- Righe vuote: {self.analysis['metrics']['empty_lines']}")
            report.append(f"- Righe commento: {self.analysis['metrics']['comment_lines']}")
            report.append(f"- Righe molto lunghe (>120): {self.analysis['metrics']['long_lines']}")
            report.append(f"- Densità codice: {self.analysis['metrics']['code_density']:.2%}")
        
        if "classes" in self.analysis["structure"]:
            report.append(f"\nSTRUTTURA CLASSI:")
            for cls in self.analysis["structure"]["classes"]:
                report.append(f"\n  Classe: {cls['name']} (righe {cls['line_start']}-{cls['line_end']}, {cls['size']} righe)")
                if cls["methods"]:
                    report.append(f"  Metodi ({len(cls['methods'])}):")
                    for method in cls["methods"][:5]:  # Mostra solo i primi 5
                        report.append(f"    - {method['name']} ({method['size']} righe)")
                    if len(cls["methods"]) > 5:
                        report.append(f"    ... e altri {len(cls['methods']) - 5} metodi")
        
        report.append(f"\nSUGGERIMENTI PER LA DIVISIONE:")
        for i, suggestion in enumerate(self.analysis["suggested_splits"], 1):
            report.append(f"\n  {i}. {suggestion['suggested_filename']}")
            if suggestion["type"] == "class_group":
                report.append(f"     Classi: {', '.join(suggestion['items'])}")
                report.append(f"     Righe totali: {suggestion['total_lines']}")
            elif suggestion["type"] == "large_class":
                report.append(f"     Classe grande: {suggestion['name']} ({suggestion['lines']} righe)")
                report.append(f"     {suggestion['recommendation']}")
            else:
                report.append(f"     Righe: {suggestion['start_line']}-{suggestion['end_line']}")
        
        return "\n".join(report)
    
    def save_analysis(self, output_path: Optional[str] = None):
        """Salva l'analisi in formato JSON."""
        if output_path is None:
            output_path = f"{self.filepath}_analysis.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis, f, indent=2, ensure_ascii=False)
        
        print(f"Analisi salvata in: {output_path}")
    
    def run(self) -> Dict[str, Any]:
        """Esegue l'analisi completa."""
        print(f"Analizzando il file: {self.filepath}")
        
        self.read_file()
        print(f"File letto: {self.analysis['file_info']['total_lines']} righe")
        
        # Determina il tipo di file e analizza
        if self.filepath.endswith('.py'):
            print("Analizzando struttura Python...")
            self.analyze_python_structure()
        else:
            print("Analizzando struttura generica...")
            self.analyze_generic_structure()
        
        print("Calcolando metriche...")
        self.analyze_complexity()
        
        print("Generando suggerimenti per la divisione...")
        self.find_logical_splits()
        
        # Genera e mostra il report
        report = self.generate_report()
        print("\n" + report)
        
        # Salva l'analisi
        self.save_analysis()
        
        return self.analysis


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python analyze_file.py <percorso_file>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Errore: Il file '{filepath}' non esiste")
        sys.exit(1)
    
    analyzer = CodeAnalyzer(filepath)
    analyzer.run()


if __name__ == "__main__":
    main()