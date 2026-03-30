#!/usr/bin/env python3
"""Audit script to validate all state handler implementations."""

import os
import ast
from pathlib import Path
from typing import Tuple, List


class HandlerAuditor:
    def __init__(self, handlers_dir: str):
        self.handlers_dir = Path(handlers_dir)
        self.implemented = []
        self.missing = []

    def audit_file(self, filepath: Path) -> Tuple[str, bool, str, List[str]]:
        """Check if handler inherits from SimpleTableHandler or overrides extract_tables()."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
            
            has_extract_tables = False
            class_name = "Unknown"
            bases = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    
                    # Extract base class names
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            bases.append(base.id)
                        elif isinstance(base, ast.Attribute):
                            bases.append(base.attr)
                    
                    # Check if defines extract_tables
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == 'extract_tables':
                            has_extract_tables = True
                            break
                    
                    # SimpleTableHandler has extract_tables built-in
                    if not has_extract_tables and 'SimpleTableHandler' in bases:
                        has_extract_tables = True
            
            return str(filepath.relative_to(self.handlers_dir)), has_extract_tables, class_name, bases
        except Exception as e:
            return str(filepath.relative_to(self.handlers_dir)), False, "ERROR", ["parse_error"]

    def run(self) -> None:
        """Run audit on all state handler files."""
        print(f"\n🔍 Auditing state handlers\n")
        print("=" * 85)
        
        handler_files = []
        for root, dirs, files in os.walk(self.handlers_dir):
            for file in files:
                if file.endswith('.py') and file != '__init__.py':
                    filepath = Path(root) / file
                    if 'shared' not in str(filepath):
                        handler_files.append(filepath)
        
        handler_files.sort()
        print(f"Found {len(handler_files)} handler files\n")
        
        for filepath in handler_files:
            rel_path, has_method, class_name, bases = self.audit_file(filepath)
            status = "✅" if has_method else "❌"
            bases_str = ", ".join(bases) if bases else "None"
            
            print(f"{status}  {rel_path:50} | {class_name:25} | {bases_str}")
            
            if has_method:
                self.implemented.append(rel_path)
            else:
                self.missing.append(rel_path)
        
        self._print_summary()
    
    def _print_summary(self) -> None:
        """Print summary."""
        print("\n" + "=" * 85)
        print(f"\n📊 AUDIT SUMMARY: {len(self.implemented)} OK | {len(self.missing)} BROKEN\n")
        
        if not self.missing:
            print("✅ SUCCESS: All handlers are properly configured!")
            print("   All inherit from SimpleTableHandler (which provides extract_tables())")
        else:
            print(f"❌ CRITICAL: {len(self.missing)} handlers need fixes:")
            for h in self.missing[:10]:
                print(f"   - {h}")
            if len(self.missing) > 10:
                print(f"   ... and {len(self.missing)-10} more")
        
        print("\n" + "=" * 85)


if __name__ == "__main__":
    handlers_dir = Path(__file__).parent.parent / "webapp" / "parser" / "handlers" / "states"
    auditor = HandlerAuditor(str(handlers_dir))
    auditor.run()
