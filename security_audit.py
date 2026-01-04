#!/usr/bin/env python3
"""
Security Integrity Verification Tool
Scans the codebase for path security compliance and generates a comprehensive report
"""
import ast
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SecurityScanner:
    """Scans Python files for security compliance"""
    
    # Patterns that indicate file operations needing validation
    FILE_OPERATIONS = {
        'open', 'os.path.join', 'os.makedirs', 'os.remove', 'os.unlink',
        'shutil.copy', 'shutil.copy2', 'shutil.move', 'shutil.rmtree',
        'Path.mkdir', 'Path.write_text', 'Path.write_bytes', 'Path.unlink',
        'Path.rmdir', 'subprocess.run', 'subprocess.call', 'subprocess.Popen'
    }
    
    # Security functions that should be used
    SECURITY_FUNCTIONS = {
        'safe_path', 'safe_filename', 'safe_join_path', 'safe_resolve_path',
        'is_path_safe', 'validate_directory_path', 'safe_join'
    }
    
    # Required security patterns
    ALLOWED_ROOTS_PATTERN = re.compile(r'ALLOWED_ROOTS\s*=\s*\[')
    SAFE_PATH_PATTERN = re.compile(r'safe_path\s*\(')
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {
            'compliant': [],
            'needs_review': [],
            'vulnerable': [],
            'statistics': defaultdict(int)
        }
        
    def scan_file(self, file_path: Path) -> Dict:
        """Scan a single Python file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content, filename=str(file_path))
            
            file_result = {
                'path': file_path.relative_to(self.project_root),
                'file_operations': [],
                'security_calls': [],
                'has_allowed_roots': bool(self.ALLOWED_ROOTS_PATTERN.search(content)),
                'issues': [],
                'status': 'compliant'
            }
            
            # AST analysis
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    self._check_call(node, file_result, content)
                    
            # Check for file operations without security
            if file_result['file_operations'] and not file_result['security_calls']:
                file_result['status'] = 'vulnerable'
                file_result['issues'].append(
                    f"Found {len(file_result['file_operations'])} file operations without security validation"
                )
            elif file_result['file_operations']:
                file_result['status'] = 'needs_review'
                
            return file_result
            
        except Exception as e:
            return {
                'path': file_path.relative_to(self.project_root),
                'error': str(e),
                'status': 'error'
            }
            
    def _check_call(self, node: ast.Call, file_result: Dict, content: str):
        """Check function calls for security relevance"""
        func_name = self._get_call_name(node)
        
        if func_name:
            # Check for file operations
            for op in self.FILE_OPERATIONS:
                if op in func_name:
                    line_num = node.lineno if hasattr(node, 'lineno') else 0
                    file_result['file_operations'].append({
                        'operation': func_name,
                        'line': line_num
                    })
                    self.results['statistics']['total_file_operations'] += 1
                    
            # Check for security function usage
            for sec_func in self.SECURITY_FUNCTIONS:
                if sec_func in func_name:
                    line_num = node.lineno if hasattr(node, 'lineno') else 0
                    file_result['security_calls'].append({
                        'function': func_name,
                        'line': line_num
                    })
                    self.results['statistics']['total_security_calls'] += 1
                    
    def _get_call_name(self, node: ast.Call) -> str:
        """Extract function call name from AST node"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.insert(0, current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.insert(0, current.id)
            return '.'.join(parts)
        return ''
        
    def scan_directory(self, directory: Path, exclude_patterns: List[str] = None):
        """Scan all Python files in a directory"""
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'node_modules', 'venv', '.venv']
            
        for file_path in directory.rglob('*.py'):
            # Skip excluded directories
            if any(pattern in str(file_path) for pattern in exclude_patterns):
                continue
                
            result = self.scan_file(file_path)
            
            if result.get('status') == 'compliant':
                self.results['compliant'].append(result)
            elif result.get('status') == 'needs_review':
                self.results['needs_review'].append(result)
            elif result.get('status') == 'vulnerable':
                self.results['vulnerable'].append(result)
                
            self.results['statistics']['total_files'] += 1
            
    def generate_report(self) -> str:
        """Generate a comprehensive security report"""
        report = []
        
        # Header
        report.append(f"{Colors.BOLD}{Colors.HEADER}")
        report.append("=" * 80)
        report.append("  SECURITY INTEGRITY VERIFICATION REPORT")
        report.append("=" * 80)
        report.append(f"{Colors.ENDC}\n")
        
        # Statistics
        stats = self.results['statistics']
        report.append(f"{Colors.BOLD}?? Statistics:{Colors.ENDC}")
        report.append(f"  Total Files Scanned: {stats.get('total_files', 0)}")
        report.append(f"  Total File Operations: {stats.get('total_file_operations', 0)}")
        report.append(f"  Total Security Calls: {stats.get('total_security_calls', 0)}")
        report.append("")
        
        # Security Status
        compliant_count = len(self.results['compliant'])
        review_count = len(self.results['needs_review'])
        vulnerable_count = len(self.results['vulnerable'])
        
        total = compliant_count + review_count + vulnerable_count
        if total > 0:
            compliance_rate = (compliant_count / total) * 100
        else:
            compliance_rate = 0
            
        report.append(f"{Colors.BOLD}?? Security Status:{Colors.ENDC}")
        report.append(f"  {Colors.OKGREEN}? Compliant:{Colors.ENDC} {compliant_count} files")
        report.append(f"  {Colors.WARNING}? Needs Review:{Colors.ENDC} {review_count} files")
        report.append(f"  {Colors.FAIL}? Vulnerable:{Colors.ENDC} {vulnerable_count} files")
        report.append(f"  Compliance Rate: {compliance_rate:.1f}%")
        report.append("")
        
        # Vulnerable Files Detail
        if vulnerable_count > 0:
            report.append(f"{Colors.FAIL}{Colors.BOLD}?? VULNERABLE FILES:{Colors.ENDC}")
            for result in self.results['vulnerable']:
                report.append(f"\n  {Colors.FAIL}? {result['path']}{Colors.ENDC}")
                for issue in result.get('issues', []):
                    report.append(f"    - {issue}")
                if result.get('file_operations'):
                    report.append(f"    File operations at lines:")
                    for op in result['file_operations'][:5]:  # Show first 5
                        report.append(f"      • {op['operation']} (line {op['line']})")
                report.append("")
                
        # Files Needing Review
        if review_count > 0 and review_count <= 10:  # Only show if manageable
            report.append(f"{Colors.WARNING}{Colors.BOLD}? FILES NEEDING REVIEW:{Colors.ENDC}")
            for result in self.results['needs_review']:
                report.append(f"\n  {Colors.WARNING}? {result['path']}{Colors.ENDC}")
                report.append(f"    File operations: {len(result.get('file_operations', []))}")
                report.append(f"    Security calls: {len(result.get('security_calls', []))}")
                
        # Recommendations
        report.append(f"\n{Colors.BOLD}?? Recommendations:{Colors.ENDC}")
        if vulnerable_count > 0:
            report.append(f"  {Colors.FAIL}• CRITICAL: Fix {vulnerable_count} vulnerable files immediately{Colors.ENDC}")
            report.append("    - Add ALLOWED_ROOTS definition")
            report.append("    - Wrap file operations with safe_path() validation")
            report.append("    - Review subprocess execution paths")
        if review_count > 0:
            report.append(f"  {Colors.WARNING}• Review {review_count} files for proper security usage{Colors.ENDC}")
            report.append("    - Verify all file operations use security helpers")
            report.append("    - Check that safe_path() is called before file operations")
        if compliance_rate >= 95:
            report.append(f"  {Colors.OKGREEN}? Excellent security posture! Maintain current practices.{Colors.ENDC}")
        elif compliance_rate >= 80:
            report.append(f"  {Colors.OKGREEN}? Good security posture. Address remaining issues.{Colors.ENDC}")
        else:
            report.append(f"  {Colors.FAIL}• Security needs immediate attention.{Colors.ENDC}")
            
        report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append(f"{Colors.BOLD}Report Complete{Colors.ENDC}")
        report.append("=" * 80)
        
        return '\n'.join(report)
        
    def save_report(self, output_path: Path):
        """Save report to file (without ANSI codes)"""
        # Remove ANSI codes for file output
        report = self.generate_report()
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        clean_report = ansi_escape.sub('', report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(clean_report)
            f.write("\n\n")
            
            # Detailed findings
            f.write("=" * 80 + "\n")
            f.write("DETAILED FINDINGS\n")
            f.write("=" * 80 + "\n\n")
            
            for category, results in [
                ('VULNERABLE', self.results['vulnerable']),
                ('NEEDS REVIEW', self.results['needs_review']),
                ('COMPLIANT', self.results['compliant'][:20])  # Sample
            ]:
                f.write(f"\n{category} FILES:\n")
                f.write("-" * 80 + "\n")
                for result in results:
                    f.write(f"\nFile: {result['path']}\n")
                    f.write(f"Status: {result.get('status', 'unknown')}\n")
                    if result.get('has_allowed_roots'):
                        f.write("Has ALLOWED_ROOTS: Yes\n")
                    if result.get('issues'):
                        f.write("Issues:\n")
                        for issue in result['issues']:
                            f.write(f"  - {issue}\n")
                    if result.get('file_operations'):
                        f.write(f"File Operations ({len(result['file_operations'])}):\n")
                        for op in result['file_operations'][:10]:
                            f.write(f"  - {op['operation']} (line {op['line']})\n")
                    if result.get('security_calls'):
                        f.write(f"Security Calls ({len(result['security_calls'])}):\n")
                        for call in result['security_calls'][:10]:
                            f.write(f"  - {call['function']} (line {call['line']})\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Security Integrity Verification Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Scan entire project
  python security_audit.py
  
  # Scan specific directory
  python security_audit.py --dir webapp/parser
  
  # Save detailed report
  python security_audit.py --output security_report.txt
  
  # Quiet mode (only show summary)
  python security_audit.py --quiet
        '''
    )
    
    parser.add_argument(
        '--dir',
        type=str,
        help='Directory to scan (default: webapp/parser)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save detailed report to file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Only show summary (no detailed output)'
    )
    
    args = parser.parse_args()
    
    # Determine project root and scan directory
    script_dir = Path(__file__).parent
    project_root = script_dir
    
    if args.dir:
        scan_dir = Path(args.dir)
        if not scan_dir.is_absolute():
            scan_dir = project_root / scan_dir
    else:
        scan_dir = project_root / "webapp" / "parser"
        
    if not scan_dir.exists():
        print(f"{Colors.FAIL}Error: Directory not found: {scan_dir}{Colors.ENDC}")
        return 1
        
    print(f"{Colors.BOLD}Scanning: {scan_dir}{Colors.ENDC}")
    print(f"Project root: {project_root}\n")
    
    # Run scan
    scanner = SecurityScanner(project_root)
    scanner.scan_directory(scan_dir)
    
    # Generate report
    report = scanner.generate_report()
    print(report)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        scanner.save_report(output_path)
        print(f"\n{Colors.OKGREEN}? Detailed report saved to: {output_path}{Colors.ENDC}")
        
    # Exit code based on vulnerabilities
    if scanner.results['vulnerable']:
        return 1
    return 0


if __name__ == '__main__':
    exit(main())
