#!/usr/bin/env python3
"""
Check for common migration issues in Mintlify documentation.
Detects Hugo shortcodes, Docusaurus imports, and other legacy syntax.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys

class MigrationChecker:
    def __init__(self, docs_dir: Path = Path.cwd()):
        self.docs_dir = docs_dir
        self.issues = []
        
        # Define patterns to check for
        self.patterns = {
            'hugo_shortcodes': [
                (r'\{\{<\s*(\w+).*?>\}\}', 'Hugo shortcode: {{< %s >}}'),
                (r'\{\{%\s*(\w+).*?%\}\}', 'Hugo shortcode: {{% %s %}}'),
                (r'\{\{<\s*/(\w+)\s*>\}\}', 'Hugo closing shortcode: {{< /%s >}}'),
                (r'\{\{%\s*/(\w+)\s*%\}\}', 'Hugo closing shortcode: {{% /%s %}}'),
            ],
            'docusaurus_imports': [
                (r'^import\s+(\w+)\s+from\s+[\'"]@(\w+)[/\w]*[\'"];?$', 'Docusaurus import: import %s from "@%s"'),
                (r'^import\s+.*\s+from\s+[\'"]@site/.*[\'"];?$', 'Docusaurus @site import'),
                (r'^import\s+.*\s+from\s+[\'"]@theme/.*[\'"];?$', 'Docusaurus @theme import'),
                (r'^import\s+.*\s+from\s+[\'"]@docusaurus/.*[\'"];?$', 'Docusaurus @docusaurus import'),
            ],
            'docusaurus_components': [
                (r'<Tabs\s+', 'Docusaurus Tabs component'),
                (r'<TabItem\s+', 'Docusaurus TabItem component'),
                (r'<CodeBlock\s+', 'Docusaurus CodeBlock component'),
                (r'<Admonition\s+', 'Docusaurus Admonition component'),
                (r':::(\w+)', 'Docusaurus admonition syntax (:::%s)'),
            ],
            'hugo_frontmatter': [
                (r'^weight:\s*\d+', 'Hugo weight frontmatter'),
                (r'^draft:\s*(true|false)', 'Hugo draft frontmatter'),
                (r'^menu:', 'Hugo menu frontmatter'),
                (r'^aliases:', 'Hugo aliases frontmatter (might need redirect)'),
            ],
            'broken_links': [
                (r'\[([^\]]+)\]\((/ref/[^)]+)\)', 'Potential broken Hugo ref link'),
                (r'\{\{<\s*ref\s+"([^"]+)"\s*>\}\}', 'Hugo ref shortcode link'),
                (r'\{\{<\s*relref\s+"([^"]+)"\s*>\}\}', 'Hugo relref shortcode link'),
            ],
            'html_issues': [
                (r'<h[1-6]\s+id="[^"]+">.*?<code>.*?</code>.*?</h[1-6]>', 'HTML heading with embedded code tags'),
                (r'^\|\s*\w+\s*\|\s*\|', 'Malformed table with empty column header'),
                (r'### .+\{#[^}]+\}', 'Markdown heading with {#id} anchor (not Mintlify compatible)'),
            ],
            'raw_html': [
                (r'<style\b[^>]*>.*?</style>', 'Raw style tags'),
                (r'<script\b[^>]*>.*?</script>', 'Raw script tags'),
                (r'<iframe\b[^>]*>.*?</iframe>', 'Raw iframe tags'),
            ],
            'legacy_syntax': [
                (r'\$\$.*?\$\$', 'LaTeX math blocks (may need conversion)'),
                (r'```mermaid', 'Mermaid diagram (may need special handling)'),
                (r'```plantuml', 'PlantUML diagram (may need special handling)'),
            ]
        }
    
    def check_file(self, file_path: Path) -> List[Dict]:
        """Check a single file for migration issues."""
        file_issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            # Check frontmatter section separately
            in_frontmatter = False
            frontmatter_lines = []
            content_lines = []
            
            for i, line in enumerate(lines):
                if i == 0 and line.strip() == '---':
                    in_frontmatter = True
                elif in_frontmatter and line.strip() == '---':
                    in_frontmatter = False
                    continue
                    
                if in_frontmatter:
                    frontmatter_lines.append((i + 1, line))
                else:
                    content_lines.append((i + 1, line))
            
            # Check frontmatter patterns
            for line_num, line in frontmatter_lines:
                for pattern, description in self.patterns['hugo_frontmatter']:
                    if re.search(pattern, line, re.MULTILINE):
                        file_issues.append({
                            'file': str(file_path.relative_to(self.docs_dir)),
                            'line': line_num,
                            'issue': description,
                            'content': line.strip()
                        })
            
            # Check content patterns
            for category, patterns in self.patterns.items():
                if category == 'hugo_frontmatter':
                    continue
                    
                for pattern, description in patterns:
                    # For multiline patterns
                    if category in ['raw_html', 'html_issues']:
                        matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            try:
                                issue_desc = description % match.groups() if match.groups() else description
                            except (TypeError, ValueError):
                                issue_desc = description
                            file_issues.append({
                                'file': str(file_path.relative_to(self.docs_dir)),
                                'line': line_num,
                                'issue': issue_desc,
                                'content': match.group(0)[:100] + ('...' if len(match.group(0)) > 100 else '')
                            })
                    else:
                        # For single line patterns
                        for line_num, line in content_lines:
                            matches = re.finditer(pattern, line)
                            for match in matches:
                                try:
                                    issue_desc = description % match.groups() if match.groups() else description
                                except (TypeError, ValueError):
                                    issue_desc = description
                                file_issues.append({
                                    'file': str(file_path.relative_to(self.docs_dir)),
                                    'line': line_num,
                                    'issue': issue_desc,
                                    'content': line.strip()[:100]
                                })
        
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
        
        return file_issues
    
    def check_all_files(self) -> Dict[str, List[Dict]]:
        """Check all MDX/MD files in the documentation."""
        all_issues = {}
        
        # Find all MDX and MD files
        mdx_files = list(self.docs_dir.rglob('*.mdx'))
        md_files = list(self.docs_dir.rglob('*.md'))
        all_files = mdx_files + md_files
        
        # Exclude node_modules and other build directories
        all_files = [f for f in all_files if 'node_modules' not in str(f) 
                     and '.next' not in str(f)
                     and '.git' not in str(f)
                     and 'venv' not in str(f)]
        
        print(f"Checking {len(all_files)} files for migration issues...")
        
        for file_path in all_files:
            issues = self.check_file(file_path)
            if issues:
                all_issues[str(file_path.relative_to(self.docs_dir))] = issues
        
        return all_issues
    
    def generate_report(self, issues: Dict[str, List[Dict]]) -> str:
        """Generate a markdown report of all issues found."""
        report = []
        report.append("# Migration Issues Report\n")
        report.append(f"Checked documentation in: {self.docs_dir}\n")
        
        if not issues:
            report.append("\n✅ **No migration issues found!**\n")
            return '\n'.join(report)
        
        # Group issues by type
        issues_by_type = {}
        total_issues = 0
        
        for file_path, file_issues in issues.items():
            for issue in file_issues:
                issue_type = issue['issue'].split(':')[0] if ':' in issue['issue'] else issue['issue']
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append({
                    'file': file_path,
                    'line': issue['line'],
                    'content': issue['content'],
                    'full_issue': issue['issue']
                })
                total_issues += 1
        
        report.append(f"\n## Summary\n")
        report.append(f"- **Total issues found:** {total_issues}")
        report.append(f"- **Files with issues:** {len(issues)}")
        report.append(f"- **Types of issues:** {len(issues_by_type)}\n")
        
        # List issue types and counts
        report.append("### Issue Types\n")
        for issue_type, instances in sorted(issues_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            report.append(f"- **{issue_type}**: {len(instances)} instances")
        
        # Detailed issues by type
        report.append("\n## Detailed Issues by Type\n")
        
        for issue_type, instances in sorted(issues_by_type.items(), key=lambda x: len(x[1]), reverse=True):
            report.append(f"\n### {issue_type} ({len(instances)} instances)\n")
            
            # Show first 10 instances
            for i, instance in enumerate(instances[:10]):
                report.append(f"{i+1}. `{instance['file']}:{instance['line']}`")
                report.append(f"   ```")
                report.append(f"   {instance['content']}")
                report.append(f"   ```")
            
            if len(instances) > 10:
                report.append(f"\n   *... and {len(instances) - 10} more instances*")
        
        # Files with most issues
        report.append("\n## Files with Most Issues\n")
        files_sorted = sorted(issues.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        
        for file_path, file_issues in files_sorted:
            report.append(f"- `{file_path}`: {len(file_issues)} issues")
        
        return '\n'.join(report)
    
    def save_json_report(self, issues: Dict[str, List[Dict]], output_file: str = "migration_issues.json"):
        """Save issues to a JSON file for programmatic processing."""
        with open(output_file, 'w') as f:
            json.dump(issues, f, indent=2)
        print(f"JSON report saved to {output_file}")

def main():
    # Check if we're in the docs directory
    docs_dir = Path.cwd()
    if not (docs_dir / 'docs.json').exists() and not (docs_dir / 'mint.json').exists():
        print("Warning: No docs.json or mint.json found. Are you in the Mintlify docs directory?")
    
    checker = MigrationChecker(docs_dir)
    issues = checker.check_all_files()
    
    # Generate and save reports
    report = checker.generate_report(issues)
    
    # Save markdown report
    report_file = docs_dir / "migration_issues_report.md"
    report_file.write_text(report)
    print(f"\n📄 Markdown report saved to {report_file}")
    
    # Save JSON report
    checker.save_json_report(issues)
    
    # Print summary to console
    print("\n" + "="*60)
    if issues:
        total = sum(len(file_issues) for file_issues in issues.values())
        print(f"⚠️  Found {total} potential issues in {len(issues)} files")
        print(f"Check {report_file} for details")
    else:
        print("✅ No migration issues found!")
    
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main())
