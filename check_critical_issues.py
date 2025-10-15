#!/usr/bin/env python3
"""
Check for CRITICAL migration issues that will actually break Mintlify rendering.
Focus on issues that cause visible problems in the output.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict
import sys

class CriticalIssueChecker:
    def __init__(self, docs_dir: Path = Path.cwd()):
        self.docs_dir = docs_dir
        
        # Only check for patterns that actually break rendering
        self.critical_patterns = {
            'hugo_shortcodes': [
                (r'\{\{<\s*(\w+).*?>\}\}', 'Hugo shortcode will render as raw text: {{< %s >}}'),
                (r'\{\{%\s*(\w+).*?%\}\}', 'Hugo shortcode will render as raw text: {{% %s %}}'),
            ],
            'docusaurus_components': [
                (r'^import\s+Tabs\s+from\s+[\'"]@theme/Tabs[\'"]', 'Docusaurus Tabs import - will break'),
                (r'^import\s+TabItem\s+from\s+[\'"]@theme/TabItem[\'"]', 'Docusaurus TabItem import - will break'),
                (r'<Tabs\s+', 'Docusaurus Tabs component - won\'t work in Mintlify'),
                (r'<TabItem\s+', 'Docusaurus TabItem component - won\'t work in Mintlify'),
                (r':::(\w+)', 'Docusaurus admonition (:::%s) - will render as text'),
            ],
            'broken_syntax': [
                (r'<h[1-6]\s+id="[^"]+">.*?<code>.*?</code>.*?</h[1-6]>', 'HTML heading with code - may not render correctly'),
                (r'^\|\s*\w+\s*\|\s*\|', 'Table with empty column header - breaks MDX parsing'),
                (r'### .+\{#[^}]+\}', 'Heading with {#id} anchor - not Mintlify compatible'),
            ],
            'missing_images': [
                (r'!\[([^\]]*)\]\((/static/[^)]+)\)', 'Image path to /static/ - likely broken'),
                (r'!\[([^\]]*)\]\((/img/[^)]+)\)', 'Image path to /img/ - likely broken'),
                (r'<img\s+src="(/static/[^"]+)"', 'HTML img with /static/ path - likely broken'),
            ],
            'broken_links': [
                (r'\{\{<\s*ref\s+"([^"]+)"\s*>\}\}', 'Hugo ref link - will show as raw text'),
                (r'\{\{<\s*relref\s+"([^"]+)"\s*>\}\}', 'Hugo relref link - will show as raw text'),
                (r'\[([^\]]+)\]\(\{\{<\s*ref\s+"([^"]+)"\s*>\}\}\)', 'Markdown link with Hugo ref - broken'),
            ]
        }
    
    def check_file(self, file_path: Path) -> List[Dict]:
        """Check a single file for critical issues."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for category, patterns in self.critical_patterns.items():
                for pattern, description in patterns:
                    if category in ['broken_syntax', 'missing_images']:
                        # Check multiline patterns
                        matches = re.finditer(pattern, content, re.DOTALL | re.MULTILINE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            try:
                                issue_desc = description % match.groups() if '%s' in description and match.groups() else description
                            except:
                                issue_desc = description
                            
                            issues.append({
                                'file': str(file_path.relative_to(self.docs_dir)),
                                'line': line_num,
                                'category': category,
                                'issue': issue_desc,
                                'content': match.group(0)[:100] + ('...' if len(match.group(0)) > 100 else '')
                            })
                    else:
                        # Check line by line
                        for i, line in enumerate(lines, 1):
                            matches = re.finditer(pattern, line)
                            for match in matches:
                                try:
                                    issue_desc = description % match.groups()[0] if '%s' in description and match.groups() else description
                                except:
                                    issue_desc = description
                                
                                issues.append({
                                    'file': str(file_path.relative_to(self.docs_dir)),
                                    'line': i,
                                    'category': category,
                                    'issue': issue_desc,
                                    'content': line.strip()[:100]
                                })
        
        except Exception as e:
            print(f"Error checking {file_path}: {e}")
        
        return issues
    
    def check_all_files(self) -> Dict[str, List[Dict]]:
        """Check all MDX/MD files for critical issues."""
        all_issues = {}
        
        # Find all MDX and MD files
        mdx_files = list(self.docs_dir.rglob('*.mdx'))
        md_files = list(self.docs_dir.rglob('*.md'))
        all_files = mdx_files + md_files
        
        # Exclude build directories
        all_files = [f for f in all_files if 'node_modules' not in str(f) 
                     and '.next' not in str(f)
                     and '.git' not in str(f)
                     and 'venv' not in str(f)]
        
        print(f"Checking {len(all_files)} files for CRITICAL migration issues...")
        
        for file_path in all_files:
            issues = self.check_file(file_path)
            if issues:
                all_issues[str(file_path.relative_to(self.docs_dir))] = issues
        
        return all_issues
    
    def generate_report(self, issues: Dict[str, List[Dict]]) -> str:
        """Generate a report of critical issues."""
        report = []
        report.append("# Critical Migration Issues Report\n")
        report.append("These issues will likely cause visible problems in the rendered documentation.\n")
        
        if not issues:
            report.append("\n✅ **No critical issues found!**\n")
            return '\n'.join(report)
        
        # Group by category
        by_category = {}
        for file_path, file_issues in issues.items():
            for issue in file_issues:
                cat = issue['category']
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append({
                    'file': file_path,
                    'line': issue['line'],
                    'issue': issue['issue'],
                    'content': issue['content']
                })
        
        total = sum(len(items) for items in by_category.values())
        
        report.append(f"\n## Summary\n")
        report.append(f"- **Total critical issues:** {total}")
        report.append(f"- **Files affected:** {len(issues)}")
        report.append(f"- **Categories:** {len(by_category)}\n")
        
        # Priority order for categories
        priority = ['hugo_shortcodes', 'docusaurus_components', 'broken_links', 'missing_images', 'broken_syntax']
        
        for cat in priority:
            if cat not in by_category:
                continue
                
            items = by_category[cat]
            report.append(f"\n## {cat.replace('_', ' ').title()} ({len(items)} issues)\n")
            
            # Show first 10 examples
            for i, item in enumerate(items[:10], 1):
                report.append(f"{i}. **{item['file']}:{item['line']}**")
                report.append(f"   - Issue: {item['issue']}")
                report.append(f"   ```")
                report.append(f"   {item['content']}")
                report.append(f"   ```")
            
            if len(items) > 10:
                report.append(f"\n*... and {len(items) - 10} more*")
        
        # Most affected files
        report.append(f"\n## Most Affected Files\n")
        sorted_files = sorted(issues.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        for file_path, file_issues in sorted_files:
            report.append(f"- **{file_path}**: {len(file_issues)} critical issues")
        
        return '\n'.join(report)

def main():
    docs_dir = Path.cwd()
    
    checker = CriticalIssueChecker(docs_dir)
    issues = checker.check_all_files()
    
    # Generate report
    report = checker.generate_report(issues)
    
    # Save report
    report_file = docs_dir / "critical_migration_issues.md"
    report_file.write_text(report)
    print(f"\n📄 Report saved to {report_file}")
    
    # Save JSON
    json_file = docs_dir / "critical_migration_issues.json"
    with open(json_file, 'w') as f:
        json.dump(issues, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    if issues:
        total = sum(len(file_issues) for file_issues in issues.values())
        print(f"⚠️  Found {total} CRITICAL issues in {len(issues)} files")
        print("These issues will likely cause rendering problems!")
    else:
        print("✅ No critical migration issues found!")
    
    return 0 if not issues else 1

if __name__ == "__main__":
    sys.exit(main())
