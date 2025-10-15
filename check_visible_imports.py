#!/usr/bin/env python3
"""
Check for import statements that will be visible in rendered MDX output.
This version better handles code blocks and MDX-specific imports.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any

def is_in_code_block(lines: List[str], line_num: int) -> bool:
    """Check if a line is inside a code block."""
    in_code_block = False
    for i in range(line_num):
        if lines[i].strip().startswith('```'):
            in_code_block = not in_code_block
    return in_code_block

def is_mdx_import(line: str) -> bool:
    """Check if this is a valid MDX import (for components)."""
    # MDX imports typically look like:
    # import {Component} from "/path/to/component"
    # import Component from "/path/to/component"
    mdx_pattern = r'^import\s+[\{\w][^;]*\s+from\s+["\']/'
    return bool(re.match(mdx_pattern, line.strip()))

def check_file_for_visible_imports(file_path: Path) -> List[Dict[str, Any]]:
    """Check a single file for potentially visible imports."""
    issues = []
    
    try:
        content = file_path.read_text()
        lines = content.splitlines()
        
        # Track frontmatter
        in_frontmatter = False
        frontmatter_end = -1
        
        for i, line in enumerate(lines):
            if line.strip() == '---':
                if not in_frontmatter and i == 0:
                    in_frontmatter = True
                elif in_frontmatter:
                    in_frontmatter = False
                    frontmatter_end = i
                    break
        
        # Check each line
        for line_num, line in enumerate(lines):
            # Skip frontmatter
            if line_num <= frontmatter_end:
                continue
            
            # Skip if in code block
            if is_in_code_block(lines, line_num):
                continue
            
            # Check for import statements
            if line.strip().startswith('import '):
                # Check if it's an MDX import (should be right after frontmatter)
                if is_mdx_import(line):
                    # MDX imports should be immediately after frontmatter
                    if frontmatter_end >= 0 and line_num > frontmatter_end + 5:
                        # Import is too far from frontmatter, might render
                        issues.append({
                            'file': str(file_path.relative_to(Path('/Users/matt.linville/docs'))),
                            'line': line_num + 1,
                            'type': 'mdx_import_late',
                            'content': line.strip()[:80],
                            'note': 'MDX import appears too late in file (should be right after frontmatter)'
                        })
                else:
                    # This looks like a Python/JS import outside code block
                    issues.append({
                        'file': str(file_path.relative_to(Path('/Users/matt.linville/docs'))),
                        'line': line_num + 1,
                        'type': 'bare_import',
                        'content': line.strip()[:80],
                        'note': 'Import statement outside code block - will render as text'
                    })
    
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
    
    return issues

def main():
    docs_dir = Path('/Users/matt.linville/docs')
    
    print("🔍 Checking for import statements that will be visible in rendered output...")
    print("=" * 70)
    
    # Find all MDX files
    mdx_files = list(docs_dir.rglob('*.mdx'))
    print(f"Scanning {len(mdx_files)} MDX files...\n")
    
    all_issues = []
    
    for file_path in mdx_files:
        issues = check_file_for_visible_imports(file_path)
        all_issues.extend(issues)
    
    if all_issues:
        print(f"⚠️  Found {len(all_issues)} potential issues:\n")
        
        # Group by type
        by_type = {}
        for issue in all_issues:
            issue_type = issue['type']
            if issue_type not in by_type:
                by_type[issue_type] = []
            by_type[issue_type].append(issue)
        
        for issue_type, issues in by_type.items():
            print(f"\n{issue_type.upper().replace('_', ' ')} ({len(issues)} issues):")
            print("-" * 50)
            
            for issue in issues[:5]:  # Show first 5 of each type
                print(f"📄 {issue['file']}:{issue['line']}")
                print(f"   {issue['content']}")
                print(f"   ⚠️  {issue['note']}\n")
            
            if len(issues) > 5:
                print(f"   ... and {len(issues) - 5} more\n")
    else:
        print("✅ No visible import issues detected!")
    
    # Quick visual check - let's also look for telltale signs
    print("\n" + "=" * 70)
    print("🔍 Quick pattern check for import visibility symptoms...")
    
    # These patterns in the rendered HTML would indicate imports are showing
    symptom_patterns = [
        (r'<p>\s*import\s+', "Import statement wrapped in paragraph tag"),
        (r'<code>import\s+.*from</code>', "Import rendered as inline code"),
        (r'>\s*import\s+{.*}\s*from\s*<', "Import between HTML tags"),
    ]
    
    # We'd need to actually render and check, but we can look for these in source
    print("\n💡 To verify imports are processed correctly:")
    print("1. Run: npx mintlify dev")
    print("2. Check pages with MDX imports (e.g., /models/automations)")
    print("3. View page source and search for 'import' - should NOT see import statements")
    print("4. If you see imports in the rendered output, they need to be fixed\n")
    
    return 0 if not all_issues else 1

if __name__ == "__main__":
    sys.exit(main())
