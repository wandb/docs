#!/usr/bin/env python3
"""
Check if import statements in MDX files might be rendering as visible text.
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Any

def check_import_visibility(file_path: Path) -> List[Dict[str, Any]]:
    """
    Check if import statements might be visible in the rendered output.
    
    Returns a list of potential issues.
    """
    issues = []
    
    try:
        content = file_path.read_text()
        lines = content.splitlines()
        
        # Track if we're in frontmatter
        in_frontmatter = False
        frontmatter_count = 0
        in_code_block = False
        code_fence_pattern = re.compile(r'^```')
        
        for line_num, line in enumerate(lines, 1):
            # Check for frontmatter boundaries
            if line.strip() == '---':
                if frontmatter_count == 0:
                    in_frontmatter = True
                    frontmatter_count = 1
                elif frontmatter_count == 1:
                    in_frontmatter = False
                    frontmatter_count = 2
                continue
            
            # Skip if we're in frontmatter
            if in_frontmatter:
                continue
                
            # Track code blocks
            if code_fence_pattern.match(line):
                in_code_block = not in_code_block
                continue
            
            # Skip if we're in a code block
            if in_code_block:
                continue
            
            # Check for import statements
            if re.match(r'^import\s+', line):
                # Check if it's immediately after frontmatter (which is correct)
                if frontmatter_count == 2 and line_num <= 10:
                    # This is likely a proper MDX import
                    continue
                    
                # Check if there's text before the import (which would make it visible)
                if line_num > 1:
                    prev_line = lines[line_num - 2].strip()
                    # If previous line is not empty and not an import, might be an issue
                    if prev_line and not prev_line.startswith('import'):
                        issues.append({
                            'file': str(file_path.relative_to(Path('/Users/matt.linville/docs'))),
                            'line': line_num,
                            'type': 'import_after_content',
                            'content': line.strip()[:100],
                            'note': 'Import statement appears after content began'
                        })
                
                # Check for imports with incorrect syntax that might render
                if not re.match(r'^import\s+[\{\w].*\s+from\s+["\']', line):
                    issues.append({
                        'file': str(file_path.relative_to(Path('/Users/matt.linville/docs'))),
                        'line': line_num,
                        'type': 'malformed_import',
                        'content': line.strip()[:100],
                        'note': 'Import statement might have incorrect syntax'
                    })
    
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
    
    return issues

def main():
    docs_dir = Path('/Users/matt.linville/docs')
    
    # Find all MDX files
    mdx_files = list(docs_dir.rglob('*.mdx'))
    
    print("🔍 Checking for potentially visible import statements in MDX files...")
    print(f"Found {len(mdx_files)} MDX files to check\n")
    
    all_issues = []
    files_with_imports = 0
    
    for file_path in mdx_files:
        content = file_path.read_text()
        
        # Quick check if file has any imports
        if 'import ' in content:
            files_with_imports += 1
            issues = check_import_visibility(file_path)
            all_issues.extend(issues)
    
    print(f"📊 Summary:")
    print(f"  - Files with import statements: {files_with_imports}")
    print(f"  - Potential visibility issues found: {len(all_issues)}")
    
    if all_issues:
        print("\n⚠️  Potential Issues Found:\n")
        for issue in all_issues:
            print(f"File: {issue['file']}")
            print(f"  Line {issue['line']}: {issue['type']}")
            print(f"  Content: {issue['content']}")
            print(f"  Note: {issue['note']}\n")
    else:
        print("\n✅ No import visibility issues detected!")
    
    # Also check for specific patterns that might indicate rendering issues
    print("\n🔍 Checking for import patterns that might indicate rendering issues...")
    
    # Pattern checks
    patterns_to_check = [
        (r'^\s*import\s+', "Import with leading whitespace (might not be processed)"),
        (r'`import\s+.*from`', "Import inside backticks (will render as code)"),
        (r'>\s*import\s+', "Import in blockquote (will render)"),
        (r'\*\s*import\s+', "Import in list item (will render)"),
    ]
    
    pattern_issues = []
    for file_path in mdx_files:
        content = file_path.read_text()
        lines = content.splitlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern, description in patterns_to_check:
                if re.search(pattern, line):
                    # Skip if in code block
                    if '```' in ''.join(lines[max(0, line_num-3):line_num]):
                        continue
                    pattern_issues.append({
                        'file': str(file_path.relative_to(docs_dir)),
                        'line': line_num,
                        'pattern': description,
                        'content': line.strip()[:100]
                    })
    
    if pattern_issues:
        print(f"\n⚠️  Found {len(pattern_issues)} pattern-based issues:")
        for issue in pattern_issues[:10]:  # Show first 10
            print(f"  {issue['file']}:{issue['line']} - {issue['pattern']}")
            print(f"    {issue['content']}\n")
    else:
        print("✅ No pattern-based import issues found!")
    
    return 0 if not (all_issues or pattern_issues) else 1

if __name__ == "__main__":
    sys.exit(main())
