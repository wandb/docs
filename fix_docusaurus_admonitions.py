#!/usr/bin/env python3
"""
Fix Docusaurus-style admonitions to Mintlify callout format.
Converts :::note, :::tip, :::warning, :::danger, :::info to Mintlify <Note>, <Tip>, <Warning>, etc.
"""

import re
from pathlib import Path
import sys

def convert_admonitions(content: str) -> str:
    """Convert Docusaurus admonitions to Mintlify callouts."""
    
    # Map Docusaurus admonition types to Mintlify components
    admonition_map = {
        'note': 'Note',
        'tip': 'Tip',
        'info': 'Info',
        'warning': 'Warning',
        'danger': 'Warning',  # Mintlify doesn't have Danger, use Warning
        'important': 'Warning',  # Use Warning for important too
        'caution': 'Warning'  # Use Warning for caution
    }
    
    # Pattern to match Docusaurus admonitions with optional title
    # Matches :::type, :::type[Title], or :::type Title
    pattern = r'^:::(\w+)(?:\[([^\]]+)\]|\s+(.+?))?$'
    
    lines = content.split('\n')
    result = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        match = re.match(pattern, line)
        
        if match:
            admonition_type = match.group(1).lower()
            title = match.group(2) or match.group(3)  # Title from brackets or after space
            
            if admonition_type in admonition_map:
                component = admonition_map[admonition_type]
                
                # Find the closing :::
                closing_index = None
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() == ':::':
                        closing_index = j
                        break
                
                if closing_index:
                    # Extract content between opening and closing
                    content_lines = lines[i + 1:closing_index]
                    
                    # Build the Mintlify component
                    if title:
                        result.append(f'<{component} title="{title}">')
                    else:
                        result.append(f'<{component}>')
                    
                    # Add the content
                    result.extend(content_lines)
                    
                    # Close the component
                    result.append(f'</{component}>')
                    
                    # Skip to after the closing :::
                    i = closing_index + 1
                else:
                    # No closing found, treat as regular line
                    result.append(line)
                    i += 1
            else:
                # Unknown admonition type, keep as is
                result.append(line)
                i += 1
        else:
            result.append(line)
            i += 1
    
    return '\n'.join(result)

def fix_file(file_path: Path) -> bool:
    """Fix admonitions in a single file."""
    try:
        content = file_path.read_text(encoding='utf-8')
        
        # Check if file has Docusaurus admonitions
        if not re.search(r'^:::\w+', content, re.MULTILINE):
            return False
        
        # Convert admonitions
        fixed_content = convert_admonitions(content)
        
        if fixed_content != content:
            file_path.write_text(fixed_content, encoding='utf-8')
            print(f"✅ Fixed: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    docs_dir = Path.cwd()
    
    # Find all MDX files with Docusaurus admonitions
    mdx_files = list(docs_dir.rglob('*.mdx'))
    
    # Exclude build directories
    mdx_files = [f for f in mdx_files if 'node_modules' not in str(f) 
                  and '.next' not in str(f)
                  and '.git' not in str(f)]
    
    print(f"Checking {len(mdx_files)} MDX files for Docusaurus admonitions...")
    
    fixed_count = 0
    for file_path in mdx_files:
        if fix_file(file_path):
            fixed_count += 1
    
    print(f"\n{'='*60}")
    if fixed_count > 0:
        print(f"✅ Fixed {fixed_count} files with Docusaurus admonitions")
    else:
        print("✅ No Docusaurus admonitions found to fix")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
