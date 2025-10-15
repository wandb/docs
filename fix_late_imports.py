#!/usr/bin/env python3
"""
Fix MDX imports that appear too late in the file (after content has started).
Moves them to right after the frontmatter where they should be.
"""

import re
from pathlib import Path

def fix_late_import(file_path: Path) -> bool:
    """
    Fix late import statements in an MDX file.
    Returns True if file was modified.
    """
    try:
        content = file_path.read_text()
        lines = content.splitlines()
        
        # Find frontmatter end
        frontmatter_end = -1
        in_frontmatter = False
        
        for i, line in enumerate(lines):
            if line.strip() == '---':
                if not in_frontmatter and i == 0:
                    in_frontmatter = True
                elif in_frontmatter:
                    frontmatter_end = i
                    break
        
        if frontmatter_end < 0:
            print(f"  ⚠️  No frontmatter found in {file_path}")
            return False
        
        # Find import statements that are too late
        imports_to_move = []
        new_lines = []
        
        for i, line in enumerate(lines):
            # Check if this is an MDX import that's too late
            if (i > frontmatter_end + 5 and 
                line.strip().startswith('import ') and 
                ' from ' in line and 
                '/' in line):
                
                # This is a late import, mark it for moving
                imports_to_move.append(line)
                # Skip this line in the output
                continue
            
            new_lines.append(line)
        
        if not imports_to_move:
            return False
        
        # Insert imports right after frontmatter
        result_lines = []
        for i, line in enumerate(new_lines):
            result_lines.append(line)
            if i == frontmatter_end:
                # Add imports here
                for import_line in imports_to_move:
                    result_lines.append(import_line)
        
        # Write back
        new_content = '\n'.join(result_lines)
        file_path.write_text(new_content)
        
        print(f"  ✅ Fixed {file_path.name} - moved {len(imports_to_move)} import(s)")
        return True
        
    except Exception as e:
        print(f"  ❌ Error fixing {file_path}: {e}")
        return False

def main():
    # Files identified as having late imports
    files_to_fix = [
        'release-notes/0.72.mdx',
        'release-notes/0.73.mdx',
        'release-notes/0.71.mdx',
        'release-notes/0.70.mdx',
        'release-notes/0.69.0.mdx'
    ]
    
    docs_dir = Path('/Users/matt.linville/docs')
    
    print("🔧 Fixing late MDX imports in release notes...")
    print("=" * 50)
    
    fixed_count = 0
    for file_path in files_to_fix:
        full_path = docs_dir / file_path
        if full_path.exists():
            if fix_late_import(full_path):
                fixed_count += 1
        else:
            print(f"  ⚠️  File not found: {file_path}")
    
    print("\n" + "=" * 50)
    print(f"✅ Fixed {fixed_count} files")
    
    if fixed_count > 0:
        print("\n💡 Next steps:")
        print("1. Review the changes with: git diff")
        print("2. Test the pages in Mintlify dev")
        print("3. Commit the fixes")

if __name__ == "__main__":
    main()
