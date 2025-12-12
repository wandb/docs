#!/usr/bin/env python3
"""
Generate the CodeSnippet.jsx component with imports for all Python code examples.
"""

from pathlib import Path


def generate_component(py_files):
    """Generate the CodeSnippet.jsx component content."""
    
    # Generate import statements
    imports = []
    mapping = []
    
    for py_file in sorted(py_files):
        # Convert to component name (PascalCase)
        component_name = ''.join(word.capitalize() for word in py_file.stem.replace('-', '_').split('_'))
        mdx_file = py_file.with_suffix('.mdx').name
        imports.append(f"import {component_name} from '/snippets/en/_includes/code-examples/{mdx_file}';")
        mapping.append(f"  '{py_file.name}': {component_name},")
    
    imports_str = '\n'.join(imports)
    mapping_str = '\n'.join(mapping)
    
    component = f"""/**
 * CodeSnippet component for including Python code examples by filename
 * Usage: <CodeSnippet file="artifact_create.py" />
 * 
 * This component consolidates all code example imports in one place,
 * allowing cheat sheet pages to reference snippets by filename without
 * needing individual import statements.
 * 
 * Note: Python files are kept as source of truth, with MDX wrappers
 * generated for compatibility with Mintlify's import system.
 * 
 * AUTO-GENERATED: Do not edit manually. Run sync_code_examples.sh to regenerate.
 */

import React from 'react';

// Import all MDX-wrapped code examples
{imports_str}

// Map filenames to imported content
const snippets = {{
{mapping_str}
}};

export const CodeSnippet = ({{ file }}) => {{
  const Component = snippets[file];
  
  if (!Component) {{
    return (
      <div style={{{{ padding: '1rem', background: '#fee', border: '1px solid #fcc', borderRadius: '4px' }}}}>
        <p style={{{{ margin: 0, color: '#c00' }}}}>Code snippet not found: {{file}}</p>
      </div>
    );
  }}
  
  return <Component />;
}};

export default CodeSnippet;
"""
    
    return component


def main():
    """Generate the CodeSnippet component."""
    script_dir = Path(__file__).parent
    docs_root = script_dir.parent
    snippets_dir = docs_root / 'snippets' / 'en' / '_includes' / 'code-examples'
    output_file = docs_root / 'snippets' / 'CodeSnippet.jsx'
    
    # Find all Python files
    py_files = list(snippets_dir.glob('*.py'))
    
    if not py_files:
        print("   ⚠ No Python files found in code-examples directory")
        return 1
    
    # Generate component
    component_content = generate_component(py_files)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(component_content)
    
    print(f"   ✓ Generated CodeSnippet component with {len(py_files)} imports")
    
    return 0


if __name__ == "__main__":
    exit(main())
