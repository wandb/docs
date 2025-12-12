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
        var_name = py_file.stem.replace('-', '_')
        imports.append(f"import {var_name} from './en/_includes/code-examples/{py_file.name}?raw';")
        mapping.append(f"  '{py_file.name}': {var_name},")
    
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
 * AUTO-GENERATED: Do not edit manually. Run sync_code_examples.sh to regenerate.
 */

// Import all Python code examples as raw text
{imports_str}

// Map filenames to imported content
const snippets = {{
{mapping_str}
}};

export const CodeSnippet = ({{ file }}) => {{
  const content = snippets[file];
  
  if (!content) {{
    return (
      <div style={{ padding: '1rem', background: '#fee', border: '1px solid #fcc', borderRadius: '4px' }}>
        <p style={{ margin: 0, color: '#c00' }}>Code snippet not found: {{file}}</p>
      </div>
    );
  }}
  
  return (
    <pre>
      <code className="language-python">{{content}}</code>
    </pre>
  );
}};
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
