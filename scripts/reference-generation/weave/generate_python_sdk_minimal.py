#!/usr/bin/env python3
"""
Minimal Python SDK documentation generator for Mintlify.
This avoids lazydocs and its dependencies to prevent license policy violations.
"""

import ast
import inspect
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any


def install_weave(version="latest"):
    """Install Weave package."""
    print(f"Installing Weave (version: {version})...")
    if version == "latest":
        subprocess.run([sys.executable, "-m", "pip", "install", "weave"], check=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", f"weave=={version}"], check=True)


def get_docstring(obj):
    """Extract and format docstring for MDX compatibility."""
    doc = inspect.getdoc(obj)
    if not doc:
        return "No description available."
    
    # Escape curly braces for MDX (they're interpreted as JSX expressions)
    # We need to be careful to only escape braces that aren't in code blocks
    lines = doc.split('\n')
    result_lines = []
    in_code_block = False
    
    for line in lines:
        # Check if this is a code block marker (triple backticks or indented code after ::)
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            result_lines.append(line)
        elif not in_code_block:
            # Escape curly braces outside of code blocks
            # Replace { with \{ and } with \}
            line = line.replace('{', '\\{').replace('}', '\\}')
            result_lines.append(line)
        else:
            # Inside code block, keep as-is
            result_lines.append(line)
    
    return '\n'.join(result_lines)


def get_function_signature(func):
    """Get function signature."""
    try:
        sig = inspect.signature(func)
        return f"{func.__name__}{sig}"
    except:
        return f"{func.__name__}(...)"


def analyze_module(module_path: str, module_name: str) -> Dict[str, Any]:
    """Analyze a Python module and extract documentation."""
    # Import the module
    spec = __import__(module_name, fromlist=[''])
    
    functions = []
    classes = []
    
    for name, obj in inspect.getmembers(spec):
        if name.startswith('_'):
            continue
            
        if inspect.isfunction(obj):
            functions.append({
                'name': name,
                'signature': get_function_signature(obj),
                'docstring': get_docstring(obj),
                'module': module_name
            })
        elif inspect.isclass(obj):
            methods = []
            for method_name, method_obj in inspect.getmembers(obj):
                if method_name.startswith('_') and method_name != '__init__':
                    continue
                if inspect.ismethod(method_obj) or inspect.isfunction(method_obj):
                    methods.append({
                        'name': method_name,
                        'signature': get_function_signature(method_obj),
                        'docstring': get_docstring(method_obj)
                    })
            
            classes.append({
                'name': name,
                'docstring': get_docstring(obj),
                'methods': methods,
                'module': module_name
            })
    
    return {
        'module': module_name,
        'functions': functions,
        'classes': classes
    }


def generate_mdx_file(data: Dict[str, Any], output_path: Path):
    """Generate MDX file from module data."""
    module_name = data['module']
    
    # Create MDX content
    content = f"""---
title: {module_name}
---

# {module_name}

"""
    
    # Add functions
    if data['functions']:
        content += "## Functions\n\n"
        for func in data['functions']:
            content += f"### `{func['signature']}`\n\n"
            content += f"{func['docstring']}\n\n"
    
    # Add classes
    if data['classes']:
        content += "## Classes\n\n"
        for cls in data['classes']:
            content += f"### {cls['name']}\n\n"
            content += f"{cls['docstring']}\n\n"
            
            if cls['methods']:
                content += "#### Methods\n\n"
                for method in cls['methods']:
                    content += f"##### `{method['signature']}`\n\n"
                    content += f"{method['docstring']}\n\n"
    
    # Write the file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)


def main():
    """Main function to generate Python SDK documentation."""
    # Install Weave
    install_weave()
    
    # Import weave to get its location
    import weave
    weave_path = Path(weave.__file__).parent
    
    print(f"Weave installed at: {weave_path}")
    
    # Define the modules we want to document
    modules_to_document = [
        "weave",
        "weave.trace.op",
        "weave.trace.weave_client",
        "weave.trace_server.trace_server_interface",
    ]
    
    # Output directory
    output_dir = Path("weave/reference/python-sdk")
    
    # Generate documentation for each module
    for module_name in modules_to_document:
        try:
            print(f"Documenting {module_name}...")
            data = analyze_module(str(weave_path), module_name)
            
            # Determine output path
            parts = module_name.split('.')
            if len(parts) == 1:
                output_path = output_dir / f"{parts[0]}/index.mdx"
            else:
                output_path = output_dir / '/'.join(parts[:-1]) / f"{parts[-1]}.mdx"
            
            generate_mdx_file(data, output_path)
            print(f"  ✓ Generated {output_path}")
            
        except Exception as e:
            print(f"  ✗ Error documenting {module_name}: {e}")
    
    print("\nPython SDK documentation generation complete!")


if __name__ == "__main__":
    main()
