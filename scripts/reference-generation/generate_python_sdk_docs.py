#!/usr/bin/env python3
"""
Generate Python SDK reference documentation for Mintlify.

This script generates Python API documentation from Weave source code,
converting it to Mintlify's MDX format.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

# Import after installing weave
lazydocs = None
pydantic = None


def check_pypi_version(version):
    """Check if a specific version exists on PyPI."""
    import urllib.request
    import json
    
    try:
        with urllib.request.urlopen("https://pypi.org/pypi/weave/json") as response:
            data = json.loads(response.read())
            available_versions = list(data["releases"].keys())
            return version in available_versions, available_versions
    except Exception as e:
        print(f"Warning: Could not check PyPI versions: {e}")
        return True, []  # Allow to proceed if we can't check


def install_dependencies(weave_version="latest"):
    """Install Weave and other required packages."""
    print(f"Installing dependencies (Weave version: {weave_version})...")
    
    try:
        # Install lazydocs first
        subprocess.run([sys.executable, "-m", "pip", "install", "lazydocs"], check=True)
        
        # Install Weave
        if weave_version == "latest":
            cmd = [sys.executable, "-m", "pip", "install", "weave"]
        elif weave_version.startswith("v") or "." in weave_version:
            version_num = weave_version.lstrip("v")
            
            # Handle -dev0 versions by stripping the suffix for PyPI
            if version_num.endswith("-dev0"):
                print(f"Note: Converting {version_num} to release version for PyPI")
                release_version = version_num.replace("-dev0", "")
                
                # Check if the release version exists on PyPI
                exists, available = check_pypi_version(release_version)
                if not exists:
                    print(f"\nError: Weave version {release_version} is not available on PyPI.")
                    print(f"This appears to be an unreleased development version ({version_num}).")
                    print(f"\nAvailable versions near this one:")
                    # Show versions that start with the same major.minor
                    prefix = ".".join(release_version.split(".")[:2])
                    matching = [v for v in available if v.startswith(prefix)]
                    for v in sorted(matching, reverse=True)[:5]:
                        print(f"  - {v}")
                    sys.exit(1)
                
                version_num = release_version
            else:
                # Regular version - also check if it exists
                exists, available = check_pypi_version(version_num)
                if not exists:
                    print(f"\nError: Weave version {version_num} is not available on PyPI.")
                    print(f"\nAvailable versions:")
                    # Show recent versions
                    for v in sorted(available, reverse=True)[:10]:
                        print(f"  - {v}")
                    sys.exit(1)
            
            cmd = [sys.executable, "-m", "pip", "install", f"weave=={version_num}"]
        else:
            # For commit hashes or branch names, we don't generate docs
            print(f"\nError: Cannot generate documentation for Git references ({weave_version}).")
            print("Please specify a released version available on PyPI or 'latest'.")
            sys.exit(1)
        
        subprocess.run(cmd, check=True)
        print("✓ Dependencies installed successfully")
        
        # Import after installation
        global lazydocs, pydantic
        import lazydocs
        import pydantic
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}", file=sys.stderr)
        sys.exit(1)


def fix_code_fence_indentation(text: str) -> str:
    """Fix code fence indentation issues in markdown."""
    lines = text.split("\n")
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Check if this line contains an opening code fence
        fence_match = re.match(r"^(\s*)(```\w*)$", line)
        if fence_match:
            fence_indent = fence_match.group(1)
            fence_content = fence_match.group(2)
            
            # Find the closing fence
            closing_fence_idx = None
            for j in range(i + 1, len(lines)):
                if re.match(r"^\s*```\s*$", lines[j]):
                    closing_fence_idx = j
                    break
            
            if closing_fence_idx is not None:
                # Get the code block content
                code_lines = lines[i + 1 : closing_fence_idx]
                
                if code_lines:
                    # Find the minimum indentation
                    min_indent = float("inf")
                    for code_line in code_lines:
                        if code_line.strip():
                            indent = len(code_line) - len(code_line.lstrip())
                            min_indent = min(min_indent, indent)
                    
                    if min_indent != float("inf"):
                        # De-indent the code block
                        deindented_code_lines = []
                        for code_line in code_lines:
                            if code_line.strip():
                                deindented_code_lines.append(code_line[min_indent:])
                            else:
                                deindented_code_lines.append(code_line)
                        
                        # Add the fences and code
                        result_lines.append(fence_content)
                        result_lines.extend(deindented_code_lines)
                        result_lines.append("```")
                        
                        i = closing_fence_idx + 1
                        continue
        
        result_lines.append(line)
        i += 1
    
    return "\n".join(result_lines)


def convert_docusaurus_to_mintlify(content: str, module_name: str) -> str:
    """Convert Docusaurus markdown to Mintlify MDX format."""
    # Remove the sidebar_label frontmatter (Mintlify uses title)
    content = re.sub(r'sidebar_label:\s*[^\n]+\n', '', content)
    
    # Fix image tags to be self-closing
    content = re.sub(r'<img(.*?)(?<!/)>', r'<img\1 />', content)
    
    # Remove style attributes (Mintlify doesn't support inline styles)
    content = re.sub(r'\s*style="[^"]*"', '', content)
    
    # Fix factory placeholders
    content = content.replace("<factory>", "&lt;factory&gt;")
    
    # Convert relative links to anchor links
    content = re.sub(r'\[([^\]]+)\]\((\./[^)#]+)?#([^)]+)\)', r'[\1](#\3)', content)
    
    # Add proper Mintlify frontmatter if not present
    if not content.startswith("---"):
        title = module_name.split('.')[-1]
        frontmatter = f"""---
title: "{title}"
description: "Python SDK reference for {module_name}"
---

"""
        content = frontmatter + content
    
    return content


def generate_module_docs(module, module_name: str, src_root_path: str) -> str:
    """Generate documentation for a single module."""
    generator = lazydocs.MarkdownGenerator(
        src_base_url="https://github.com/wandb/weave/blob/master",
        src_root_path=src_root_path,
        remove_package_prefix=True,
    )
    
    sections = []
    
    # Generate overview
    overview = generator.overview2md()
    overview = re.sub(r"## Functions\n\n- No functions", "", overview)
    overview = re.sub(r"## Modules\n\n- No modules", "", overview)
    overview = re.sub(r"## Classes\n\n- No classes", "", overview)
    sections.append(overview)
    
    # Process module contents
    if hasattr(module, "__docspec__"):
        # Use the special __docspec__ attribute if available
        for obj in module.__docspec__:
            sections.append(process_object(obj, generator, module_name))
    else:
        # Fall back to processing all public members
        for name in dir(module):
            if name.startswith("_"):
                continue
            
            obj = getattr(module, name)
            if hasattr(obj, "__module__") and obj.__module__ != module_name:
                continue
            
            sections.append(process_object(obj, generator, module_name))
    
    # Combine all sections
    content = "\n\n---\n\n".join(filter(None, sections))
    
    # Fix code fence indentation
    content = fix_code_fence_indentation(content)
    
    # Convert to Mintlify format
    content = convert_docusaurus_to_mintlify(content, module_name)
    
    return content


def process_object(obj, generator, module_name: str) -> Optional[str]:
    """Process a single object (class, function, etc.) for documentation."""
    try:
        # Special handling for Pydantic models
        if isinstance(obj, type) and issubclass(obj, pydantic.BaseModel):
            return process_pydantic_model(obj, generator, module_name)
        elif callable(obj) and not isinstance(obj, type):
            return generator.func2md(obj)
        elif isinstance(obj, type):
            return generator.class2md(obj)
    except Exception as e:
        print(f"Warning: Error processing {obj}: {e}")
        return None


def process_pydantic_model(obj, generator, module_name: str) -> str:
    """Special processing for Pydantic models."""
    content = generator.class2md(obj)
    
    # Remove unhelpful Pydantic properties
    patterns_to_remove = [
        r"---\n\n#### <kbd>property</kbd> model_extra.*?(?=---|\Z)",
        r"---\n\n#### <kbd>property</kbd> model_fields_set.*?(?=---|\Z)",
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, "---", content, flags=re.DOTALL)
    
    # Clean up multiple separator lines
    content = re.sub(r"(---\n)+", "---\n", content)
    
    # Add Pydantic fields documentation
    if hasattr(obj, "model_fields") and obj.model_fields:
        fields_doc = "\n**Pydantic Fields:**\n\n"
        for name, field in obj.model_fields.items():
            field_name = getattr(field, "alias", name) or name
            annotation = str(getattr(field, "annotation", "Any"))
            annotation = annotation.replace(f"{module_name}.", "")
            fields_doc += f"- `{field_name}`: `{annotation}`\n"
        
        # Insert fields documentation after the class header
        class_header_end = content.find("---")
        if class_header_end != -1:
            content = content[:class_header_end] + fields_doc + "\n" + content[class_header_end:]
    
    return content


def get_modules_to_document():
    """Get the list of modules to document."""
    # Import weave modules
    import weave
    from weave.trace import op, weave_client, util
    from weave.trace_server import trace_server_interface
    
    return [
        (weave, "weave"),
        (weave_client, "weave.trace.weave_client"),
        (op, "weave.trace.op"),
        (util, "weave.trace.util"),
        (trace_server_interface, "weave.trace_server.trace_server_interface"),
    ]


def main():
    """Main function."""
    # Get Weave version from command line, environment variable, or use latest
    if len(sys.argv) > 1:
        weave_version = sys.argv[1]
    else:
        weave_version = os.environ.get("WEAVE_VERSION", "latest")
    
    # Install dependencies
    install_dependencies(weave_version)
    
    # Import weave to get the source path
    import weave
    module_root_path = Path(weave.__file__).parent.parent
    
    # Output directory
    output_dir = Path("weave/reference/python-sdk/weave")
    
    # Clean existing docs
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating Python SDK documentation...")
    
    # Get modules to document
    modules = get_modules_to_document()
    
    for module, module_name in modules:
        print(f"  Generating docs for {module_name}...")
        
        try:
            # Generate documentation
            content = generate_module_docs(module, module_name, str(module_root_path))
            
            # Determine output path
            parts = module_name.split(".")
            if hasattr(module, "__file__") and module.__file__.endswith("__init__.py"):
                # For __init__ modules, create an index.mdx
                file_path = output_dir / Path(*parts[1:]) / "index.mdx"
            else:
                # For regular modules, use the module name
                file_path = output_dir / Path(*parts[1:-1]) / f"{parts[-1]}.mdx"
            
            # Create directory and write file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            
            print(f"    ✓ Saved to {file_path}")
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    print("\n✓ Python SDK documentation generation complete!")


if __name__ == "__main__":
    main()
