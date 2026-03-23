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
import textwrap
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


def get_installed_version():
    """Get the installed Weave version."""
    try:
        import weave
        return weave.__version__
    except:
        return None


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
    """Fix code fence indentation issues in markdown.
    
    Uses textwrap.dedent() to remove common leading whitespace from code blocks,
    following standard Python text processing patterns.
    """
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
                    # Use textwrap.dedent() to remove common leading whitespace
                    code_text = "\n".join(code_lines)
                    dedented_code = textwrap.dedent(code_text)
                    deindented_code_lines = dedented_code.split("\n")
                    
                    # Add the fences and code
                    result_lines.append(fence_content)
                    result_lines.extend(deindented_code_lines)
                    result_lines.append("```")
                    
                    i = closing_fence_idx + 1
                    continue
        
        result_lines.append(line)
        i += 1
    
    return "\n".join(result_lines)


def convert_source_badges_to_buttons(content: str) -> str:
    """Convert shields.io source badge images to SourceLink components.
    
    This avoids Mintlify's image lightbox from triggering when clicking source links.
    
    Converts:
        <a href="..."><img ... src="...badge/-source..." ></a>
    To:
        <SourceLink url="..." />
    """
    # Pattern matches both self-closing (/>) and non-self-closing (>) img tags
    # lazydocs generates non-self-closing tags: <img ... >
    pattern = r'<a href="(https://github\.com/wandb/weave/blob/[^"]+)">\s*<img[^>]*src="https://img\.shields\.io/badge/-source[^"]*"[^>]*/?>\s*</a>'
    replacement = r'<SourceLink url="\1" />'
    return re.sub(pattern, replacement, content)


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

import {{ SourceLink }} from '/snippets/en/_includes/source-link.mdx';

"""
        content = frontmatter + content
    
    return content


def generate_module_docs(module, module_name: str, src_root_path: str, version: str = "master") -> str:
    """Generate documentation for a single module."""
    # Use the specific version tag for source links
    src_url = f"https://github.com/wandb/weave/blob/v{version}"
    if version == "latest":
        src_url = "https://github.com/wandb/weave/blob/master"
    
    generator = lazydocs.MarkdownGenerator(
        src_base_url=src_url,
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
        # Track objects we've already documented to avoid duplicates from aliases
        documented_objects = set()
        
        for name in dir(module):
            if name.startswith("_"):
                continue
            
            obj = getattr(module, name)
            
            # For the root "weave" module, include re-exported items from submodules
            # Otherwise, only include items that belong to this specific module
            if module_name != "weave" and hasattr(obj, "__module__") and obj.__module__ != module_name:
                continue
            
            # Skip if we've already documented this object (handles aliases)
            # Use id() to check object identity, not equality
            obj_id = id(obj)
            if obj_id in documented_objects:
                continue
            documented_objects.add(obj_id)
            
            sections.append(process_object(obj, generator, module_name))
    
    # Combine all sections
    content = "\n\n---\n\n".join(filter(None, sections))
    
    # Fix code fence indentation
    content = fix_code_fence_indentation(content)
    
    # Clean up excessive whitespace (multiple blank lines)
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Fix broken URLs where lazydocs splits them incorrectly
    # Pattern: [link text](https`</b>: //domain...)
    content = re.sub(r'\]\(https`</b>:\s*//', '](https://', content)
    # Also fix any other protocol splits
    content = re.sub(r'\]\(http`</b>:\s*//', '](http://', content)
    # Fix broken bold tags around URLs
    content = re.sub(r'<b>`([^`]+)\]\(([^)]+)`</b>', r'**\1](\2**', content)
    # Fix unclosed <b> tags that break MDX parsing
    # Remove <b>` at the start of lines that don't have a closing </b>
    content = re.sub(r'^- <b>`([^`\n]*?)$', r'- \1', content, flags=re.MULTILINE)
    
    # Remove malformed table separators that lazydocs sometimes generates
    # These appear as standalone lines with just dashes (------) which break markdown parsing
    content = re.sub(r'\n\s*------+\s*\n', '\n\n', content)
    
    # Fix lazydocs bug: inline code fences in Examples sections
    # lazydocs incorrectly wraps text after colons (like "Basic usage:") with inline backticks
    # Pattern: " text: ``` code```" should be "text:\n```python\ncode"
    # This happens when lazydocs misapplies _RE_ARGSTART in Examples sections
    content = re.sub(r'^ ([\w\s]+): ``` (.*?)```\n(    >>> )', r' \1:\n\n```python\n\2\n\3', content, flags=re.MULTILINE)
    
    # Fix parameter lists that have been broken by lazydocs
    # Strategy: Parse all parameters into a structured format, then reconstruct them properly
    def fix_parameter_lists(text):
        """Fix parameter lists where continuations are incorrectly marked as new parameters."""
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check if we're at the start of a parameter list (Args: or similar)
            if line.strip() in ['**Args:**', '**Arguments:**', '**Parameters:**', '**Kwargs:**']:
                fixed_lines.append(line)
                i += 1
                
                # Collect all parameter lines until we hit a non-parameter section
                params = []
                current_param = None
                
                while i < len(lines):
                    curr_line = lines[i]
                    stripped = curr_line.strip()
                    
                    # Check if we've reached the end of the parameter list
                    if stripped.startswith('**') and stripped.endswith(':**'):
                        break  # Hit the next section (Returns:, Raises:, etc.)
                    if not stripped:
                        # Empty line might signal end of params
                        if current_param:
                            params.append(current_param)
                            current_param = None
                        fixed_lines.append(curr_line)
                        i += 1
                        continue
                    
                    # Check if this is a parameter line
                    if stripped.startswith('- <b>`'):
                        # Try to parse as a properly formatted parameter
                        # A valid parameter line should match: - <b>`param_name`</b>: description
                        match = re.match(r'^- <b>`([^`]+)`</b>:\s*(.*)', stripped)
                        if match:
                            # Extract parameter name and description
                            param_name = match.group(1)
                            param_desc = match.group(2)
                            
                            # Check if the parameter name is valid (alphanumeric/underscore only)
                            # Valid parameter names don't contain spaces or special characters
                            if re.match(r'^[a-zA-Z_][\w_]*(\.[a-zA-Z_][\w_]*)*$', param_name):
                                # This is a real parameter
                                if current_param:
                                    params.append(current_param)
                                current_param = {'name': param_name, 'desc': param_desc}
                            else:
                                # The "parameter name" contains invalid characters
                                # This is probably a continuation line that was incorrectly formatted
                                # Extract the full content and treat as continuation
                                continuation = stripped[6:]  # Skip '- <b>`'
                                continuation = continuation.replace('`</b>:', '').replace('`</b>', '').replace('`', '').strip()
                                if current_param:
                                    current_param['desc'] += ' ' + continuation
                                else:
                                    fixed_lines.append(curr_line)
                        else:
                            # Doesn't match the parameter pattern at all
                            # This is a malformed line - likely a continuation
                            # Extract whatever content we can find
                            continuation = stripped[6:] if len(stripped) > 6 else ''
                            # Clean up any stray markup
                            continuation = re.sub(r'`</b>:?', '', continuation)
                            continuation = continuation.replace('`', '').strip()
                            
                            if current_param:
                                current_param['desc'] += ' ' + continuation
                            else:
                                # No current parameter to attach to
                                fixed_lines.append(curr_line)
                    else:
                        # Regular line, might be continuation of description
                        if current_param and stripped:
                            # This is a continuation of the current parameter's description
                            current_param['desc'] += ' ' + stripped
                        else:
                            fixed_lines.append(curr_line)
                    
                    i += 1
                
                # Add the last parameter if exists
                if current_param:
                    params.append(current_param)
                
                # Reconstruct the parameter list with proper formatting
                for param in params:
                    fixed_lines.append(f" - <b>`{param['name']}`</b>: {param['desc']}")
                
            else:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)
    
    content = fix_parameter_lists(content)
    
    # Convert source badge images to text buttons (avoids Mintlify lightbox issue)
    content = convert_source_badges_to_buttons(content)
    
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
    import weave
    
    # Define the modules we want to document based on upstream Weave docs
    # These are the public API modules that users should interact with
    public_api_modules = [
        "weave",  # Main module
        "weave.trace.feedback",  # Feedback API
        "weave.trace.op",  # Operations API
        "weave.trace.util",  # Utility functions
        "weave.trace.weave_client",  # Client API
        "weave.trace_server.trace_server_interface",  # Server interface
        "weave.trace_server_bindings.remote_http_trace_server",  # Remote server bindings
    ]
    
    modules_to_document = []
    
    for module_name in public_api_modules:
        try:
            # Try to import the module
            import importlib
            module = importlib.import_module(module_name)
            modules_to_document.append((module, module_name))
            print(f"  ✓ Found module: {module_name}")
        except ImportError as e:
            print(f"  ⚠ Could not import {module_name}: {e}")
        except Exception as e:
            print(f"  ⚠ Error with {module_name}: {e}")
    
    return modules_to_document


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
    output_dir = Path("weave/reference/python-sdk")
    
    # Clean existing docs
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating Python SDK documentation...")
    print("Discovering public modules...")
    
    # Get modules to document
    modules = get_modules_to_document()
    
    # Determine the version tag to use for source links
    link_version = weave_version
    if weave_version == "latest":
        # Get the actual installed version for links
        actual_version = get_installed_version()
        if actual_version:
            link_version = actual_version
    
    for module, module_name in modules:
        print(f"  Generating docs for {module_name}...")
        
        try:
            # Generate documentation
            content = generate_module_docs(module, module_name, str(module_root_path), link_version)
            
            # Determine output path
            parts = module_name.split(".")
            
            # Special case: root weave module becomes the landing page
            if module_name == "weave":
                file_path = output_dir.parent / "python-sdk.mdx"
            elif hasattr(module, "__file__") and module.__file__.endswith("__init__.py"):
                # For other __init__ modules, create an index.mdx
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
    
    # Output the actual installed version for the workflow
    actual_version = get_installed_version()
    if actual_version:
        print(f"\nACTUAL_VERSION={actual_version}")


if __name__ == "__main__":
    main()
