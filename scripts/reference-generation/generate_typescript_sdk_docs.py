#!/usr/bin/env python3
"""
Generate TypeScript SDK reference documentation for Mintlify.

This script generates TypeScript documentation using typedoc and converts
it to Mintlify's format.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def check_node_dependencies():
    """Check if Node.js and npm are available."""
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
        print("✓ Node.js is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Node.js is not installed", file=sys.stderr)
        print("  Please install Node.js 18+ from https://nodejs.org/")
        sys.exit(1)
    
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
        print("✓ npm is installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ npm is not installed", file=sys.stderr)
        sys.exit(1)


def download_weave_source(version="main"):
    """Download Weave source code for a specific version using shallow clone."""
    print(f"\nDownloading Weave source code (version: {version})...")
    
    # Handle "latest" by fetching the latest release tag
    if version == "latest":
        # Use git ls-remote to get latest tag without API calls
        try:
            result = subprocess.run(
                ["git", "ls-remote", "--tags", "--sort=-v:refname", "https://github.com/wandb/weave.git"],
                capture_output=True,
                text=True,
                check=True
            )
            # Parse the output to get the latest version tag
            for line in result.stdout.strip().split('\n'):
                if '\trefs/tags/' in line and not line.endswith('^{}'):
                    tag = line.split('\trefs/tags/')[-1]
                    if tag.startswith('v') and not 'dev' in tag and not 'rc' in tag:
                        version = tag
                        print(f"  Using latest release: {version}")
                        break
            else:
                print("  Warning: Could not determine latest release, using main branch")
                version = "main"
        except Exception as e:
            print(f"  Warning: Could not fetch latest release, using main branch: {e}")
            version = "main"
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    weave_dir = Path(temp_dir) / "weave"
    
    try:
        # Use shallow clone with single branch
        print(f"  Cloning Weave repository (shallow clone)...")
        clone_cmd = [
            "git", "clone",
            "--depth", "1",
            "--single-branch"
        ]
        
        # Add branch/tag specification
        if version != "main":
            clone_cmd.extend(["--branch", version])
        
        clone_cmd.extend([
            "https://github.com/wandb/weave.git",
            str(weave_dir)
        ])
        
        # Run the clone command
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True)
        
        print(f"  ✓ Successfully cloned Weave {version} to {weave_dir}")
        
        return weave_dir
        
    except subprocess.CalledProcessError as e:
        print(f"Error cloning Weave repository: {e}", file=sys.stderr)
        if e.stderr:
            print(f"  stderr: {e.stderr}", file=sys.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading Weave source: {e}", file=sys.stderr)
        shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)


def setup_typescript_project(weave_source):
    """Set up the TypeScript project and install dependencies."""
    sdk_path = Path(weave_source) / "sdks" / "node"
    
    if not sdk_path.exists():
        print(f"TypeScript SDK not found at {sdk_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"\nSetting up TypeScript project...")
    os.chdir(sdk_path)
    
    try:
        # Install dependencies
        print("  Installing dependencies...")
        subprocess.run(["npm", "install"], check=True)
        
        # Install typedoc and markdown plugin with compatible versions
        print("  Installing typedoc...")
        subprocess.run([
            "npm", "install", "--save-dev",
            "typedoc@0.25.13",
            "typedoc-plugin-markdown@3.17.1"
        ], check=True)
        
        print("  ✓ Dependencies installed")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}", file=sys.stderr)
        sys.exit(1)
    
    return sdk_path


def generate_typedoc(sdk_path, output_path):
    """Generate TypeScript documentation using typedoc."""
    print(f"\nGenerating TypeScript documentation...")
    
    # Create typedoc config
    config = {
        "entryPoints": ["src/index.ts"],
        "out": str(output_path),
        "plugin": ["typedoc-plugin-markdown"],
        "readme": "none",
        "githubPages": False,
        "excludePrivate": True,
        "excludeProtected": True,
        "excludeInternal": True,
        "disableSources": False,
        "cleanOutputDir": True
    }
    
    config_path = sdk_path / "typedoc.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    try:
        # Run typedoc
        subprocess.run(["npx", "typedoc"], check=True, cwd=sdk_path)
        print("  ✓ Documentation generated")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to generate documentation: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up config file
        if config_path.exists():
            config_path.unlink()


def convert_to_mintlify_format(docs_dir):
    """Convert TypeDoc markdown to Mintlify MDX format."""
    print(f"\nConverting to Mintlify format...")
    
    docs_path = Path(docs_dir)
    md_files = list(docs_path.rglob("*.md"))
    
    for md_file in md_files:
        content = md_file.read_text()
        
        # Skip if already has frontmatter
        if content.startswith("---"):
            continue
        
        # Extract title from content
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else md_file.stem
        
        # Remove the title line if it exists (Mintlify uses frontmatter title)
        if title_match:
            content = content.replace(title_match.group(0), '', 1).strip()
        
        # Fix escaped angle brackets in title (TypeDoc escapes them as \< and \>)
        title_fixed = title.replace('\\<', '<').replace('\\>', '>')
        
        # Add Mintlify frontmatter
        frontmatter = f"""---
title: "{title_fixed}"
description: "TypeScript SDK reference"
---

"""
        content = frontmatter + content
        
        # Fix TypeDoc specific formatting
        # Convert **`code`** to just `code`
        content = re.sub(r'\*\*`([^`]+)`\*\*', r'`\1`', content)
        
        # Fix parameter tables
        content = re.sub(r'\|\s*:--\s*\|', '| --- |', content)
        
        # Fix internal links to use relative paths
        # Keep relative links as-is for files in subdirectories
        # TypeDoc generates links like ../classes/WeaveObject.md which are already relative
        # We just need to remove the .md extension
        
        # Fix internal links
        # 1. Remove .md extensions from all links
        content = re.sub(r'\.md(#[^)]+)?\)', r'\1)', content)
        
        # 2. Fix links to README/index to point to landing page
        content = re.sub(r'\]\(\.\./README', '](../', content)
        content = re.sub(r'\]\(\.\./index', '](../', content)
        content = re.sub(r'\]\(README', '](typescript-sdk', content)
        
        # 3. Fix same-directory links (e.g., WeaveObject -> ./WeaveObject)
        # For links to class names (start with capital letter) that don't have a path
        content = re.sub(r'\]\(([A-Z][a-zA-Z]+)(#[^)]+)?\)', r'](./\1\2)', content)
        
        # 4. Fix cross-directory links based on file location
        if '/functions/' in str(md_file):
            # Functions linking to classes/interfaces need ../
            content = re.sub(r'\]\(classes/([^)]+)\)', r'](../classes/\1)', content)
            content = re.sub(r'\]\(interfaces/([^)]+)\)', r'](../interfaces/\1)', content)
        elif '/type-aliases/' in str(md_file):
            # Type aliases linking to classes need ../
            content = re.sub(r'\]\(classes/([^)]+)\)', r'](../classes/\1)', content)
        elif md_file.name == 'README.md' or md_file.name == 'index.mdx':
            # Index/README will become landing page, so needs ./typescript-sdk/ prefix
            content = re.sub(r'\]\(classes/([^)]+)\)', r'](./typescript-sdk/classes/\1)', content)
            content = re.sub(r'\]\(interfaces/([^)]+)\)', r'](./typescript-sdk/interfaces/\1)', content)
            content = re.sub(r'\]\(functions/([^)]+)\)', r'](./typescript-sdk/functions/\1)', content)
            content = re.sub(r'\]\(type-aliases/([^)]+)\)', r'](./typescript-sdk/type-aliases/\1)', content)
        
        # Special handling for index files (README.md)
        # These files are at the root level and link directly to subdirectories
        # Keep these as relative links too
        if md_file.name == "README.md":
            # Links like [Dataset](classes/Dataset.md) are already relative
            # Just ensure .md extension is removed (already done above)
            pass
        
        # Write as .mdx file
        mdx_file = md_file.with_suffix('.mdx')
        mdx_file.write_text(content)
        
        # Remove original .md file
        md_file.unlink()
        
        print(f"  ✓ Converted {md_file.name} → {mdx_file.name}")


def extract_members_to_separate_files(docs_path):
    """Extract function and type-alias documentation from typescript-sdk.mdx landing page to separate files."""
    # Check for the landing page
    landing_file = docs_path.parent / "typescript-sdk.mdx"
    if landing_file.exists():
        index_file = landing_file
    else:
        # Fallback to index.mdx if it exists (shouldn't happen with new generation)
        index_file = docs_path / "index.mdx"
        if not index_file.exists():
            return
    
    content = index_file.read_text()
    
    # Check if we need to extract anything
    has_functions = "### Functions" in content
    has_type_aliases = "### Type Aliases" in content or "### Type aliases" in content
    
    if not has_functions and not has_type_aliases:
        return
    
    print("  Extracting consolidated members to separate files...")
    
    # Extract functions
    if has_functions:
        functions_dir = docs_path / "functions"
        functions_dir.mkdir(exist_ok=True)
        
        # Find and extract each function section
        # Pattern to match function sections (from ### functionName to the next ### or end)
        function_pattern = re.compile(
            r'(### (init|login|op|requireCurrentCallStackEntry|requireCurrentChildSummary|weaveAudio|weaveImage|wrapOpenAI)\n.*?)(?=\n### |\Z)',
            re.DOTALL
        )
        
        functions_found = []
        for match in function_pattern.finditer(content):
            func_content = match.group(1)
            func_name = match.group(2)
            
            # Create the function file content
            func_title = func_name
            func_file_content = f"""---
title: "{func_title}"
description: "TypeScript SDK reference"
---

{func_content.replace(f'### {func_name}', f'# {func_name}')}"""
            
            # Write the function file
            func_file = functions_dir / f"{func_name}.mdx"
            func_file.write_text(func_file_content)
            functions_found.append(func_name)
            print(f"    ✓ Extracted {func_name}.mdx")
        
        if functions_found:
            # Remove the detailed function documentation from index
            content = function_pattern.sub('', content)
            
            # Update the Functions section with links
            functions_section = "\n### Functions\n\n"
            for func in functions_found:
                functions_section += f"- [{func}](functions/{func})\n"
            
            # Replace existing Functions section with links
            content = re.sub(
                r'### Functions.*?(?=\n### |\n## |\Z)',
                functions_section,
                content,
                flags=re.DOTALL
            )
    
    # Extract type aliases
    if has_type_aliases:
        type_aliases_dir = docs_path / "type-aliases"
        type_aliases_dir.mkdir(exist_ok=True)
        
        # Pattern to match all type alias sections
        # Look for ### TypeAliasName pattern
        type_alias_pattern = re.compile(
            r'(### ([A-Za-z][A-Za-z0-9]*)\n\nƬ .*?)(?=\n### |\Z)',
            re.DOTALL
        )
        
        type_aliases = type_alias_pattern.findall(content)
        for alias_content, alias_name in type_aliases:
            # Skip if it's not a type alias (e.g., if it's a function or class)
            if not alias_content.startswith(f"### {alias_name}\n\nƬ "):
                continue
                
            # Create the type alias file content
            alias_file_content = f"""---
title: "{alias_name}"
description: "TypeScript SDK reference"
---

{alias_content.replace(f'### {alias_name}', f'# {alias_name}')}"""
            
            # Write the type alias file
            alias_file = type_aliases_dir / f"{alias_name}.mdx"
            alias_file.write_text(alias_file_content)
            print(f"    ✓ Extracted {alias_name}.mdx")
        
        # Remove all extracted type aliases from index
        content = type_alias_pattern.sub('', content)
            
        # Update Type Aliases section with links to all extracted type aliases
        if type_aliases:
            type_aliases_links = [f"- [{name}](type-aliases/{name})" for _, name in type_aliases if _.startswith(f"### {name}\n\nƬ ")]
            if type_aliases_links:
                type_aliases_section = "\n### Type Aliases\n\n" + "\n".join(sorted(type_aliases_links)) + "\n"
                
                # Replace existing Type Aliases section
                content = re.sub(
                    r'### Type [Aa]liases.*?(?=\n### |\n## |\Z)',
                    type_aliases_section,
                    content,
                    flags=re.DOTALL
                )
    
    # Write updated content back
    index_file.write_text(content)
    if index_file.name == "typescript-sdk.mdx":
        print(f"  ✓ Updated typescript-sdk.mdx landing page")
    else:
        print(f"  ✓ Updated index.mdx")


def organize_for_mintlify(temp_output, final_output):
    """Organize the generated docs according to Mintlify structure."""
    print(f"\nOrganizing documentation structure...")
    
    temp_path = Path(temp_output)
    final_path = Path(final_output)
    
    # Instead of deleting everything, selectively update
    # This preserves files that might not be regenerated
    if not final_path.exists():
        final_path.mkdir(parents=True, exist_ok=True)
    
    # Copy all files from temp to final, overwriting existing ones
    for item in temp_path.iterdir():
        dest = final_path / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)
    
    # Move README.mdx to typescript-sdk.mdx landing page if it exists
    readme_path = final_path / "README.mdx"
    if readme_path.exists():
        landing_path = final_path.parent / "typescript-sdk.mdx"
        readme_path.rename(landing_path)
        print("  ✓ Created typescript-sdk.mdx landing page from README")
    
    # Extract functions and type aliases to separate files if they're consolidated
    extract_members_to_separate_files(final_path)
    
    print("  ✓ Documentation organized")


def cleanup_temp_dirs(*paths):
    """Clean up temporary directories."""
    for path in paths:
        if path and Path(path).exists():
            shutil.rmtree(path, ignore_errors=True)


def main():
    """Main function."""
    # Check dependencies
    check_node_dependencies()
    
    # Get Weave version from environment or use latest
    weave_version = os.environ.get("WEAVE_VERSION", "latest")
    
    # Store the original working directory
    original_cwd = os.getcwd()
    
    # Download Weave source
    weave_source = download_weave_source(weave_version)
    
    try:
        # Set up TypeScript project
        sdk_path = setup_typescript_project(weave_source)
        
        # Generate documentation to a temporary directory
        temp_output = tempfile.mkdtemp()
        generate_typedoc(sdk_path, temp_output)
        
        # Convert to Mintlify format
        convert_to_mintlify_format(temp_output)
        
        # Organize for Mintlify - use absolute path from original directory
        final_output = os.path.join(original_cwd, "weave/reference/typescript-sdk")
        organize_for_mintlify(temp_output, final_output)
        
        print("\n✓ TypeScript SDK documentation generation complete!")
        
    finally:
        # Clean up temporary directories
        cleanup_temp_dirs(weave_source.parent if 'weave_source' in locals() else None,
                         temp_output if 'temp_output' in locals() else None)


if __name__ == "__main__":
    main()