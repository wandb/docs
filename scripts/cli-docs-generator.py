#!/usr/bin/env python3
"""
Generate Mintlify-compatible markdown documentation for W&B CLI commands.

This script automatically generates reference documentation for the W&B CLI
by introspecting the Click-based command structure. It creates properly
formatted markdown files with Mintlify front matter for use in the documentation site.

Note: Launch-only commands (launch, launch-agent, launch-sweep, scheduler) are generated 
but excluded from the docs.json navigation (hidden from sidebar).

Usage:
    python scripts/cli-docs-generator.py          # Auto-overwrites existing docs (default)
    python scripts/cli-docs-generator.py -i       # Interactive mode (prompts for confirmation)
    python scripts/cli-docs-generator.py -o PATH  # Custom output directory
    
    Options:
        -i, --interactive  Prompt for confirmation before overwriting
        -o, --output       Custom output directory (default: models/ref/cli/)

Output:
    Generated documentation will be placed in: models/ref/cli/ (by default)
    Navigation structure will be updated in: docs.json

Requirements:
    - wandb package must be installed (pip install wandb)
    - Must be run from the project root directory

"""

import os
import re
import sys
import argparse
import shutil
import json
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import click
from click.core import Command, Group

def clean_text(text: str) -> str:
    """Clean and format text for markdown.
    
    Uses inspect.cleandoc() to handle dedenting docstrings, then detects and converts
    indented code blocks to fenced code blocks for Mintlify compatibility.
    
    Mintlify doesn't support implicit indented code blocks, so we convert them to
    triple-backtick fenced blocks. Regular paragraphs are collapsed into single lines.
    """
    if not text:
        return ""
    
    # Use inspect.cleandoc() to clean and dedent the docstring
    # This handles common leading whitespace removal like most doc generators
    text = inspect.cleandoc(text)
    
    # Split into paragraphs (separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        
        lines = paragraph.split('\n')
        
        # Detect if this paragraph is a code block
        # After cleandoc(), code blocks still have relative indentation (4+ spaces)
        is_code_block = _is_code_block(lines)
        
        if is_code_block:
            # Convert indented code block to fenced code block for Mintlify
            code_lines = [line.strip() for line in lines if line.strip()]
            if code_lines:
                fenced_block = '```bash\n' + '\n'.join(code_lines) + '\n```'
                cleaned_paragraphs.append(fenced_block)
        else:
            # Regular paragraph: collapse into single line
            cleaned_text = ' '.join(line.strip() for line in lines if line.strip())
            # Normalize whitespace
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            if cleaned_text:
                cleaned_paragraphs.append(cleaned_text)
    
    # Join paragraphs with double newlines for markdown
    result = '\n\n'.join(cleaned_paragraphs)
    
    # Escape pipe characters for markdown tables
    result = result.replace('|', '\\|')
    return result


def _is_code_block(lines: List[str]) -> bool:
    """Detect if a paragraph is a code block based on indentation.
    
    After inspect.cleandoc(), code blocks still have relative indentation.
    Code blocks typically have 4+ spaces of indentation for all lines.
    """
    if not lines:
        return False
    
    non_empty_lines = [line for line in lines if line.strip()]
    if not non_empty_lines:
        return False
    
    # Check if all non-empty lines start with significant indentation (4+ spaces)
    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
    
    # Code blocks have at least 4 spaces of indentation
    # Regular text paragraphs have 0 spaces after cleandoc()
    return min_indent >= 4

def clean_text_for_table(text: str) -> str:
    """Clean text for use in markdown table cells.
    
    Uses inspect.cleandoc() then collapses all whitespace into single spaces,
    suitable for inline table cell content.
    """
    if not text:
        return ""
    
    # Clean and dedent the text
    text = inspect.cleandoc(text)
    
    # Collapse all whitespace (spaces, tabs, newlines) into single spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Escape pipe characters for markdown tables
    text = text.replace('|', '\\|')
    
    return text

def get_brief_description(text: str) -> str:
    """Get just the first sentence for table display.
    
    Extracts and returns the first sentence from the text, suitable for
    brief descriptions in command tables.
    """
    if not text:
        return "No description available"
    
    # Clean and dedent the text
    text = inspect.cleandoc(text)
    
    # Split into paragraphs and get the first one
    paragraphs = re.split(r'\n\s*\n', text)
    if not paragraphs or not paragraphs[0].strip():
        return "No description available"
    
    # Collapse the first paragraph into a single line
    first_para = re.sub(r'\s+', ' ', paragraphs[0].strip())
    
    # Extract the first sentence
    sentence_match = re.match(r'^(.*?[.!?])\s', first_para + ' ')
    first_sentence = sentence_match.group(1) if sentence_match else first_para
    
    # Limit length for table display
    max_length = 150
    if len(first_sentence) > max_length:
        first_sentence = first_sentence[:max_length-3] + "..."
    
    # Escape pipe characters for markdown tables
    first_sentence = first_sentence.replace('|', '\\|')
    
    return first_sentence

def get_param_info(param) -> Dict[str, Any]:
    """Extract parameter information from a Click parameter."""
    # Check if it's an Option or Argument
    is_option = isinstance(param, click.Option)
    
    info = {
        'name': param.name,
        'opts': getattr(param, 'opts', []) or [],
        'help': clean_text(getattr(param, 'help', '') or ''),
        'type': str(param.type) if hasattr(param, 'type') else 'TEXT',
        'required': getattr(param, 'required', False),
        'default': getattr(param, 'default', None),
        'multiple': getattr(param, 'multiple', False),
    }
    
    # Format options string
    if info['opts']:
        info['opts_str'] = ', '.join(f'`{opt}`' for opt in info['opts'])
    else:
        info['opts_str'] = f'`{info["name"]}`'
    
    # Format default value
    if info['default'] is not None and info['default'] != () and not callable(info['default']):
        info['default_str'] = f" (default: {info['default']})"
    else:
        info['default_str'] = ""
    
    return info

def extract_command_info(cmd: Command, parent_name: str = "") -> Dict[str, Any]:
    """Extract information from a Click command."""
    # Launch-only commands to exclude from docs.json navigation
    LAUNCH_COMMANDS = {'launch', 'launch-agent', 'launch-sweep', 'scheduler'}
    
    # For the root command, use "wandb" as the name
    name = cmd.name if cmd.name else "wandb"
    full_name = f"{parent_name} {name}".strip() if parent_name else name
    
    info = {
        'name': name,
        'full_name': full_name,
        'help': clean_text(cmd.help or cmd.short_help or ''),
        'epilog': clean_text(cmd.epilog) if hasattr(cmd, 'epilog') and cmd.epilog else '',
        'params': [],
        'options': [],
        'arguments': [],
        'subcommands': {},
        'is_group': isinstance(cmd, Group),
        'is_launch_command': name in LAUNCH_COMMANDS,
    }
    
    # Extract parameters
    for param in cmd.params:
        # Skip hidden parameters (they don't show in --help)
        if isinstance(param, click.Option) and getattr(param, 'hidden', False):
            continue
            
        param_info = get_param_info(param)
        if isinstance(param, click.Option):
            info['options'].append(param_info)
        elif isinstance(param, click.Argument):
            info['arguments'].append(param_info)
    
    # Extract subcommands if it's a group
    if isinstance(cmd, Group):
        for name, subcmd in cmd.commands.items():
            info['subcommands'][name] = extract_command_info(subcmd, full_name)
    
    return info

def generate_markdown(cmd_info: Dict[str, Any], file_dir_path: str = "", project_root: Path = None) -> str:
    """Generate markdown content for a command.
    
    Args:
        cmd_info: Command information dictionary
        file_dir_path: Directory path where this file will be located (e.g., "wandb-artifact" for files in that dir)
        project_root: Project root path for finding snippets
    """
    lines = []
    
    # Mintlify front matter (simple, no Hugo properties)
    lines.append("---")
    # Fix title to show 'wandb' instead of 'cli'
    title = cmd_info['full_name'].replace('cli', 'wandb')
    lines.append(f"title: \"{title}\"")
    lines.append("---")
    
    # Check if there's a snippet file for this command
    command_name = cmd_info['full_name'].replace('cli', 'wandb').replace(' ', '-')
    snippet_path = project_root / "snippets/en/_includes/cli" / f"{command_name}.mdx" if project_root else None
    
    if snippet_path and snippet_path.exists():
        # Add import and include the snippet
        # Convert command name to PascalCase for the component name
        component_name = ''.join(word.capitalize() for word in command_name.split('-'))
        lines.append(f'import {component_name} from "/snippets/_includes/cli/{command_name}.mdx";')
        lines.append("")
        lines.append(f"<{component_name}/>")
        lines.append("")
    else:
        # Add commented-out template for adding intro content later
        lines.append("{/*")
        lines.append(f"  To add introductory content for this command:")
        lines.append(f"  1. Create the snippet file: /snippets/_includes/cli/{command_name}.mdx")
        lines.append(f"  2. Add your intro content to that file")
        lines.append(f"  3. Delete this entire comment block and keep only the two lines below:")
        lines.append("")
        component_name = ''.join(word.capitalize() for word in command_name.split('-'))
        lines.append(f'import {component_name} from "/snippets/_includes/cli/{command_name}.mdx";')
        lines.append("")
        lines.append(f"<{component_name}/>")
        lines.append("")
        lines.append(f"  The snippet will be auto-detected on the next regeneration.")
        lines.append("*/}")
        lines.append("")
        
        # Only add CLI help text when there's no snippet file
        # (When a snippet exists, it should contain any necessary intro content)
        if cmd_info['help']:
            lines.append(cmd_info['help'])
            lines.append("")
    
    # Usage section
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    
    # Fix usage to always show 'wandb' as the base command
    if cmd_info['name'] in ['wandb', 'cli']:
        usage = "wandb"
    else:
        usage = cmd_info['full_name'].replace('cli', 'wandb')
    
    # Add arguments to usage
    for arg in cmd_info['arguments']:
        if arg['required']:
            usage += f" {arg['name'].upper()}"
        else:
            usage += f" [{arg['name'].upper()}]"
    
    # Add [OPTIONS] if there are options
    if cmd_info['options']:
        usage += " [OPTIONS]"
    
    # Add subcommand placeholder if it's a group
    if cmd_info['is_group'] and cmd_info['subcommands']:
        usage += " COMMAND [ARGS]..."
    
    lines.append(usage)
    lines.append("```")
    lines.append("")
    
    # Arguments section
    if cmd_info['arguments']:
        lines.append("## Arguments")
        lines.append("")
        lines.append("| Argument | Description | Required |")
        lines.append("| :--- | :--- | :--- |")
        for arg in cmd_info['arguments']:
            required = "Yes" if arg['required'] else "No"
            desc = clean_text_for_table(arg['help']) if arg['help'] else "No description available"
            lines.append(f"| `{arg['name'].upper()}` | {desc} | {required} |")
        lines.append("")
    
    # Options section
    if cmd_info['options']:
        lines.append("## Options")
        lines.append("")
        lines.append("| Option | Description |")
        lines.append("| :--- | :--- |")
        for opt in cmd_info['options']:
            desc = clean_text_for_table(opt['help']) if opt['help'] else "No description available"
            if opt['default_str']:
                desc += opt['default_str']
            lines.append(f"| {opt['opts_str']} | {desc} |")
        lines.append("")
    
    # Subcommands section
    if cmd_info['is_group'] and cmd_info['subcommands']:
        lines.append("## Commands")
        lines.append("")
        lines.append("| Command | Description |")
        lines.append("| :--- | :--- |")
        for name, subcmd_info in sorted(cmd_info['subcommands'].items()):
            # Use brief description for table display
            desc = get_brief_description(subcmd_info['help'] or "")
            # For nested commands, we need to determine the correct reference path
            # Get the full parent prefix from the full_name
            parent_parts = cmd_info['full_name'].replace('cli', 'wandb').split()
            parent_prefix = '-'.join(parent_parts)
            # Subcommands are in a subdirectory named after the parent
            # Build the full path including the directory this file is in
            if file_dir_path:
                full_path = f"/models/ref/cli/{file_dir_path}/{parent_prefix}/{parent_prefix}-{name}"
            else:
                full_path = f"/models/ref/cli/{parent_prefix}/{parent_prefix}-{name}"
            lines.append(f"| [{name}]({full_path}) | {desc} |")
        lines.append("")
    
    # Epilog section
    if cmd_info['epilog']:
        lines.append("## Additional Information")
        lines.append("")
        lines.append(cmd_info['epilog'])
        lines.append("")
    
    return "\n".join(lines)

def generate_index_markdown(cmd_info: Dict[str, Any], subcommands_only: bool = False, file_dir_path: str = "", project_root: Path = None, wandb_version: str = None) -> str:
    """Generate index markdown for a command group.
    
    Args:
        cmd_info: Command information dictionary
        subcommands_only: Whether this is a subcommand index page
        file_dir_path: Directory path where this file will be located (e.g., "wandb-artifact" for files in that dir)
        project_root: Project root path for finding snippets
        wandb_version: W&B SDK version string (e.g., "0.23.0")
    """
    lines = []
    
    # Mintlify front matter (simple, no Hugo properties)
    lines.append("---")
    if subcommands_only:
        # For subcommand index pages
        title = f"{cmd_info['full_name']}".replace('cli', 'wandb')
        lines.append(f"title: \"{title}\"")
    else:
        # Main CLI landing page - include version in title
        if wandb_version:
            lines.append(f"title: \"CLI Reference SDK {wandb_version}\"")
        else:
            lines.append("title: \"CLI Reference\"")
        lines.append("description: \"Use the W&B Command Line Interface (CLI) to log in, run jobs, execute sweeps, and more using shell commands\"")
    lines.append("---")
    lines.append("")
    
    # Check if there's a snippet file for this command/index
    command_name = cmd_info['full_name'].replace('cli', 'wandb').replace(' ', '-')
    snippet_path = project_root / "snippets/en/_includes/cli" / f"{command_name}.mdx" if project_root else None
    
    # Add snippet or template for manual content
    has_snippet = snippet_path and snippet_path.exists()
    if has_snippet:
        # Add import and include the snippet
        # Convert command name to PascalCase for the component name
        component_name = ''.join(word.capitalize() for word in command_name.split('-'))
        lines.append(f'import {component_name} from "/snippets/_includes/cli/{command_name}.mdx";')
        lines.append("")
        lines.append(f"<{component_name}/>")
        lines.append("")
    elif not subcommands_only:
        # For main CLI page only, show commented template when no snippet exists
        lines.append("{/*")
        lines.append(f"  To add introductory content for this command:")
        lines.append(f"  1. Create the snippet file: /snippets/_includes/cli/{command_name}.mdx")
        lines.append(f"  2. Add your intro content to that file")
        lines.append(f"  3. Delete this entire comment block and keep only the two lines below:")
        lines.append("")
        component_name = ''.join(word.capitalize() for word in command_name.split('-'))
        lines.append(f'import {component_name} from "/snippets/_includes/cli/{command_name}.mdx";')
        lines.append("")
        lines.append(f"<{component_name}/>")
        lines.append("")
        lines.append(f"  The snippet will be auto-detected on the next regeneration.")
        lines.append("*/}")
        lines.append("")
    
    # Only add CLI help text when there's no snippet file
    # (When a snippet exists, it should contain any necessary intro content)
    if not has_snippet and cmd_info['help']:
        lines.append(cmd_info['help'])
        lines.append("")
    
    # Usage section
    lines.append("## Usage")
    lines.append("")
    lines.append("```bash")
    # Always use 'wandb' as the base command
    if cmd_info['name'] in ['wandb', 'cli']:
        usage = "wandb"
    else:
        usage = cmd_info['full_name'].replace('cli', 'wandb')
    if cmd_info['options']:
        usage += " [OPTIONS]"
    if cmd_info['is_group'] and cmd_info['subcommands']:
        usage += " COMMAND [ARGS]..."
    lines.append(usage)
    lines.append("```")
    lines.append("")
    
    # Options section (for main command)
    if cmd_info['options']:
        lines.append("## Options")
        lines.append("")
        lines.append("| Option | Description |")
        lines.append("| :--- | :--- |")
        for opt in cmd_info['options']:
            desc = clean_text_for_table(opt['help']) if opt['help'] else "No description available"
            if opt['default_str']:
                desc += opt['default_str']
            lines.append(f"| {opt['opts_str']} | {desc} |")
        lines.append("")
    
    # Commands section
    if cmd_info['is_group'] and cmd_info['subcommands']:
        lines.append("## Commands")
        lines.append("")
        lines.append("| Command | Description |")
        lines.append("| :--- | :--- |")
        for name, subcmd_info in sorted(cmd_info['subcommands'].items()):
            # Use brief description for table display
            desc = get_brief_description(subcmd_info['help'] or "")
            # Link to the command page using absolute Mintlify paths
            if subcommands_only:
                # Get the full parent prefix from the full_name
                parent_parts = cmd_info['full_name'].replace('cli', 'wandb').split()
                parent_prefix = '-'.join(parent_parts)
                # Subcommands are in a subdirectory named after the parent command
                # The subdirectory is named {parent_prefix} and contains files named {parent_prefix}-{name}
                if file_dir_path:
                    full_path = f"/models/ref/cli/{file_dir_path}/{parent_prefix}/{parent_prefix}-{name}"
                else:
                    full_path = f"/models/ref/cli/{parent_prefix}/{parent_prefix}-{name}"
                lines.append(f"| [{name}]({full_path}) | {desc} |")
            else:
                # Top-level commands from main CLI page
                lines.append(f"| [{name}](/models/ref/cli/wandb-{name}) | {desc} |")
        lines.append("")
    
    return "\n".join(lines)

def write_command_docs(cmd_info: Dict[str, Any], output_dir: Path, parent_path: str = "", file_dir_path: str = "", project_root: Path = None, wandb_version: str = None):
    """Write documentation files for a command and its subcommands.
    
    For Mintlify, the structure is:
    - Root: cli.mdx (main index) in models/ref/
    - Command groups: wandb-artifact.mdx (group file) + wandb-artifact/ (directory for subcommands)
    - Simple commands: wandb-login.mdx (single file)
    - Nested in directories: wandb-artifact/wandb-artifact-get.mdx
    
    Args:
        cmd_info: Command information dictionary
        output_dir: Base output directory for files
        parent_path: Accumulated path prefix for file naming (e.g., "artifact-" for cache in artifact)
        file_dir_path: Directory path where files are located relative to models/ref/cli/ (e.g., "wandb-artifact")
        project_root: Project root path for finding snippets
        wandb_version: W&B SDK version string (only used for main CLI page)
    """
    cmd_name = cmd_info['name']
    
    # Check if this is the root wandb command (either 'cli' or 'wandb' or None)
    is_root = cmd_name in ['wandb', 'cli', None] and parent_path == ""
    
    if is_root:
        # Main CLI index - Mintlify wants cli.mdx in models/ref/
        index_path = output_dir.parent / "cli.mdx"
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, 'w') as f:
            f.write(generate_index_markdown(cmd_info, subcommands_only=False, file_dir_path="", project_root=project_root, wandb_version=wandb_version))
        print(f"Generated: {index_path}")
        
        # Process top-level commands (alphabetically sorted)
        for name, subcmd_info in sorted(cmd_info['subcommands'].items()):
            write_command_docs(subcmd_info, output_dir, "", "", project_root, wandb_version)
    else:
        # Determine if this command has subcommands
        has_subcommands = cmd_info['is_group'] and cmd_info['subcommands']
        
        if has_subcommands:
            # For command groups, create BOTH a .mdx file AND a directory
            # The .mdx file serves as the group page
            cmd_file_name = f"wandb-{parent_path}{cmd_name}"
            cmd_file_path = output_dir / f"{cmd_file_name}.mdx"
            with open(cmd_file_path, 'w') as f:
                f.write(generate_index_markdown(cmd_info, subcommands_only=True, file_dir_path=file_dir_path, project_root=project_root))
            print(f"Generated: {cmd_file_path}")
            
            # Create the directory for subcommands
            cmd_dir = output_dir / cmd_file_name
            cmd_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate the new file_dir_path for subcommands (append this directory to the path)
            if file_dir_path:
                new_file_dir_path = f"{file_dir_path}/{cmd_file_name}"
            else:
                new_file_dir_path = cmd_file_name
            
            # Write individual subcommand files (alphabetically sorted)
            for subcmd_name, subcmd_info in sorted(cmd_info['subcommands'].items()):
                # Check if this subcommand also has subcommands
                if subcmd_info['is_group'] and subcmd_info['subcommands']:
                    # Handle nested command groups recursively
                    write_command_docs(subcmd_info, cmd_dir, f"{parent_path}{cmd_name}-", new_file_dir_path, project_root, wandb_version)
                else:
                    # Write single subcommand file in the directory
                    subcmd_path = cmd_dir / f"wandb-{parent_path}{cmd_name}-{subcmd_name}.mdx"
                    with open(subcmd_path, 'w') as f:
                        f.write(generate_markdown(subcmd_info, file_dir_path=new_file_dir_path, project_root=project_root))
                    print(f"Generated: {subcmd_path}")
        else:
            # Write single command file
            cmd_path = output_dir / f"wandb-{parent_path}{cmd_name}.mdx"
            with open(cmd_path, 'w') as f:
                f.write(generate_markdown(cmd_info, file_dir_path=file_dir_path, project_root=project_root))
            print(f"Generated: {cmd_path}")

def build_nav_structure(cmd_info: Dict[str, Any], parent_path: str = "") -> List[Union[str, Dict[str, Any]]]:
    """Build navigation structure for docs.json, excluding launch commands.
    
    Returns a list of navigation items (either strings for simple pages or dicts for groups).
    """
    nav_items = []
    
    if not cmd_info['is_group'] or not cmd_info['subcommands']:
        return nav_items
    
    for name, subcmd_info in sorted(cmd_info['subcommands'].items()):
        # Skip launch commands in navigation
        if subcmd_info.get('is_launch_command', False):
            continue
        
        # Build the path for this command
        if parent_path:
            path = f"{parent_path}-{name}"
        else:
            path = f"wandb-{name}"
        
        if subcmd_info['is_group'] and subcmd_info['subcommands']:
            # This is a group with subcommands
            subpages = []
            
            # IMPORTANT: Include the group's own page first!
            # This is the wandb-{name}.mdx file that serves as the group overview
            subpages.append(f"models/ref/cli/{path}")
            
            # Recursively build subcommand navigation
            for subname, subinfo in sorted(subcmd_info['subcommands'].items()):
                # Skip launch commands
                if subinfo.get('is_launch_command', False):
                    continue
                
                subpath = f"{path}-{subname}"
                
                if subinfo['is_group'] and subinfo['subcommands']:
                    # Nested group - needs its own page plus subpages
                    nested_pages = []
                    # Include the nested group's own page first
                    nested_pages.append(f"models/ref/cli/{path}/{subpath}")
                    # Then add its subcommands
                    for nestedname, nestedinfo in sorted(subinfo['subcommands'].items()):
                        if nestedinfo.get('is_launch_command', False):
                            continue
                        nested_pages.append(f"models/ref/cli/{path}/{subpath}/{subpath}-{nestedname}")
                    
                    if nested_pages:
                        subpages.append({
                            "group": f"{name} {subname}",
                            "pages": nested_pages
                        })
                else:
                    # Simple subcommand
                    subpages.append(f"models/ref/cli/{path}/{subpath}")
            
            if subpages:
                nav_items.append({
                    "group": f"wandb {name}",
                    "pages": subpages
                })
        else:
            # Simple command
            nav_items.append(f"models/ref/cli/{path}")
    
    return nav_items


def update_docs_json(project_root: Path, nav_items: List[Union[str, Dict[str, Any]]]):
    """Update docs.json with the new CLI navigation structure."""
    docs_json_path = project_root / "docs.json"
    
    if not docs_json_path.exists():
        print(f"❌ Error: docs.json not found at {docs_json_path}")
        return False
    
    try:
        with open(docs_json_path, 'r') as f:
            docs_data = json.load(f)
        
        # Find the CLI section in the navigation
        # It's nested under navigation -> pages -> (some models section) -> pages -> CLI group
        found = False
        
        def find_and_update_cli(obj):
            """Recursively find and update the CLI section."""
            nonlocal found
            
            if isinstance(obj, dict):
                if obj.get('group') == 'Command Line Interface (CLI)':
                    # Found it! Update the pages
                    obj['pages'] = ["models/ref/cli"] + nav_items
                    found = True
                    return True
                
                # Recurse into nested structures
                for key, value in obj.items():
                    if find_and_update_cli(value):
                        return True
            
            elif isinstance(obj, list):
                for item in obj:
                    if find_and_update_cli(item):
                        return True
            
            return False
        
        find_and_update_cli(docs_data)
        
        if not found:
            print("⚠️  Warning: Could not find 'Command Line Interface (CLI)' section in docs.json")
            return False
        
        # Write back to docs.json
        with open(docs_json_path, 'w') as f:
            json.dump(docs_data, f, indent=2)
        
        print(f"✅ Updated docs.json with CLI navigation (excluded launch commands)")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Error: Could not parse docs.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error updating docs.json: {e}")
        return False


def main():
    """Main function to generate CLI documentation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate W&B CLI documentation')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Prompt for confirmation before overwriting existing documentation')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Custom output directory (default: models/ref/cli)')
    args = parser.parse_args()
    
    # Get the script's parent directory (scripts/) and then go up to the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "models" / "ref" / "cli"
    
    print("Generating W&B CLI documentation...")
    print(f"Output directory: {output_dir}")
    
    # Warning about overwriting existing documentation (only in interactive mode)
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.interactive:
            print(f"⚠️  Warning: This will regenerate documentation in {output_dir}")
            print("   Note: Manual content should be in snippet files (/snippets/_includes/cli/*.mdx)")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        
        # Clear directory selectively to remove any vestigial files from removed commands
        print("Clearing existing documentation...")
        files_removed = 0
        dirs_removed = 0
        for item in output_dir.iterdir():
            if item.is_file():
                item.unlink()
                files_removed += 1
            elif item.is_dir():
                shutil.rmtree(item)
                dirs_removed += 1
        print(f"  Removed {files_removed} files and {dirs_removed} directories")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Import W&B CLI and get version
        import wandb
        from wandb.cli.cli import cli as wandb_cli
        
        wandb_version = wandb.__version__
        print(f"Generating documentation for W&B CLI version {wandb_version}")
        
        # Extract command information
        print("Extracting CLI structure...")
        cli_info = extract_command_info(wandb_cli)
        
        # Generate documentation files
        print("Generating markdown files...")
        write_command_docs(cli_info, output_dir, "", "", project_root, wandb_version)
        
        print(f"\n✅ Documentation generated successfully in {output_dir}")
        # Count both .mdx files in output_dir and the cli.mdx in parent
        mdx_files = list(output_dir.rglob('*.mdx'))
        cli_index = output_dir.parent / "cli.mdx"
        if cli_index.exists():
            mdx_files.append(cli_index)
        print(f"Total files generated: {len(mdx_files)}")
        
        # Update docs.json with navigation structure
        print("\nUpdating docs.json...")
        nav_items = build_nav_structure(cli_info)
        if update_docs_json(project_root, nav_items):
            print("✅ Navigation structure updated (Launch commands excluded from TOC)")
        else:
            print("⚠️  Warning: Could not update docs.json navigation")
        
    except ImportError as e:
        print(f"❌ Error: Could not import wandb CLI. Make sure wandb is installed.")
        print(f"   Run: pip install wandb")
        print(f"   Error details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
