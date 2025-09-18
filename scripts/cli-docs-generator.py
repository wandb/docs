#!/usr/bin/env python3
"""
Generate Hugo-compatible markdown documentation for W&B CLI commands.

This script automatically generates reference documentation for the W&B CLI
by introspecting the Click-based command structure. It creates properly
formatted markdown files with Hugo front matter for use in the documentation site.

Note: This script excludes launch-only commands (launch, launch-agent, launch-sweep, 
scheduler) from documentation as these are internal/experimental features.

Usage:
    python scripts/cli-docs-generator.py          # Auto-overwrites existing docs (default)
    python scripts/cli-docs-generator.py -i       # Interactive mode (prompts for confirmation)
    python scripts/cli-docs-generator.py -o PATH  # Custom output directory
    
    Options:
        -i, --interactive  Prompt for confirmation before overwriting
        -o, --output       Custom output directory (default: content/en/ref/cli/)

Output:
    Generated documentation will be placed in: content/en/ref/cli/ (by default)

Requirements:
    - wandb package must be installed (pip install wandb)
    - Must be run from the project root directory

"""

import os
import re
import sys
import argparse
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
import click
from click.core import Command, Group

def clean_text(text: str) -> str:
    """Clean and format text for markdown."""
    if not text:
        return ""
    
    # First, normalize line endings and strip the whole text
    text = text.strip()
    
    # Split into paragraphs (separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Clean each paragraph: remove extra spaces within lines, but keep paragraph structure
    cleaned_paragraphs = []
    for paragraph in paragraphs:
        # Replace multiple spaces/tabs with single space, but keep newlines
        lines = paragraph.split('\n')
        cleaned_lines = []
        for line in lines:
            # Clean up spaces within the line
            cleaned_line = re.sub(r'[ \t]+', ' ', line.strip())
            if cleaned_line:  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)
        if cleaned_lines:
            cleaned_paragraphs.append(' '.join(cleaned_lines))
    
    # Join paragraphs with double newlines for markdown
    text = '\n\n'.join(cleaned_paragraphs)
    
    # Escape pipe characters for markdown tables
    text = text.replace('|', '\\|')
    return text

def get_brief_description(text: str) -> str:
    """Get just the first sentence or line for table display."""
    if not text:
        return "No description available"
    
    # First clean the text but without escaping pipes yet
    text = text.strip()
    
    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    if not paragraphs:
        return "No description available"
    
    # Get the first paragraph
    first_para = paragraphs[0]
    
    # Clean up spaces in the first paragraph
    first_para = re.sub(r'[ \t\n]+', ' ', first_para.strip())
    
    # Try to get just the first sentence
    # Look for sentence endings
    sentence_match = re.match(r'^(.*?[.!?])\s', first_para + ' ')
    if sentence_match:
        first_sentence = sentence_match.group(1)
    else:
        # If no sentence ending found, take the whole first paragraph
        # but limit to reasonable length
        first_sentence = first_para
    
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
    # Commands to exclude from documentation (launch-only features)
    EXCLUDED_COMMANDS = {'launch', 'launch-agent', 'launch-sweep', 'scheduler'}
    
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
    }
    
    # Extract parameters
    for param in cmd.params:
        param_info = get_param_info(param)
        if isinstance(param, click.Option):
            info['options'].append(param_info)
        elif isinstance(param, click.Argument):
            info['arguments'].append(param_info)
    
    # Extract subcommands if it's a group
    if isinstance(cmd, Group):
        for name, subcmd in cmd.commands.items():
            # Skip excluded commands
            if name not in EXCLUDED_COMMANDS:
                info['subcommands'][name] = extract_command_info(subcmd, full_name)
    
    return info

def generate_markdown(cmd_info: Dict[str, Any]) -> str:
    """Generate markdown content for a command."""
    lines = []
    
    # Hugo front matter
    lines.append("---")
    # Fix title to show 'wandb' instead of 'cli'
    title = cmd_info['full_name'].replace('cli', 'wandb')
    lines.append(f"title: {title}")
    lines.append("---")
    lines.append("")
    
    # Command description
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
        usage = f"wandb {cmd_info['name']}"
    
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
            desc = arg['help'] or "No description available"
            lines.append(f"| `{arg['name'].upper()}` | {desc} | {required} |")
        lines.append("")
    
    # Options section
    if cmd_info['options']:
        lines.append("## Options")
        lines.append("")
        lines.append("| Option | Description |")
        lines.append("| :--- | :--- |")
        for opt in cmd_info['options']:
            desc = opt['help'] or "No description available"
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
            if subcmd_info['is_group'] and subcmd_info['subcommands']:
                # Link to subdirectory index
                lines.append(f"| [{name}]({{{{< relref \"{parent_prefix}-{name}/_index\" >}}}}) | {desc} |")
            else:
                # Link to single file
                lines.append(f"| [{name}]({{{{< relref \"{parent_prefix}-{name}\" >}}}}) | {desc} |")
        lines.append("")
    
    # Epilog section
    if cmd_info['epilog']:
        lines.append("## Additional Information")
        lines.append("")
        lines.append(cmd_info['epilog'])
        lines.append("")
    
    return "\n".join(lines)

def generate_index_markdown(cmd_info: Dict[str, Any], subcommands_only: bool = False, content_before: str = None, content_after: str = None) -> str:
    """Generate index markdown for a command group.
    
    Args:
        cmd_info: Command information dictionary
        subcommands_only: Whether this is a subcommand index page
        content_before: Manual content to preserve before auto-generated content
        content_after: Manual content to preserve after auto-generated content
    """
    lines = []
    
    # Hugo front matter
    lines.append("---")
    if subcommands_only:
        # For subcommand index pages
        title = f"{cmd_info['full_name']}".replace('cli', 'wandb')
        lines.append(f"title: {title}")
        # Suppress automatic child page listing for subcommand pages
        lines.append("no_list: true")
    else:
        lines.append("title: Command Line Interface")
        # Weight for proper ordering in navigation (only for main CLI page)
        lines.append("weight: 2")
        # Suppress automatic child page listing for main CLI page
        lines.append("no_list: true")
    lines.append("---")
    lines.append("")
    
    # Insert preserved manual content before auto-generated content
    if content_before:
        lines.append(content_before)
        lines.append("")
    
    # Auto-generated marker start
    if not subcommands_only:
        lines.append("<!-- AUTO-GENERATED CONTENT STARTS HERE -->")
        lines.append("<!-- WARNING: Do not edit below this line. This content is auto-generated by scripts/cli-docs-generator.py -->")
        lines.append("")
    
    # Description (only if no preserved content)
    if not content_before and cmd_info['help']:
        lines.append(cmd_info['help'])
        lines.append("")
    
    # Usage section
    lines.append("## Basic Usage" if content_before else "## Usage")
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
            desc = opt['help'] or "No description available"
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
            # Link to the command page
            if subcommands_only:
                # Check if the subcommand has its own subcommands
                if subcmd_info['is_group'] and subcmd_info['subcommands']:
                    # Link to the subdirectory index
                    # Get the full parent prefix from the full_name
                    parent_parts = cmd_info['full_name'].replace('cli', 'wandb').split()
                    parent_prefix = '-'.join(parent_parts)
                    lines.append(f"| [{name}]({{{{< relref \"{parent_prefix}-{name}/_index\" >}}}}) | {desc} |")
                else:
                    # Link to the single file
                    # Get the full parent prefix from the full_name
                    parent_parts = cmd_info['full_name'].replace('cli', 'wandb').split()
                    parent_prefix = '-'.join(parent_parts)
                    lines.append(f"| [{name}]({{{{< relref \"{parent_prefix}-{name}\" >}}}}) | {desc} |")
            else:
                lines.append(f"| [{name}]({{{{< relref \"wandb-{name}\" >}}}}) | {desc} |")
        lines.append("")
    
    # Auto-generated marker end and preserved content after
    if not subcommands_only:
        lines.append("<!-- AUTO-GENERATED CONTENT ENDS HERE -->")
        if content_after:
            lines.append("<!-- Manual content continues below -->")
            lines.append("")
            lines.append(content_after)
    
    return "\n".join(lines)

def extract_manual_content(file_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Extract manually written content from existing _index.md file.
    
    Returns a tuple of (content_before, content_after) where:
    - content_before: Content between front matter and auto-generated marker
    - content_after: Content after the auto-generated section ends
    Both can be None if the file doesn't exist or has no manual content.
    """
    if not file_path.exists():
        return (None, None)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find the end of front matter
    if not content.startswith('---'):
        return (None, None)
    
    # Find the second '---' that ends the front matter
    front_matter_end = content.find('---', 3)
    if front_matter_end == -1:
        return (None, None)
    
    # Move past the closing --- and newline
    content_start = front_matter_end + 3
    while content_start < len(content) and content[content_start] in ['\n', '\r']:
        content_start += 1
    
    # Find the auto-generated markers
    auto_gen_start_marker = "<!-- AUTO-GENERATED CONTENT STARTS HERE -->"
    auto_gen_end_marker = "<!-- AUTO-GENERATED CONTENT ENDS HERE -->"
    
    auto_gen_start = content.find(auto_gen_start_marker)
    auto_gen_end = content.find(auto_gen_end_marker)
    
    content_before = None
    content_after = None
    
    if auto_gen_start == -1:
        # No marker found, check if there's a "## Usage" or "## Basic Usage" section
        usage_match = re.search(r'\n## (?:Basic )?Usage\n', content)
        if usage_match:
            # Content before Usage section is likely manual
            content_before = content[content_start:usage_match.start()].strip()
    else:
        # Extract content between front matter and auto-generated marker
        content_before = content[content_start:auto_gen_start].strip()
        
        # If we have an end marker, extract content after it
        if auto_gen_end != -1:
            # Find content after the end marker and following comment line
            content_after_start = content.find('\n', auto_gen_end) + 1
            # Skip the "Manual content continues below" comment if present
            manual_comment = "<!-- Manual content continues below -->"
            if content[content_after_start:].strip().startswith(manual_comment):
                content_after_start = content.find('\n', content_after_start + len(manual_comment)) + 1
            
            remaining = content[content_after_start:].strip()
            if remaining:
                content_after = remaining
    
    return (content_before, content_after)

def write_command_docs(cmd_info: Dict[str, Any], output_dir: Path, parent_path: str = ""):
    """Write documentation files for a command and its subcommands."""
    cmd_name = cmd_info['name']
    
    # Check if this is the root wandb command (either 'cli' or 'wandb' or None)
    is_root = cmd_name in ['wandb', 'cli', None] and parent_path == ""
    
    if is_root:
        # Main CLI index
        index_path = output_dir / "_index.md"
        
        # Extract any existing manual content before regenerating
        content_before, content_after = extract_manual_content(index_path)
        
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, 'w') as f:
            f.write(generate_index_markdown(cmd_info, content_before=content_before, content_after=content_after))
        print(f"Generated: {index_path}")
        if content_before or content_after:
            print(f"  ✓ Preserved manual content")
        
        # Process top-level commands (alphabetically sorted)
        for name, subcmd_info in sorted(cmd_info['subcommands'].items()):
            write_command_docs(subcmd_info, output_dir, "")
    else:
        # Determine if this command has subcommands
        has_subcommands = cmd_info['is_group'] and cmd_info['subcommands']
        
        if has_subcommands:
            # Create a directory for the command group
            cmd_dir = output_dir / f"wandb-{parent_path}{cmd_name}"
            cmd_dir.mkdir(parents=True, exist_ok=True)
            
            # Write index file for the command group
            index_path = cmd_dir / "_index.md"
            with open(index_path, 'w') as f:
                f.write(generate_index_markdown(cmd_info, subcommands_only=True))
            print(f"Generated: {index_path}")
            
            # Write individual subcommand files (alphabetically sorted)
            for subcmd_name, subcmd_info in sorted(cmd_info['subcommands'].items()):
                # Check if this subcommand also has subcommands
                if subcmd_info['is_group'] and subcmd_info['subcommands']:
                    # Handle nested command groups recursively
                    # Pass the cmd_dir as the new output_dir to nest properly
                    write_command_docs(subcmd_info, cmd_dir, f"{parent_path}{cmd_name}-")
                else:
                    # Write single subcommand file
                    subcmd_path = cmd_dir / f"wandb-{parent_path}{cmd_name}-{subcmd_name}.md"
                    with open(subcmd_path, 'w') as f:
                        f.write(generate_markdown(subcmd_info))
                    print(f"Generated: {subcmd_path}")
        else:
            # Write single command file
            cmd_path = output_dir / f"wandb-{parent_path}{cmd_name}.md"
            with open(cmd_path, 'w') as f:
                f.write(generate_markdown(cmd_info))
            print(f"Generated: {cmd_path}")

def main():
    """Main function to generate CLI documentation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate W&B CLI documentation')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Prompt for confirmation before overwriting existing documentation')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Custom output directory (default: content/en/ref/cli)')
    args = parser.parse_args()
    
    # Get the script's parent directory (scripts/) and then go up to the project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = project_root / "content" / "en" / "ref" / "cli"
    
    print("Generating W&B CLI documentation...")
    print(f"Output directory: {output_dir}")
    
    # Warning about overwriting existing documentation (only in interactive mode)
    if output_dir.exists() and any(output_dir.iterdir()):
        # Check if _index.md has manual content
        index_path = output_dir / "_index.md"
        content_before, content_after = extract_manual_content(index_path)
        has_manual_content = content_before is not None or content_after is not None
        
        if args.interactive:
            print(f"⚠️  Warning: This will regenerate documentation in {output_dir}")
            if has_manual_content:
                print("   Note: Manual content in _index.md will be preserved.")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
        
        # Clear directory selectively to remove any vestigial files from removed commands
        print("Clearing existing documentation...")
        files_removed = 0
        dirs_removed = 0
        for item in output_dir.iterdir():
            # Skip _index.md if it has manual content
            if item.name == "_index.md" and has_manual_content:
                print(f"  Preserving {item.name} (contains manual content)")
                continue
            
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
        # Import W&B CLI - the cli module contains a Click group called 'cli'
        from wandb.cli.cli import cli as wandb_cli
        
        # Extract command information
        print("Extracting CLI structure...")
        cli_info = extract_command_info(wandb_cli)
        
        # Generate documentation files
        print("Generating markdown files...")
        write_command_docs(cli_info, output_dir)
        
        print(f"\n✅ Documentation generated successfully in {output_dir}")
        print(f"Total files generated: {len(list(output_dir.rglob('*.md')))}")
        print(f"Note: Excluded launch-only commands: launch, launch-agent, launch-sweep, scheduler")
        
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
