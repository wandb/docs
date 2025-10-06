#!/usr/bin/env python3
"""
Convert W&B SDK CHANGELOG.md to Hugo-compatible release notes.

This script parses the SDK's CHANGELOG.md file and generates individual
Hugo markdown files for each release, with various transformations to
make the content more suitable for documentation.

Usage:
    python sdk-changelog-to-hugo.py [--changelog PATH] [--output DIR] [--limit N]

Args:
    --changelog: Path to CHANGELOG.md (default: ../wandb/CHANGELOG.md)
    --output: Output directory (default: ./content/en/ref/release-notes/sdk)
    --limit: Limit number of releases to process (default: all)

Examples:
    # Process all releases
    python sdk-changelog-to-hugo.py
    
    # Process only the 5 most recent releases
    python sdk-changelog-to-hugo.py --limit 5
"""

import os
import re
import argparse
from datetime import datetime
from pathlib import Path


# Common conventional commit prefixes to remove
COMMIT_PREFIXES = [
    r'^feat\([^)]+\):\s*',
    r'^fix\([^)]+\):\s*',
    r'^docs\([^)]+\):\s*',
    r'^style\([^)]+\):\s*',
    r'^refactor\([^)]+\):\s*',
    r'^perf\([^)]+\):\s*',
    r'^test\([^)]+\):\s*',
    r'^build\([^)]+\):\s*',
    r'^ci\([^)]+\):\s*',
    r'^chore\([^)]+\):\s*',
    r'^revert\([^)]+\):\s*',
    # Without scope
    r'^feat:\s*',
    r'^fix:\s*',
    r'^docs:\s*',
    r'^style:\s*',
    r'^refactor:\s*',
    r'^perf:\s*',
    r'^test:\s*',
    r'^build:\s*',
    r'^ci:\s*',
    r'^chore:\s*',
    r'^revert:\s*',
]


def remove_emojis(text):
    """
    Remove emojis from text.
    
    Args:
        text (str): Text potentially containing emojis.
        
    Returns:
        str: Text with emojis removed.
    """
    # Pattern to match most common emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002500-\U00002BEF"  # chinese char
        "\U00002702-\U000027B0"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"  # dingbats
        "\u3030"
        "]+", 
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text).strip()


def parse_changelog(changelog_path):
    """
    Parse the CHANGELOG.md file and extract releases.
    
    Args:
        changelog_path (str): Path to the CHANGELOG.md file.
        
    Returns:
        list: List of dictionaries containing release information.
    """
    releases = []
    current_release = None
    content_lines = []
    
    with open(changelog_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines until we find the first release
    in_header = True
    
    for line in lines:
        # Check for release header pattern: ## [0.21.1] - 2025-08-07
        release_match = re.match(r'^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})', line)
        
        if release_match:
            # Save previous release if exists
            if current_release and content_lines:
                current_release['content'] = ''.join(content_lines).strip()
                releases.append(current_release)
            
            # Start new release
            in_header = False
            version = release_match.group(1)
            date_str = release_match.group(2)
            
            current_release = {
                'version': version,
                'date': date_str,
                'title': f"v{version}"
            }
            content_lines = []
            
        elif not in_header and current_release:
            # Accumulate content for current release
            content_lines.append(line)
    
    # Save the last release
    if current_release and content_lines:
        current_release['content'] = ''.join(content_lines).strip()
        releases.append(current_release)
    
    return releases


def transform_content(content):
    """
    Transform the changelog content to be more suitable for documentation.
    
    Args:
        content (str): The raw changelog content.
        
    Returns:
        str: The transformed content.
    """
    lines = content.split('\n')
    transformed_lines = []
    in_contributors_section = False
    
    for line in lines:
        # Remove emojis from the line
        line = remove_emojis(line)
        
        # Skip empty lines at the start
        if not transformed_lines and not line.strip():
            continue
        
        # Check if we're entering the Contributors section
        if line.strip().startswith('### Contributors') or line.strip().startswith('## Contributors'):
            in_contributors_section = True
            continue
        
        # Skip everything in the Contributors section
        if in_contributors_section:
            # Check if we're starting a new section (which would end Contributors)
            if line.strip().startswith('###') or line.strip().startswith('##'):
                in_contributors_section = False
                # Process this line normally (it's a new section header)
            else:
                continue  # Skip contributor lines
        
        # Transform list items
        if line.startswith('- '):
            # Check if this is a docs(sdk) item - skip it entirely
            if re.search(r'docs\(sdk\)', line, re.IGNORECASE):
                continue
                
            # Extract the main content before the PR link
            # Handle both formats: "(@user in URL)" and "by @user in URL"
            # Also handle when period comes after the PR info
            match = re.match(r'^- (.+?)(\s+(?:\(@[^)]+\s+in\s+https://[^)]+\)|by\s+@[^)]+\s+in\s+https://[^)]+))?\.?\s*$', line)
            if match:
                main_content = match.group(1)
                pr_info = match.group(2) or ''
                
                # Remove conventional commit prefixes
                for prefix in COMMIT_PREFIXES:
                    main_content = re.sub(prefix, '', main_content, flags=re.IGNORECASE)
                
                # Clean up the main content
                # Ensure it ends with a period
                if main_content and not main_content.endswith(('.', '!', '?')):
                    main_content += '.'
                
                # Fix whitespace issues
                main_content = re.sub(r'\s+', ' ', main_content).strip()
                
                # Create the transformed line
                if pr_info:
                    # Move PR info to HTML comment
                    pr_comment = f"<!-- {pr_info.strip()} -->"
                    transformed_line = f"- {main_content} {pr_comment}"
                else:
                    transformed_line = f"- {main_content}"
                
                transformed_lines.append(transformed_line)
            else:
                # Keep the line as-is if it doesn't match our pattern
                transformed_lines.append(line)
        elif line.startswith('#'):
            # Adjust heading levels - reduce by one level (### becomes ##, ## becomes #)
            # But don't go below H2 for the content (H1 is reserved for page title)
            if line.startswith('### '):
                # H3 becomes H2
                transformed_lines.append(line.replace('### ', '## ', 1))
            elif line.startswith('#### '):
                # H4 becomes H3
                transformed_lines.append(line.replace('#### ', '### ', 1))
            elif line.startswith('##### '):
                # H5 becomes H4
                transformed_lines.append(line.replace('##### ', '#### ', 1))
            elif line.startswith('###### '):
                # H6 becomes H5
                transformed_lines.append(line.replace('###### ', '##### ', 1))
            else:
                # Keep H1 and H2 as-is (though changelog shouldn't have H1)
                transformed_lines.append(line)
        else:
            # Keep other lines as-is (but with emojis removed)
            transformed_lines.append(line)
    
    # Join lines and clean up trailing whitespace
    result = '\n'.join(transformed_lines)
    result = '\n'.join(line.rstrip() for line in result.split('\n'))
    
    return result


def format_hugo_markdown(release):
    """
    Format a release as Hugo-compatible markdown.
    
    Args:
        release (dict): Release information dictionary.
        
    Returns:
        str: Hugo-formatted markdown content.
    """
    # Parse the date to format it nicely
    try:
        date_obj = datetime.strptime(release['date'], '%Y-%m-%d')
        formatted_date = date_obj.strftime('%B %d, %Y')
    except:
        formatted_date = release['date']
    
    # Create Hugo frontmatter
    frontmatter = f"""---
title: "{release['title']}"
date: {release['date']}
description: "{formatted_date}"
---

"""
    
    # Add a call-to-action to view on GitHub
    version_tag = f"v{release['version']}"
    note = f'{{{{% alert color="info" %}}}}View the [{version_tag} changelog](https://github.com/wandb/wandb/releases/tag/{version_tag}) on GitHub.{{{{% /alert %}}}}\n\n'
    
    # Transform the content
    transformed_content = transform_content(release['content'])
    
    # Combine frontmatter, note, and content
    return frontmatter + note + transformed_content


def save_release_note(release, output_dir):
    """
    Save a release note as a Hugo markdown file.
    
    Args:
        release (dict): Release information dictionary.
        output_dir (str): Output directory path.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename (use version without 'v' prefix)
    filename = f"{release['version']}.md"
    filepath = os.path.join(output_dir, filename)
    
    # Format content
    content = format_hugo_markdown(release)
    
    # Write file
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"✓ Created: {filepath}")


def main():
    """Main function to process changelog and generate release notes."""
    parser = argparse.ArgumentParser(description='Convert SDK CHANGELOG to Hugo release notes')
    parser.add_argument('--changelog', 
                       default='/Users/matt.linville/wandb/CHANGELOG.md',
                       help='Path to CHANGELOG.md')
    parser.add_argument('--output', 
                       default='/Users/matt.linville/docs/content/en/ref/release-notes/sdk',
                       help='Output directory for release notes')
    parser.add_argument('--limit', 
                       type=int, 
                       default=None,
                       help='Limit number of releases to process')
    parser.add_argument('--version',
                       help='Specific version to process (e.g., 0.21.1). If not specified, processes all versions.')
    
    args = parser.parse_args()
    
    # Check if changelog exists
    if not os.path.exists(args.changelog):
        print(f"Error: Changelog not found at {args.changelog}")
        return 1
    
    print(f"📖 Reading changelog from: {args.changelog}")
    print(f"📁 Output directory: {args.output}")
    
    # Parse changelog
    releases = parse_changelog(args.changelog)
    print(f"📊 Found {len(releases)} releases")
    
    # Filter for specific version if requested
    if args.version:
        # Normalize version (remove 'v' prefix if present)
        target_version = args.version.lstrip('v')
        print(f"🔍 Looking for version: {target_version}")
        print(f"📋 Available versions: {[r['version'] for r in releases[:10]]}")  # Show first 10
        releases = [r for r in releases if r['version'] == target_version]
        if not releases:
            print(f"❌ Error: Version {target_version} not found in changelog")
            return 1
        print(f"🎯 Processing specific version: v{target_version}")
    # Apply limit if specified
    elif args.limit:
        releases = releases[:args.limit]
        print(f"🎯 Processing {len(releases)} releases (limited)")
    
    # Process each release
    for i, release in enumerate(releases, 1):
        print(f"\n[{i}/{len(releases)}] Processing {release['title']} ({release['date']})")
        save_release_note(release, args.output)
    
    print(f"\n✅ Successfully generated {len(releases)} release note(s)!")
    print(f"📍 Location: {args.output}")
    
    return 0


if __name__ == '__main__':
    exit(main())
