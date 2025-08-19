#!/usr/bin/env python3
"""
Auto-fix common link issues based on lychee output
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.parse


def load_lychee_json(filepath: str) -> Dict:
    """Load lychee JSON output file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_redirect_url(status_text: str) -> str:
    """Extract redirect URL from status text if available"""
    # Look for patterns like "301 Moved Permanently: https://new-url.com"
    match = re.search(r':\s*(https?://[^\s]+)', status_text)
    if match:
        return match.group(1).rstrip('.')
    return ""


def fix_common_issues(url: str, status: Dict) -> Tuple[str, str]:
    """
    Fix common URL issues based on the error status
    Returns: (fixed_url, fix_description)
    """
    status_code = status.get('code', 0)
    status_text = status.get('text', '')
    
    # Handle redirects
    if status_code in [301, 302, 307, 308]:
        new_url = extract_redirect_url(status_text)
        if new_url:
            return new_url, f"Updated redirect: {url} → {new_url}"
    
    # Handle common URL transformations
    parsed = urllib.parse.urlparse(url)
    
    # Fix http to https
    if parsed.scheme == 'http' and status_code == 0:
        https_url = url.replace('http://', 'https://', 1)
        return https_url, f"Updated protocol: http → https"
    
    # Remove trailing slashes for non-directory URLs
    if url.endswith('/') and not url.endswith('//'):
        no_slash = url.rstrip('/')
        return no_slash, "Removed trailing slash"
    
    # Fix common GitHub raw content URLs
    if 'github.com' in parsed.netloc and '/blob/' in parsed.path:
        raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
        return raw_url, "Fixed GitHub raw content URL"
    
    return "", ""


def update_file_links(file_path: str, replacements: List[Tuple[str, str, str]]):
    """Update links in a file"""
    if not replacements:
        return
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_url, new_url, description in replacements:
            # Escape special regex characters in the URL
            escaped_url = re.escape(old_url)
            # Replace the URL in various contexts (markdown links, plain URLs, etc.)
            patterns = [
                f'({escaped_url})',  # Plain URL
                f'\\[([^\\]]+)\\]\\({escaped_url}\\)',  # Markdown link
                f'href="{escaped_url}"',  # HTML href
                f"href='{escaped_url}'",  # HTML href with single quotes
            ]
            
            for pattern in patterns:
                if pattern.startswith('\\['):
                    # For markdown links, preserve the link text
                    content = re.sub(pattern, f'[\\1]({new_url})', content)
                elif 'href=' in pattern:
                    # For HTML links
                    content = re.sub(pattern, pattern.replace(escaped_url, new_url), content)
                else:
                    # For plain URLs
                    content = re.sub(pattern, new_url, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated {file_path}:")
            for _, _, description in replacements:
                print(f"  - {description}")
    
    except Exception as e:
        print(f"Error updating {file_path}: {e}")


def process_and_fix_links(lychee_json_path: str):
    """Main function to process and fix links"""
    data = load_lychee_json(lychee_json_path)
    
    # Track fixes by file
    fixes_by_file = {}
    
    # Process fail_map for fixable errors
    if 'fail_map' in data:
        for file_path, links in data['fail_map'].items():
            for link in links:
                url = link['url']
                status = link['status']
                
                # Try to fix the URL
                fixed_url, description = fix_common_issues(url, status)
                
                if fixed_url:
                    if file_path not in fixes_by_file:
                        fixes_by_file[file_path] = []
                    fixes_by_file[file_path].append((url, fixed_url, description))
    
    # Apply fixes
    total_fixes = 0
    for file_path, fixes in fixes_by_file.items():
        update_file_links(file_path, fixes)
        total_fixes += len(fixes)
    
    print(f"\nTotal fixes applied: {total_fixes} across {len(fixes_by_file)} files")
    
    # Generate fix summary for PR
    if total_fixes > 0:
        with open('lychee-fixes-summary.md', 'w') as f:
            f.write("## Auto-fixed Links Summary\n\n")
            f.write(f"This PR automatically fixed {total_fixes} link issues:\n\n")
            
            for file_path, fixes in sorted(fixes_by_file.items()):
                f.write(f"### `{file_path}`\n")
                for old_url, new_url, description in fixes:
                    f.write(f"- {description}\n")
                f.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python auto_fix_links.py <lychee_output.json>")
        sys.exit(1)
    
    lychee_json_path = sys.argv[1]
    
    if not Path(lychee_json_path).exists():
        print(f"Error: File '{lychee_json_path}' not found")
        sys.exit(1)
    
    process_and_fix_links(lychee_json_path)