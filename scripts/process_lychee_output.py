#!/usr/bin/env python3
"""
Process lychee JSON output into human-readable markdown report
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def load_lychee_json(filepath: str) -> Dict[str, Any]:
    """Load lychee JSON output file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def categorize_links(data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Categorize links by their status"""
    categorized = defaultdict(list)
    
    # Process fail_map for errors
    if 'fail_map' in data:
        for file_path, links in data['fail_map'].items():
            for link in links:
                link_info = {
                    'file': file_path,
                    'url': link['url'],
                    'status': link['status']
                }
                # Check if it's a redirect (3xx status codes)
                status_code = link['status'].get('code', 0)
                if 300 <= status_code < 400:
                    categorized['redirects'].append(link_info)
                else:
                    categorized['errors'].append(link_info)
    
    # Also check for excluded/unsupported links if needed
    if 'exclude_map' in data:
        for file_path, links in data.get('exclude_map', {}).items():
            for link in links:
                link_info = {
                    'file': file_path,
                    'url': link['url'] if isinstance(link, dict) else link,
                    'status': {'text': 'Excluded', 'code': 0}
                }
                categorized['excluded'].append(link_info)
    
    return dict(categorized)


def generate_markdown_report(data: Dict[str, Any], categorized: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate a human-readable markdown report"""
    report = []
    
    # Header
    report.append("# Lychee Link Check Report\n")
    
    # Summary Statistics
    report.append("## Summary\n")
    report.append(f"- **Total links checked**: {data.get('total', 0)}")
    report.append(f"- **Successful**: {data.get('successful', 0)}")
    report.append(f"- **Errors**: {data.get('errors', 0)}")
    report.append(f"- **Redirects**: {data.get('redirects', 0)}")
    report.append(f"- **Timeouts**: {data.get('timeouts', 0)}")
    report.append(f"- **Excluded**: {data.get('excludes', 0)}")
    report.append(f"- **Cached**: {data.get('cached', 0)}")
    report.append(f"- **Duration**: {data.get('duration_secs', 0)} seconds\n")
    
    # Errors Section
    if categorized.get('errors'):
        report.append("## ❌ Broken Links\n")
        report.append("These links returned errors and need to be fixed:\n")
        
        # Group errors by status code
        errors_by_status = defaultdict(list)
        for error in categorized['errors']:
            status_text = error['status'].get('text', 'Unknown')
            errors_by_status[status_text].append(error)
        
        for status, errors in sorted(errors_by_status.items()):
            report.append(f"### {status}\n")
            for error in sorted(errors, key=lambda x: x['file']):
                report.append(f"- **File**: `{error['file']}`")
                report.append(f"  - **URL**: {error['url']}")
            report.append("")
    
    # Redirects Section (if any)
    if categorized.get('redirects'):
        report.append("## ↪️ Redirected Links\n")
        report.append("These links are redirecting and should be updated:\n")
        
        # Group redirects by file
        redirects_by_file = defaultdict(list)
        for redirect in categorized['redirects']:
            redirects_by_file[redirect['file']].append(redirect)
        
        for file_path, redirects in sorted(redirects_by_file.items()):
            report.append(f"### `{file_path}`\n")
            for redirect in redirects:
                status_text = redirect['status'].get('text', 'Unknown')
                # Try to extract the redirect destination
                if ':' in status_text and 'http' in status_text:
                    parts = status_text.split(':', 1)
                    status_code = parts[0]
                    new_url = parts[1].strip() if len(parts) > 1 else ''
                    report.append(f"- {redirect['url']}")
                    report.append(f"  - **Redirects to**: {new_url}")
                else:
                    report.append(f"- {redirect['url']}")
                    report.append(f"  - **Status**: {status_text}")
            report.append("")
    
    # Action Items
    if categorized.get('errors') or categorized.get('redirects'):
        report.append("## 📋 Action Items\n")
        
        if categorized.get('errors'):
            report.append("### For Broken Links:")
            report.append("1. Review each broken link and determine if:")
            report.append("   - The URL has changed (update to the new URL)")
            report.append("   - The page no longer exists (remove the link or find an alternative)")
            report.append("   - It's a temporary issue (verify and possibly retry later)")
            report.append("")
        
        if categorized.get('redirects'):
            report.append("### For Redirected Links:")
            report.append("1. Update all redirected links to their final destinations")
            report.append("2. This will improve page load times and SEO")
            report.append("")
    
    else:
        report.append("## ✅ All Clear!\n")
        report.append("No broken links or redirects found. Great job maintaining the documentation!")
    
    return "\n".join(report)


def process_lychee_output(input_file: str, output_file: str):
    """Main processing function"""
    # Load the JSON data
    data = load_lychee_json(input_file)
    
    # Categorize the links
    categorized = categorize_links(data)
    
    # Generate the markdown report
    report = generate_markdown_report(data, categorized)
    
    # Write the report
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Report generated: {output_file}")
    
    # Return exit code based on errors
    return 1 if categorized.get('errors') else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_lychee_output.py <input.json> <output.md>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    exit_code = process_lychee_output(input_file, output_file)
    sys.exit(exit_code)