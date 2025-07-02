#!/usr/bin/env python3
"""
Alt Text Checker for W&B Documentation

This script checks for missing or empty alt text in:
- HTML img tags
- Markdown image syntax
- Hugo img shortcodes

It analyzes context and suggests appropriate alt text based on:
- Surrounding content
- Existing alt text patterns in the repository
- Best practices for accessibility
"""

import os
import re
import sys
import json
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup
import subprocess

class AltTextChecker:
    def __init__(self):
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.pr_number = os.environ.get('PR_NUMBER')
        self.repo_owner = os.environ.get('REPO_OWNER')
        self.repo_name = os.environ.get('REPO_NAME')
        self.changed_files = os.environ.get('CHANGED_FILES', '').split()
        
        # Patterns for different image formats
        self.hugo_img_pattern = re.compile(
            r'{{<\s*img\s+([^>]*)>}}',
            re.IGNORECASE | re.MULTILINE
        )
        
        self.markdown_img_pattern = re.compile(
            r'!\[([^\]]*)\]\(([^)]+)\)',
            re.MULTILINE
        )
        
        self.html_img_pattern = re.compile(
            r'<img\s+([^>]*)>',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Store found issues
        self.issues = []
        
        # Store existing alt text for learning patterns
        self.existing_alt_texts = []

    def extract_hugo_attributes(self, attr_string: str) -> Dict[str, str]:
        """Extract attributes from Hugo shortcode string"""
        attrs = {}
        # Match key="value" or key=value patterns
        attr_pattern = re.compile(r'(\w+)=(["\']?)([^"\'\s]+)\2')
        matches = attr_pattern.findall(attr_string)
        
        for key, quote, value in matches:
            attrs[key.lower()] = value
            
        return attrs

    def extract_html_attributes(self, attr_string: str) -> Dict[str, str]:
        """Extract attributes from HTML img tag"""
        attrs = {}
        # Use BeautifulSoup for robust HTML parsing
        try:
            soup = BeautifulSoup(f'<img {attr_string}>', 'html.parser')
            img_tag = soup.find('img')
            if img_tag:
                attrs = dict(img_tag.attrs)
        except Exception:
            # Fallback to regex if BeautifulSoup fails
            attr_pattern = re.compile(r'(\w+)=(["\']?)([^"\'\s]+)\2')
            matches = attr_pattern.findall(attr_string)
            for key, quote, value in matches:
                attrs[key.lower()] = value
                
        return attrs

    def get_context_around_line(self, content: str, line_num: int, context_lines: int = 3) -> str:
        """Get context around a specific line for better alt text suggestions"""
        lines = content.split('\n')
        start = max(0, line_num - context_lines)
        end = min(len(lines), line_num + context_lines + 1)
        
        context_lines = lines[start:end]
        # Remove the image line itself to focus on surrounding context
        context_text = '\n'.join([line for i, line in enumerate(context_lines) 
                                 if i != (line_num - start)])
        
        # Clean up markdown syntax for better context understanding
        context_text = re.sub(r'[#*`_\[\](){}]', '', context_text)
        context_text = re.sub(r'\s+', ' ', context_text).strip()
        
        return context_text

    def suggest_alt_text(self, image_src: str, context: str, existing_patterns: List[str]) -> str:
        """Generate suggested alt text based on image source, context, and existing patterns"""
        # Extract meaningful parts from image path
        src_parts = Path(image_src).stem.lower()
        src_parts = re.sub(r'[_-]', ' ', src_parts)
        
        # Look for key terms in context
        context_lower = context.lower()
        
        # Common W&B terms and their descriptions
        wb_terms = {
            'dashboard': 'dashboard view',
            'sweep': 'hyperparameter sweep',
            'artifact': 'artifact management',
            'report': 'W&B report',
            'chart': 'chart visualization',
            'table': 'data table',
            'metric': 'metrics display',
            'experiment': 'experiment tracking',
            'model': 'model management',
            'training': 'training progress',
            'validation': 'validation results',
            'config': 'configuration settings',
            'launch': 'W&B Launch',
            'registry': 'model registry',
            'login': 'login interface',
            'setup': 'setup process',
            'integration': 'integration example',
            'ui': 'user interface',
            'workflow': 'workflow diagram',
            'architecture': 'architecture diagram'
        }
        
        # Check for W&B specific terms
        for term, description in wb_terms.items():
            if term in context_lower or term in src_parts:
                if 'gif' in image_src.lower():
                    return f"{description.capitalize()} animation"
                else:
                    return f"{description.capitalize()}"
        
        # Generic suggestions based on file type and context
        if 'gif' in image_src.lower():
            if 'tutorial' in context_lower or 'example' in context_lower:
                return "Tutorial demonstration"
            return "Step-by-step process"
        
        if any(word in context_lower for word in ['tutorial', 'example', 'demo']):
            return "Tutorial example"
        
        if any(word in context_lower for word in ['result', 'output', 'visualization']):
            return "Example output"
        
        if any(word in src_parts for word in ['diagram', 'architecture', 'flow']):
            return "Architecture diagram"
        
        # Analyze existing patterns for similar contexts
        for existing_alt in existing_patterns:
            if len(existing_alt) > 5:  # Skip very short alt texts
                alt_words = set(existing_alt.lower().split())
                context_words = set(context_lower.split())
                if len(alt_words.intersection(context_words)) >= 2:
                    return existing_alt
        
        # Fallback suggestions
        return "W&B interface screenshot"

    def collect_existing_alt_texts(self):
        """Collect existing alt text patterns from the repository for learning"""
        print("Collecting existing alt text patterns...")
        
        try:
            # Search for files with img tags in content directory
            result = subprocess.run([
                'find', 'content', '-name', '*.md', '-type', 'f'
            ], capture_output=True, text=True, check=True)
            
            md_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            for file_path in md_files[:50]:  # Limit to avoid excessive processing
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Find Hugo shortcodes with alt text
                    for match in self.hugo_img_pattern.finditer(content):
                        attrs = self.extract_hugo_attributes(match.group(1))
                        alt_text = attrs.get('alt')
                        if alt_text and len(alt_text.strip()) > 3:
                            self.existing_alt_texts.append(alt_text.strip())
                    
                    # Find HTML img tags with alt text
                    for match in self.html_img_pattern.finditer(content):
                        attrs = self.extract_html_attributes(match.group(1))
                        alt_text = attrs.get('alt')
                        if alt_text and len(alt_text.strip()) > 3:
                            self.existing_alt_texts.append(alt_text.strip())
                    
                    # Find Markdown images with alt text
                    for match in self.markdown_img_pattern.finditer(content):
                        alt_text = match.group(1).strip()
                        if alt_text and len(alt_text) > 3:
                            self.existing_alt_texts.append(alt_text)
                            
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error collecting existing alt texts: {e}")
        
        # Remove duplicates and keep unique patterns
        self.existing_alt_texts = list(set(self.existing_alt_texts))
        print(f"Collected {len(self.existing_alt_texts)} existing alt text patterns")

    def check_file(self, file_path: str):
        """Check a single file for alt text issues"""
        print(f"Checking file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return
        
        lines = content.split('\n')
        
        # Check Hugo img shortcodes
        for match in self.hugo_img_pattern.finditer(content):
            line_num = content[:match.start()].count('\n')
            attrs = self.extract_hugo_attributes(match.group(1))
            src = attrs.get('src', '')
            alt = attrs.get('alt', '').strip()
            
            if not alt:
                context = self.get_context_around_line(content, line_num)
                suggested_alt = self.suggest_alt_text(src, context, self.existing_alt_texts)
                
                self.issues.append({
                    'file': file_path,
                    'line': line_num + 1,
                    'type': 'Hugo img shortcode',
                    'issue': 'Missing alt text',
                    'current': match.group(0),
                    'suggested': f'{{{{< img src="{src}" alt="{suggested_alt}" >}}}}',
                    'context': context[:100] + '...' if len(context) > 100 else context
                })
        
        # Check HTML img tags
        for match in self.html_img_pattern.finditer(content):
            line_num = content[:match.start()].count('\n')
            attrs = self.extract_html_attributes(match.group(1))
            src = attrs.get('src', '')
            alt = attrs.get('alt', '').strip()
            
            if not alt:
                context = self.get_context_around_line(content, line_num)
                suggested_alt = self.suggest_alt_text(src, context, self.existing_alt_texts)
                
                # Reconstruct img tag with alt text
                other_attrs = ' '.join([f'{k}="{v}"' for k, v in attrs.items() if k != 'alt'])
                suggested_tag = f'<img {other_attrs} alt="{suggested_alt}">'
                
                self.issues.append({
                    'file': file_path,
                    'line': line_num + 1,
                    'type': 'HTML img tag',
                    'issue': 'Missing alt text',
                    'current': match.group(0),
                    'suggested': suggested_tag,
                    'context': context[:100] + '...' if len(context) > 100 else context
                })
        
        # Check Markdown images
        for match in self.markdown_img_pattern.finditer(content):
            line_num = content[:match.start()].count('\n')
            alt_text = match.group(1).strip()
            src = match.group(2)
            
            if not alt_text:
                context = self.get_context_around_line(content, line_num)
                suggested_alt = self.suggest_alt_text(src, context, self.existing_alt_texts)
                
                self.issues.append({
                    'file': file_path,
                    'line': line_num + 1,
                    'type': 'Markdown image',
                    'issue': 'Missing alt text',
                    'current': match.group(0),
                    'suggested': f'![{suggested_alt}]({src})',
                    'context': context[:100] + '...' if len(context) > 100 else context
                })

    def create_pr_comment(self) -> str:
        """Create a formatted PR comment with all issues and suggestions"""
        if not self.issues:
            return ""
        
        comment = """## 🖼️ Alt Text Issues Found

This PR contains image references that are missing alt text. Alt text is important for accessibility and helps screen reader users understand the content of images.

### Issues Found:

"""
        
        for i, issue in enumerate(self.issues, 1):
            comment += f"""
#### {i}. {issue['type']} in `{issue['file']}` (line {issue['line']})

**Issue:** {issue['issue']}

**Current:**
```
{issue['current']}
```

**Suggested:**
```
{issue['suggested']}
```

**Context:** {issue['context']}

---
"""
        
        comment += f"""
### How to Fix

1. Replace the current image syntax with the suggested version that includes alt text
2. The suggested alt text is based on the surrounding context and existing patterns in the repository
3. Feel free to customize the alt text to better describe the specific image content

**Found {len(self.issues)} image(s) missing alt text.** Please add descriptive alt text to make the documentation more accessible.

<details>
<summary>About Alt Text</summary>

Alt text should:
- Be concise and descriptive
- Explain what the image shows in context
- Skip redundant phrases like "image of" or "picture showing"
- Be empty (`alt=""`) only for purely decorative images

For more information, see the [Web Content Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/Understanding/images-of-text.html).
</details>
"""
        
        return comment

    def post_pr_comment(self, comment: str):
        """Post a comment to the GitHub PR"""
        if not self.github_token or not self.pr_number:
            print("Missing GitHub token or PR number, cannot post comment")
            return
        
        url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/issues/{self.pr_number}/comments"
        
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        data = {'body': comment}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code == 201:
                print("Successfully posted PR comment")
            else:
                print(f"Failed to post PR comment: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error posting PR comment: {e}")

    def run(self):
        """Main execution function"""
        print("Starting Alt Text Checker...")
        print(f"Checking {len(self.changed_files)} changed files")
        
        # First collect existing alt text patterns
        self.collect_existing_alt_texts()
        
        # Check each changed file
        for file_path in self.changed_files:
            if file_path.endswith(('.md', '.html')):
                self.check_file(file_path)
        
        print(f"\nFound {len(self.issues)} alt text issues")
        
        if self.issues:
            comment = self.create_pr_comment()
            print("\nGenerated PR comment:")
            print(comment)
            print("\n" + "="*50)
            
            self.post_pr_comment(comment)
        else:
            print("No alt text issues found. Great job! 🎉")

if __name__ == "__main__":
    checker = AltTextChecker()
    checker.run() 