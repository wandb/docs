#!/usr/bin/env python3
"""
Simple test script for the Alt Text Checker
"""

import tempfile
import os
import sys
from pathlib import Path

# Add the script directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from pathlib import Path
import importlib.util

# Load the alt-text-checker module
script_path = Path(__file__).parent / "alt-text-checker.py"
spec = importlib.util.spec_from_file_location("alt_text_checker", script_path)
if spec is None or spec.loader is None:
    raise ImportError("Could not load alt-text-checker.py")

alt_text_checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alt_text_checker)

AltTextChecker = alt_text_checker.AltTextChecker

def create_test_content():
    """Create test markdown content with various image formats"""
    return """
# Test Documentation

This is a sample documentation page for testing the alt text checker.

## Hugo Shortcodes

Here's an image with missing alt text:
{{< img src="/images/dashboard.png" >}}

Here's an image with alt text (should not be flagged):
{{< img src="/images/working.png" alt="Working example" >}}

## Markdown Images

Missing alt text:
![](https://example.com/image.png)

With alt text (should not be flagged):
![Example image](https://example.com/good-image.png)

## HTML Images

Missing alt text:
<img src="/images/html-test.png">

With alt text (should not be flagged):
<img src="/images/html-good.png" alt="Good HTML image">

This section talks about W&B dashboard and experiment tracking to provide context.
"""

def test_alt_text_checker():
    """Test the alt text checker functionality"""
    print("Testing Alt Text Checker...")
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(create_test_content())
        test_file_path = f.name
    
    try:
        # Initialize checker with test environment
        os.environ['CHANGED_FILES'] = test_file_path
        os.environ['GITHUB_TOKEN'] = 'test_token'
        os.environ['PR_NUMBER'] = '123'
        os.environ['REPO_OWNER'] = 'test_owner'
        os.environ['REPO_NAME'] = 'test_repo'
        
        checker = AltTextChecker()
        
        # Check the test file
        checker.check_file(test_file_path)
        
        print(f"\nTest Results:")
        print(f"Found {len(checker.issues)} alt text issues:")
        
        for i, issue in enumerate(checker.issues, 1):
            print(f"\n{i}. {issue['type']} - {issue['issue']}")
            print(f"   Current: {issue['current']}")
            print(f"   Suggested: {issue['suggested']}")
        
        # Verify expected issues were found
        expected_issues = 3  # 1 Hugo, 1 Markdown, 1 HTML
        if len(checker.issues) == expected_issues:
            print(f"\n✅ Test PASSED: Found expected {expected_issues} issues")
        else:
            print(f"\n❌ Test FAILED: Expected {expected_issues} issues, found {len(checker.issues)}")
        
        # Test comment generation
        if checker.issues:
            comment = checker.create_pr_comment()
            print(f"\nGenerated comment length: {len(comment)} characters")
            print("Comment preview:")
            print(comment[:200] + "..." if len(comment) > 200 else comment)
        
    finally:
        # Clean up
        os.unlink(test_file_path)
        print(f"\nCleaned up test file: {test_file_path}")

if __name__ == "__main__":
    test_alt_text_checker() 