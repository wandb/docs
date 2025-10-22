#!/usr/bin/env python3
"""
Master script to generate and update all Weave reference documentation.

This script:
1. Syncs the OpenAPI spec
2. Generates Python SDK docs
3. Generates TypeScript SDK docs  
4. Updates Service API landing page
5. Fixes broken links including Service API links
6. Updates the TOC
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ✗ Failed: {result.stderr}")
            return False
        print(f"  ✓ Success")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """Main function to orchestrate all generation steps."""
    print("=" * 60)
    print("Weave Reference Documentation Generation")
    print("=" * 60)
    
    scripts_dir = Path(__file__).parent
    
    # Step 1: Sync OpenAPI spec
    print("\n1. Syncing OpenAPI specification")
    if not run_command(
        [sys.executable, str(scripts_dir / "sync_openapi_spec.py")],
        "  Syncing OpenAPI spec"
    ):
        print("  ⚠ Warning: Could not sync OpenAPI spec, continuing with existing")
    
    # Step 2: Generate Python SDK docs
    print("\n2. Generating Python SDK documentation")
    if not run_command(
        [sys.executable, str(scripts_dir / "generate_python_sdk_minimal.py")],
        "  Generating Python SDK docs"
    ):
        print("  ✗ Failed to generate Python SDK docs")
        return 1
    
    # Step 3: Generate TypeScript SDK docs
    print("\n3. Generating TypeScript SDK documentation")
    if not run_command(
        [sys.executable, str(scripts_dir / "generate_typescript_sdk_docs.py")],
        "  Generating TypeScript SDK docs"
    ):
        print("  ✗ Failed to generate TypeScript SDK docs")
        return 1
    
    # Step 4: Update Service API landing page
    print("\n4. Updating Service API landing page")
    if not run_command(
        [sys.executable, str(scripts_dir / "update_service_api_landing.py")],
        "  Updating Service API endpoints list"
    ):
        print("  ⚠ Warning: Could not update Service API landing page")
    
    # Step 5: Fix broken links
    print("\n5. Fixing broken links")
    
    # First fix general broken links
    if not run_command(
        [sys.executable, str(scripts_dir / "fix_broken_links.py")],
        "  Fixing general broken links"
    ):
        print("  ⚠ Warning: Some links could not be fixed")
    
    # Then fix Service API specific links
    if not run_command(
        [sys.executable, str(scripts_dir / "fix_service_api_links.py")],
        "  Fixing Service API links"
    ):
        print("  ⚠ Warning: Some Service API links could not be fixed")
    
    # Step 6: Update TOC
    print("\n6. Updating table of contents")
    if not run_command(
        [sys.executable, str(scripts_dir / "update_weave_toc.py")],
        "  Updating docs.json TOC"
    ):
        print("  ⚠ Warning: Could not update TOC")
    
    print("\n" + "=" * 60)
    print("✓ Weave reference documentation generation complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. Review the changes with: git diff")
    print("  2. Test locally with: npm run dev")
    print("  3. Run link checker: npm run broken-links")
    print("  4. Commit the changes")
    
    return 0


if __name__ == "__main__":
    exit(main())
