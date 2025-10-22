#!/usr/bin/env python3
"""
Master script to generate and update Training API reference documentation.

This script:
1. Syncs the OpenAPI spec
2. Updates Training API landing page
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
    print("Training API Reference Documentation Generation")
    print("=" * 60)
    
    scripts_dir = Path(__file__).parent
    
    # Step 1: Sync OpenAPI spec
    print("\n1. Syncing OpenAPI specification")
    if not run_command(
        [sys.executable, str(scripts_dir / "sync_openapi_spec.py")],
        "  Syncing OpenAPI spec"
    ):
        print("  ⚠ Warning: Could not sync OpenAPI spec, continuing with existing")
    
    # Step 2: Update Training API landing page
    print("\n2. Updating Training API landing page")
    if not run_command(
        [sys.executable, str(scripts_dir / "update_training_api_landing.py")],
        "  Updating Training API endpoints list"
    ):
        print("  ⚠ Warning: Could not update Training API landing page")
    
    print("\n" + "=" * 60)
    print("✓ Training API reference documentation generation complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("  1. Review the changes with: git diff")
    print("  2. Test locally with: npm run dev")
    print("  3. Commit the changes")
    
    return 0


if __name__ == "__main__":
    exit(main())
