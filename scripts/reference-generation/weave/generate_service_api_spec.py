#!/usr/bin/env python3
"""
Set up Service API documentation to use the remote OpenAPI specification.

The Service API documentation uses the OpenAPI spec directly from the Weave service.
Mintlify will fetch and process this spec to generate the API documentation.
"""

from pathlib import Path


def main():
    """Main function."""
    print("Service API configuration:")
    print("  Using OpenAPI spec from sync_openapi_spec.py")
    print("  Mintlify will generate documentation for endpoints")
    print("")
    
    # Create the service-api directory structure for openapi.json
    # Note: The landing page is service-api.mdx (not service-api/index.mdx)
    # and is managed by update_service_api_landing.py
    service_api_dir = Path("weave/reference/service-api")
    service_api_dir.mkdir(parents=True, exist_ok=True)
    
    print("✓ Service API directory structure ready")
    print("  Note: Landing page at weave/reference/service-api.mdx")
    print("  Note: OpenAPI spec at weave/reference/service-api/openapi.json")


if __name__ == "__main__":
    main()