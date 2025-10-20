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
    print("  Using remote OpenAPI spec: https://trace.wandb.ai/openapi.json")
    print("  Mintlify will generate documentation for all 41 endpoints")
    print("")
    
    # Create the service-api directory structure
    service_api_dir = Path("weave/reference/service-api")
    service_api_dir.mkdir(parents=True, exist_ok=True)
    
    # Create an index file if it doesn't exist
    index_file = service_api_dir / "index.mdx"
    if not index_file.exists():
        index_content = """---
title: "Service API"
description: "REST API endpoints for the Weave service"
---

# Weave Service API

The Weave Service API provides REST endpoints for interacting with the Weave tracing service.

## Available Endpoints

This documentation is automatically generated from the OpenAPI specification at https://trace.wandb.ai/openapi.json.

The API includes endpoints for:
- **Calls**: Start, end, update, query, and manage traces
- **Tables**: Create, update, and query data tables
- **Files**: Upload and manage file attachments
- **Objects**: Store and retrieve versioned objects
- **Feedback**: Collect and query user feedback
- **Costs**: Track and query usage costs
- **Inference**: OpenAI-compatible inference endpoints

## Authentication

Most endpoints require authentication. Include your W&B API key in the request headers:

```
Authorization: Bearer YOUR_API_KEY
```

## Base URL

All API requests should be made to:

```
https://trace.wandb.ai
```
"""
        index_file.write_text(index_content)
        print(f"✓ Created Service API index at {index_file}")
    
    print("✓ Service API setup complete!")


if __name__ == "__main__":
    main()