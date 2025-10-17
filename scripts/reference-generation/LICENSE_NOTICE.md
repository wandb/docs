# License Notice for Reference Documentation Generation

## Overview

The scripts in this directory are used solely for generating reference documentation during the development/CI process. They are NOT distributed with the Weave library or included in any production code.

## Dependencies and Their Licenses

### Direct Dependencies
- **requests** (Apache-2.0): Used for HTTP requests
- **lazydocs** (MIT): Documentation generator maintained by W&B

### Transitive Dependencies (via lazydocs)
- **setuptools** (MIT with vendored LGPL-3.0 components): Build system
- Various other dependencies with mixed licenses

## Important Notes

1. **Development Only**: These dependencies are only installed temporarily during documentation generation in CI/GitHub Actions. They are never included in the distributed Weave package.

2. **No Distribution**: The generated documentation consists only of MDX/Markdown files with no executable code or dependencies.

3. **Isolated Execution**: The GitHub Action runs these scripts in isolated virtual environments that are destroyed after use.

4. **License Compliance**: Since these tools are not distributed with Weave, the LGPL-3.0 components in setuptools' vendored dependencies do not create license obligations for Weave users.

## For Organizations with Strict License Policies

If your organization has policies against any LGPL code in development tools:

1. Use the GitHub Action to generate docs in the cloud (recommended)
2. Use the minimal Python generator that avoids lazydocs
3. Generate documentation in a Docker container
4. Request an exception for development-only tools

## Socket Security

The `.socketignore` file in the repository root excludes these scripts from security scanning since they are development tools, not production code.

### Known Socket Security Warnings

- **Native code in wheel**: The `wheel` package contains native code, which is normal for Python packaging tools
- **License violations**: Some transitive dependencies may have LGPL or other licenses that trigger policy warnings

These warnings are acceptable because:
1. The tools are only used during documentation generation
2. They run in isolated CI environments
3. They are never distributed with Weave
4. The generated documentation contains no executable code
