# AI Assistant Resources

This directory contains resources, prompts, and templates specifically designed for AI agents working with the W&B documentation repository.

## Directory Structure

```
.ai/
├── migrations/           # One-off migration tools and templates
├── style-guide/         # Writing and code style guidance for AI agents
└── runbooks/           # Step-by-step procedures for complex tasks
```

## For Human Contributors

These resources help AI assistants provide consistent, high-quality contributions to the docs. When working with an AI agent:

1. **For migrations**: Direct the agent to `.ai/migrations/` for specific migration templates
2. **For style questions**: Reference `.ai/style-guide/` for consistency
3. **For complex procedures**: Use `.ai/runbooks/` for detailed workflows

## For AI Agents

When assisting with documentation tasks:

1. **Check this directory first** for relevant templates and guidance
2. **Follow style guides** in `.ai/style-guide/` to ensure consistency
3. **Use migration templates** in `.ai/migrations/` for structural changes
4. **Reference runbooks** in `.ai/runbooks/` for multi-step procedures

## Current Resources

### Migrations
- **hugo-to-mintlify/**: Templates for migrating PRs from Hugo to Mintlify structure
  - `migration_prompt_template.md`: Step-by-step migration guide
  - `migration_user_prompt.md`: User-facing prompts for requesting migrations

### Style Guide
*(Coming soon)*
- Writing style guidelines
- Code formatting standards
- Naming conventions

### Runbooks
*(Coming soon)*
- Release process
- Documentation testing procedures
- PR review checklist

## Adding New Resources

When adding new AI resources:
1. Place them in the appropriate subdirectory
2. Update this README with a description
3. Include clear instructions for both humans and AI agents
4. Consider whether the resource is temporary (migration) or permanent (style guide)

## Note on Tool Compatibility

These resources are tool-agnostic and work with:
- Cursor
- GitHub Copilot
- Claude (via API or web)
- ChatGPT
- Other AI coding assistants

For tool-specific configurations (like `.cursorrules`), those remain in their respective directories.