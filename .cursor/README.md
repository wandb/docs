# Cursor configuration for W&B documentation

This directory contains Cursor-specific prompts and configurations for AI agents working with the W&B documentation repository. These prompts are optimized for use with Cursor while maintaining human readability.

> **Note**: The format used here (Markdown with XML-like tags) is based on community best practices. Cursor has not published an official specification for `.cursor/` directory contents at the time of writing.

## Directory structure

### Core configuration files

#### `rules.md`
Core project rules and conventions that apply to all work in this repository. Includes:
- Mandatory rules (what to always/never do)
- Repository structure overview
- Security and collaboration guidelines
- General project context

#### `style.md`
Comprehensive style guide for W&B documentation covering:
- Detailed writing style rules
- Product naming conventions
- Accessibility guidelines
- Code example patterns
- Punctuation and formatting standards

#### `docs.md`
Documentation-specific guidelines including:
- Role definition for documentation contributors
- Documentation philosophy and values
- Content organization patterns
- Writing patterns and templates
- Quality checklists

### `runbooks/`
Task-specific prompts for complex operations:
- Step-by-step instructions for recurring tasks
- Formatted for optimal AI agent consumption
- Includes context, prerequisites, and troubleshooting

## How to use these prompts

### For Cursor users
1. Cursor detects that `.cursor/` files exist but does NOT automatically load their contents.
2. You still need to prompt the agent to read these files.
3. Use the example prompts below to direct your agent to these guidelines.

### For other AI tools
1. Provide relevant files as context when starting a session.
2. Reference specific sections when needed.
3. The structured format with XML-like tags aids parsing.

### Example prompts to direct your agent

Since agents don't automatically load file contents (even in Cursor), use these prompts to ensure they follow the guidelines:

**Before starting work:**
```
Please read .cursor/rules.md and .cursor/style.md to understand the project conventions, then help me with my task.
```

**When reviewing changes:**
```
Review your proposed changes against .cursor/style.md and make any necessary adjustments to match the style guidelines.
```

**For specific tasks:**
```
Check .cursor/runbooks/ for any relevant runbooks before proceeding with [task description].
```

**To ensure consistency:**
```
Before editing [filename], use grep to check how similar concepts are documented in this directory, then match that style.
```

## Key differences from `.ai/` directory

| Aspect | `.ai/` Directory | `.cursor/` Directory |
|--------|-----------------|-------------------|
| Format | Natural language prose | Structured with XML-like tags |
| Target | Human-readable first | AI-optimized, human-readable |
| Organization | Topic-based files | Role-based separation |
| Usage | Manual context provision | Automatic loading in Cursor |

## Maintenance guidelines

### When to update these files
- When documentation standards change
- When new patterns emerge
- When common issues need addressing
- When new runbooks are needed

### How to update
1. Maintain the structured format with clear sections.
2. Keep XML-like tags for major sections.
3. Ensure human readability alongside AI optimization.
4. Test changes with actual documentation tasks.

### Creating new runbooks
1. Copy the template from `runbooks/TEMPLATE.md`
2. Fill in all sections with specific details.
3. Include all gotchas and edge cases discovered.
4. Test with an AI agent before finalizing.

## Security notes

- Some runbooks require W&B employee access (clearly marked)
- Never include sensitive information (API keys, passwords, internal URLs)
- These files are in a public repository

## Related resources

- `.ai/` directory - Natural language versions of these prompts
- Main repository README - Overall repository documentation
- [W&B Documentation Site](https://docs.wandb.ai) - The live documentation

## Contributing

When updating these configurations:
1. Ensure consistency with existing patterns.
2. Test changes with real documentation tasks.
3. Keep both human and AI usability in mind.
4. Update both `.cursor/` and `.ai/` directories if applicable.
