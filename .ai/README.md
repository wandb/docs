# AI resources for wandb/docs

This directory contains resources, prompts, and tools designed to help AI agents work effectively with the wandb/docs repository.

## Getting started

Start by reading the [system prompt](./system-prompt.md) to understand your role as a member of the W&B docs team.

## Directory structure

### `system-prompt.md`
The foundational context for AI agents joining the W&B docs team. Defines your role, responsibilities, and key principles for creating documentation.

### `style-guide.md`
Comprehensive style guide for W&B documentation. Covers formatting, punctuation, code examples, and W&B-specific conventions.

### `runbooks/`
Standardized, task-specific instructions for AI agents performing complex operations in this repository. These runbooks ensure consistent, reliable execution of recurring tasks.

**Example tasks:**
- Testing GitHub Actions changes
- Performing large-scale refactoring
- Managing release processes

See [runbooks/README.md](./runbooks/README.md) for detailed information.

### Future directories (planned)

#### `prompts/`
System prompts and context for different types of AI interactions:
- Documentation writing guidelines
- Code review standards
- Style and tone specifications

#### `tools/`
Scripts and utilities that AI agents can use:
- Validation scripts
- Automation helpers
- Analysis tools

#### `context/`
Repository-specific context that helps AI agents understand:
- Architecture decisions
- Historical context
- Domain-specific knowledge

## Usage guidelines

### For repository maintainers
1. Keep runbooks up-to-date as processes change
2. Test runbooks regularly with AI agents
3. Document any repository-specific quirks or constraints
4. Version control all AI resources alongside code

### For AI agent users
1. Always check for relevant runbooks before starting complex tasks
2. Provide runbooks as context to your AI agent
3. Report issues or ambiguities in runbooks
4. Contribute improvements based on your experience

## Collaboration guidelines

When working with human and AI teammates on documentation changes:

### Before creating a PR
- **Feature branches**: Push feature branches to collaborate with others before opening a PR.
- **Branch naming**: Use descriptive names that indicate the purpose (for example, `fix/broken-links-automation-guide`).

### Creating pull requests
1. **Start with draft PRs**: Create a draft pull request initially. This won't request reviews and can't be merged accidentally.
2. **Wait for tests**: Ensure all PR tests pass before marking as ready for review.
3. **Coordinate with humans**: The human coordinating the work should verify the changes meet requirements before marking the PR ready for review. A human should ultimately merge a PR, not an agent.

### PR titles and descriptions
- **Be meaningful but concise**: Write clear titles that describe what changed.
- **Link relevant context**: Include:
  - JIRA IDs (for example, `DOCS-1234`)
  - GitHub issue IDs (for example, `#456`)
  - Related PRs for additional context
- **Add before/after comparison**: When applicable, include:
  - Links to the current live documentation
  - Links to the PR's HTML preview showing the changes
  - Brief description of what changed and why

### Example PR description
```
## Description

Updates AI agent style guide to improve accessibility and code quality.

The style guide now provides clearer guidance for AI agents creating documentation:
- Added accessibility guidelines (no emojis)
- Added code example best practices  
- Created runbook template

### Before/After
- Live docs: https://docs.wandb.ai/guides/ai-resources
- PR preview: https://preview-12345.pages.dev/guides/ai-resources

## Related issues

- Fixes DOCS-1234
- Related to #456
```

## Best practices

1. **Capture knowledge immediately**: After completing any complex task with an AI agent, ask them to draft a runbook while the details are fresh
2. **Specificity**: Be explicit about every step, assumption, and requirement
3. **Adaptability**: Include variations for common scenarios
4. **Safety**: Always include cleanup steps and error handling
5. **Testing**: Verify runbooks work with multiple AI providers
6. **Maintenance**: Update runbooks when workflows or tools change
7. **Iterative improvement**: Have agents review and improve existing runbooks based on their experience

## Contributing

When adding AI resources:

1. Follow existing naming conventions and structure
2. Include comprehensive documentation
3. Test with at least one AI agent
4. Update relevant README files
5. Consider security implications of any shared context

## Security notes

- Never include sensitive information (API keys, passwords, etc.)
- Be cautious about exposing internal URLs or infrastructure details
- Review all contributions for potential security risks
