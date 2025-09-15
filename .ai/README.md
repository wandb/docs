# AI Resources for wandb/docs

This directory contains resources, prompts, and tools designed to help AI agents work effectively with the wandb/docs repository.

## Directory Structure

### `runbooks/`
Standardized, task-specific instructions for AI agents performing complex operations in this repository. These runbooks ensure consistent, reliable execution of recurring tasks.

**Example tasks:**
- Testing GitHub Actions changes
- Performing large-scale refactoring
- Managing release processes

See [runbooks/README.md](./runbooks/README.md) for detailed information.

### Future Directories (Planned)

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

## Usage Guidelines

### For Repository Maintainers
1. Keep runbooks up-to-date as processes change
2. Test runbooks regularly with AI agents
3. Document any repository-specific quirks or constraints
4. Version control all AI resources alongside code

### For AI Agent Users
1. Always check for relevant runbooks before starting complex tasks
2. Provide runbooks as context to your AI agent
3. Report issues or ambiguities in runbooks
4. Contribute improvements based on your experience

## Best Practices

1. **Specificity**: Be explicit about every step, assumption, and requirement
2. **Adaptability**: Include variations for common scenarios
3. **Safety**: Always include cleanup steps and error handling
4. **Testing**: Verify runbooks work with multiple AI providers
5. **Maintenance**: Update runbooks when workflows or tools change

## Contributing

When adding AI resources:

1. Follow existing naming conventions and structure
2. Include comprehensive documentation
3. Test with at least one AI agent
4. Update relevant README files
5. Consider security implications of any shared context

## Security Notes

- Never include sensitive information (API keys, passwords, etc.)
- Be cautious about exposing internal URLs or infrastructure details
- Review all contributions for potential security risks