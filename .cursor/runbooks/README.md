# Cursor runbooks for W&B documentation

This directory contains task-specific runbooks formatted for optimal use with Cursor and other AI agents. These runbooks provide detailed, step-by-step instructions for complex or recurring tasks in the wandb/docs repository.

## What are runbooks?

Runbooks are structured guides that help AI agents (and humans) perform complex tasks consistently and correctly. They include:

- Clear prerequisites and requirements
- Step-by-step procedures with exact commands
- Context about system limitations and constraints
- Common issues and their solutions
- Cleanup instructions to leave the system in a good state

## Available runbooks

### [test-github-action-changes.md](./test-github-action-changes.md)
Tests changes to GitHub Actions workflows using a fork, particularly for workflows that depend on Cloudflare Pages deployments.

**Use cases:**
- Testing Dependabot PRs for action upgrades
- Verifying workflow functionality changes  
- Debugging GitHub Actions issues
- Testing workflow changes before merging to main

**Requirements**: W&B employee access

### [TEMPLATE.md](./TEMPLATE.md)
A template for creating new runbooks. Copy this file and fill in the sections to create standardized, AI-friendly runbooks.

## How to use these runbooks

### For AI agents
1. Read the entire runbook before starting any task.
2. Gather all prerequisites from the user upfront.
3. Follow the steps exactly, adapting only where explicitly noted.
4. Pay attention to "Agent note" sections for special instructions.
5. Ask for clarification if any step is unclear.
6. Always complete the cleanup steps.

### For humans
1. Provide the runbook to your AI agent as context.
2. Answer any prerequisite questions the agent asks.
3. Follow along as the agent executes the steps.
4. Complete any manual steps the agent cannot perform.
5. Verify the results match expected outcomes.

## Creating new runbooks

### When to create a runbook
Create a runbook when:
- A task requires multiple complex steps
- The task will be repeated in the future
- There are specific gotchas or edge cases to remember
- The task requires specific permissions or access
- You've just completed a complex task and want to capture the knowledge

### Best practice: Agent-first authoring
The most effective runbooks are created immediately after completing a complex task. Ask your AI agent:

> "Based on what we just did together, please create a runbook that would help another agent perform this same task in the future. Include all the context, gotchas, and workarounds we discovered."

### Structure guidelines

All runbooks follow this structure:
1. **Overview** - What the runbook accomplishes
2. **Requirements** - Access and permissions needed
3. **Prerequisites** - Information to gather from users
4. **Context and Constraints** - Important background and limitations
5. **Step-by-Step Process** - Detailed procedures
6. **Verification** - How to confirm success
7. **Troubleshooting** - Common issues and solutions
8. **Cleanup** - How to reset/clean up
9. **Summary Checklist** - Quick reference
10. **Additional Notes** - Extra context or tips

### Formatting conventions

- Use XML-like tags for major sections: `<section_name>...</section_name>`
- Include code blocks with appropriate language tags
- Use **Agent note:** for AI-specific instructions
- Provide exact commands and file paths
- Include placeholder syntax like `<username>` for values to replace

## Maintenance

### Keeping runbooks current
- Test runbooks periodically to ensure they still work
- Update when tools, APIs, or processes change
- Add new troubleshooting items as issues are discovered
- Get feedback from agents and humans using the runbooks

### Review process
1. Have an AI agent review new runbooks for clarity.
2. Test the runbook with a different agent.
3. Iterate based on confusion points or failures.
4. Update both .cursor and .ai versions if applicable.

## Security considerations

- **Access Requirements**: Clearly mark which runbooks require special access (e.g., W&B employee)
- **Sensitive Information**: Never include API keys, passwords, or internal URLs
- **Public Repository**: Remember these files are in a public repo
- **Temporary Changes**: Always include cleanup steps for any temporary modifications

## Tips for effective runbooks

1. **Be Explicit**: Don't assume knowledge; spell everything out
2. **Include Context**: Explain why steps are needed, not just what to do
3. **Anticipate Failures**: Include troubleshooting for common issues
4. **Test Thoroughly**: Run through the entire runbook before publishing
5. **Version Awareness**: Note any version-specific requirements
6. **Platform Differences**: Mention OS-specific variations if relevant

## Getting help

If you need help with runbooks:
1. Check the template for structure guidance.
2. Review existing runbooks for examples.
3. Ask an AI agent to review your draft.
4. Test with a fresh agent to find unclear parts.
