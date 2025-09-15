# AI Agent Runbooks

This directory contains standardized runbooks (task-specific prompts) for AI agents working with the wandb/docs repository.

## What are AI Runbooks?

AI runbooks are detailed, step-by-step instructions designed to help AI agents perform complex, recurring tasks consistently and correctly. They include:

- **Context and constraints** specific to the task
- **Prerequisites** the agent should gather from users
- **Step-by-step procedures** with exact commands
- **Common issues and solutions**
- **Cleanup instructions**

## Available Runbooks

### [test-github-action-changes.md](./test-github-action-changes.md)
Tests changes to GitHub Actions workflows using a fork, particularly for workflows that depend on Cloudflare Pages deployments.

**Use cases:**
- Testing dependency upgrades (e.g., Dependabot PRs)
- Verifying workflow functionality changes
- Debugging GitHub Actions issues

## How to Use These Runbooks

### For Humans
1. Provide the runbook to your AI agent as context
2. Answer any prerequisite questions the agent asks
3. Follow along as the agent executes the steps
4. Complete any manual steps the agent cannot perform

### For AI Agents
1. Read the entire runbook before starting
2. Gather all prerequisites from the user
3. Follow the steps exactly, adapting only where explicitly noted
4. Ask for clarification if any step is unclear
5. Clean up all temporary resources after completion

## Creating New Runbooks

When creating a new runbook:

1. **Use the template structure**:
   - Agent Prerequisites
   - Task Overview
   - Context and Constraints
   - Step-by-Step Process
   - Common Issues and Solutions
   - Cleanup Instructions

2. **Make it agent-friendly**:
   - Use placeholders like `<username>` that agents should replace
   - Include explicit instructions for what agents should ask users
   - Provide fallback procedures when agents lack permissions

3. **Include all necessary context**:
   - Repository-specific constraints
   - Tool-specific limitations
   - Security considerations

4. **Test the runbook**:
   - Have an AI agent follow it exactly
   - Note any ambiguities or missing steps
   - Iterate until the process is smooth

## Contributing

To add or improve runbooks:

1. Follow the existing format and style
2. Test thoroughly with an AI agent
3. Include real examples where helpful
4. Update this README with your new runbook