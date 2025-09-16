# AI agent runbooks

This directory contains standardized runbooks (task-specific prompts) for AI agents working with the wandb/docs repository.

## What are AI runbooks?

AI runbooks are detailed, step-by-step instructions designed to help AI agents perform complex, recurring tasks consistently and correctly. They include:

- **Context and constraints** specific to the task
- **Prerequisites** the agent should gather from users
- **Step-by-step procedures** with exact commands
- **Common issues and solutions**
- **Cleanup instructions**

## Available runbooks

### [test-github-action-changes.md](./test-github-action-changes.md)
Tests changes to GitHub Actions workflows using a fork, particularly for workflows that depend on Cloudflare Pages deployments.

**Use cases:**
- Testing dependency upgrades (for example, Dependabot PRs)
- Verifying workflow functionality changes
- Debugging GitHub Actions issues

### [TEMPLATE.md](./TEMPLATE.md)
A template for creating new runbooks. Copy this file and fill in the sections to create standardized, AI-friendly runbooks.

## How to use these runbooks

### For humans
1. Provide the runbook to your AI agent as context
2. Answer any prerequisite questions the agent asks
3. Follow along as the agent executes the steps
4. Complete any manual steps the agent cannot perform

### For AI agents
1. Read the entire runbook before starting
2. Gather all prerequisites from the user
3. Follow the steps exactly, adapting only where explicitly noted
4. Ask for clarification if any step is unclear
5. Clean up all temporary resources after completion

## Creating new runbooks

### Best practice: Agent-first authoring
The most effective runbooks are often created by having an AI agent write the first draft immediately after completing a complex task together. At the end of any challenging interactive task, consider asking your agent:

> "Based on what we just did together, please create a runbook that would help another agent perform this same task in the future. Include all the context, gotchas, and workarounds we discovered."

This captures the knowledge while it's fresh and ensures nothing important is forgotten.

### When creating a new runbook

1. **Start with the template**: Copy [TEMPLATE.md](./TEMPLATE.md) as a starting point for consistency

2. **Follow the template structure**:
   - Requirements
   - Agent Prerequisites
   - Task Overview
   - Context and Constraints
   - Step-by-Step Process
   - Common Issues and Solutions
   - Cleanup Instructions

3. **Make it agent-friendly**:
   - Use placeholders like `<username>` that agents should replace
   - Include explicit instructions for what agents should ask users
   - Provide fallback procedures when agents lack permissions

4. **Include all necessary context**:
   - Repository-specific constraints
   - Tool-specific limitations
   - Security considerations

5. **Test the runbook**:
   - Have an AI agent follow it exactly
   - Note any ambiguities or missing steps
   - Iterate until the process is smooth

6. **Get an agent review**:
   - Ask an AI agent to review the runbook for agent-friendliness
   - Example prompt: "Please review this runbook and suggest improvements to make it more useful for AI agents. Focus on clarity, completeness, and removing ambiguity."
   - Incorporate the suggested improvements

## Contributing

To add or improve runbooks:

1. Follow the existing format and style
2. Test thoroughly with an AI agent
3. Include real examples where helpful
4. Update this README with your new runbook
