# W&B documentation workflow guidelines

This guide covers development practices, pull request processes, and collaboration guidelines for W&B documentation.

## Pull request guidelines

### PR creation
- **Always create PRs as drafts initially**: This allows for review and refinement before formal review
- Use descriptive PR titles that summarize the change
- Include relevant JIRA ticket numbers in PR descriptions
- Reference related issues or PRs when applicable

### Commit messages
- Write descriptive messages that explain what changed
- Keep messages concise - detailed context belongs in the PR description
- Focus on the "what" rather than the "why" (PRs provide the why)
- Note: The repository uses squash and merge, so individual commit messages are less critical
- The final commit message is typically adjusted based on the PR description during merge

### PR description
- Explain what changes were made and why
- List any testing performed
- Note any areas requiring special review attention
- Include before/after screenshots for UI changes

## Development process

### Before starting work
1. Pull latest changes from main branch
2. Create a feature branch with descriptive name
3. Check for existing related work or issues

### During development
- Test changes locally with `hugo server`
- Verify all links work correctly
- Check that code examples run successfully
- Review changes in browser before committing

### Testing requirements
- Build and preview documentation locally
- Test all code examples
- Verify internal and external links
- Check rendering on different screen sizes

## File management

### Creating new files
- Prefer editing existing files over creating new ones
- Only create files when absolutely necessary
- Never proactively create documentation files (*.md) or README files unless explicitly requested

### File organization
- Follow existing directory structure
- Use consistent naming conventions
- Keep related content together

## Collaboration practices

### Working with JIRA
- Reference JIRA tickets in commits and PRs
- Update ticket status as work progresses
- Follow the workflow: Backlog → Todo → In progress → In review

### Code review process
- Address all review comments
- Ask for clarification when needed
- Re-request review after making changes
- Don't merge without approval

## Branch management

### Naming conventions
- Use descriptive branch names
- Include ticket numbers when applicable
- Follow existing patterns in the repository

### Branch hygiene
- Keep branches up to date with main
- Delete branches after merging
- Don't reuse old branches

## Quality checks

### Before creating PR
- Run local build to check for errors
- Review all changes in the diff
- Ensure no unintended files are included
- Verify commit messages follow conventions

### Common issues to avoid
- Broken links
- Missing images
- Incorrect code examples
- Formatting inconsistencies

## Special considerations

### Large changes
- Consider breaking into smaller PRs
- Communicate with team about impact
- Plan for gradual rollout if needed

### Documentation updates
- Coordinate with SDK releases when needed
- Consider impact on existing users
- Update related documentation together

## Automation and tools

### Local development
- Use `hugo server` for local preview
- Install all prerequisites before starting
- Keep dependencies up to date

### CI/CD checks
- Ensure all CI checks pass
- Fix any build errors promptly
- Don't ignore test failures
