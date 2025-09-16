# Style guide for wandb/docs content

## Overview

This document provides style guidance for AI agents creating or editing content in the wandb/docs repository.

## Style hierarchy

1. **Match existing content first**: When editing near existing content, match its style to maintain consistency
2. **Google Developer Style Guide**: Primary reference for new content
3. **Microsoft Style Guide**: Secondary reference when Google doesn't cover something
4. **Chicago Manual of Style**: Tertiary reference for edge cases

## Key style rules

### Headings
- Use sentence case for all headings (capitalize only the first word and proper nouns)
- Examples:
  - ✅ "Getting started with W&B"
  - ❌ "Getting Started with W&B"
  - ✅ "Integrating with GitHub Actions"
  - ❌ "Integrating With GitHub Actions"

### Product names
- **Company name**: W&B (not Weights & Biases in running text)
- **First mention pattern**: Use "W&B [Product]" on first mention in a page, then drop "W&B" for subsequent mentions
- **Capitalization patterns**:
  - Products that remain capitalized: W&B Weave → Weave, W&B Models → Models, W&B Launch → Launch
  - Products that become lowercase: W&B Run → run, W&B Sweep → sweep
  - Special case: W&B artifact (lowercase even on first mention due to API object naming)
- **General rule**: Check existing content for the established pattern - consistency matters more than guessing
- **Examples**:
  - "Configure W&B Automations to monitor your runs. Automations can send alerts..."
  - "Create a W&B Run to track your experiment. The run will capture..."
  - "W&B artifact versioning helps manage datasets. Each artifact can..."

### Voice and tone
- Direct and concise
- Second person ("you") for instructions
- Active voice preferred
- Present tense for descriptions

### Code and technical terms
- Use backticks for:
  - File names: `config.yaml`
  - Commands: `git push`
  - Code elements: `wandb.init()`
  - Directory paths: `runbooks/`

### Lists
- Use sentence case for list items
- Include periods for complete sentences
- Omit periods for fragments

### Important principle
**Avoid mixing style refactors with content changes**. If you're adding or editing content, match the existing style even if it's not perfect. Style refactors should be separate PRs.

## For AI agents

When editing wandb/docs content:

1. **First check**: Is this near existing content? If yes, match its style exactly
2. **New content**: Follow this guide and the Google Developer Style Guide
3. **When in doubt**: Prioritize consistency over perfection
4. **Always**: Keep style changes separate from content changes