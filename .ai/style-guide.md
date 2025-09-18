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
- Use sentence case for all headings (capitalize only the first word and proper nouns).
- Examples:
  - ✓ "Get started with W&B"
  - ✗ "Getting Started with W&B"
  - ✓ "Integrate with GitHub Actions"
  - ✗ "Integrating With GitHub Actions"

### Product names
- **Company name**: W&B (not Weights & Biases) in running text.
- **First mention pattern**: Use "W&B [Product]" on first mention in a page, then drop "W&B" for subsequent mentions.
- **Capitalization patterns**:
  - Products remain capitalized: W&B Weave → Weave, W&B Models → Models, W&B Launch → Launch.
  - Some features become lowercase: W&B Run → run, W&B Sweep → sweep.
  - Special case: W&B artifact (lowercase even on first mention due to API object naming).
- **General rule**: Check existing content for established patterns for  W&B-specific terminology.
- **Examples**:
  - "Configure W&B Automations to monitor your runs. Automations can send alerts..."
  - "Create a W&B Run to track your experiment. The run will capture..."
  - "W&B artifact versioning helps manage datasets. Each artifact can..."

### Voice and tone
- Direct and concise.
- Second person ("you") for instructions.
- Active voice preferred.
- Present tense for descriptions.
- Simple verbs over gerunds.

### Accessibility
- **Avoid emojis in documentation content**: Avoid using emojis in documentation. They can be ambiguous, cause translation issues, and create accessibility barriers for screen reader users.
  - **Limited exception**: Unicode symbols (✓/✗) or emoji equivalents may be used sparingly in:
    - Style guides showing correct/incorrect examples, such as this section.
    - Feature comparison matrixes or compatibility tables.
    - Only when they add clarity without relying on color or visual interpretation.
  - **Prefer Unicode symbols**: Use simple Unicode checkmarks (✓) over colorful emoji versions (✅) when possible.
  - **Never use emojis in certain contexts**: Headings, body text, instructions, or as decorative elements.
  - **Always consider**: 
    - Will this symbol be clear without color?
    - Can screen readers and accessibility tools interpret it meaningfully?
    - Does it add genuine value?
    - Could text work better?

### Punctuation
- **Simplify punctuation**: In regular text, avoid:
  - Exclamation points
  - Ellipses (...) (typographic or plain text)
  - Semicolons
  - Em dashes (—) and en dashes (–)
  - Complex inline lists requiring both commas and semicolons
  - Other unusual characters
- **Use plain text typography**: Always use straight quotes and apostrophes:
  - ✓ Use: `"straight quotes"` and `'straight apostrophes'`
  - ✗ Avoid: "curly quotes" and 'smart apostrophes'
  - **Why**: Smart quotes can creep in from copy-pasting from Word, Google Docs, or other rich text editors. They can break code examples and cause parsing issues. Let the static site generator (SSG) handle any typographic enhancements.
- **Why**: Simpler sentences improve reading comprehension and accessibility.

### Latin phrases and abbreviations
- **Use plain English**: Replace Latin abbreviations with their English equivalents:
  - ✓ "for example" instead of "e.g."
  - ✓ "that is" or "specifically" instead of "i.e."
  - ✓ "and so on" or "and others" instead of "etc."
  - ✓ "versus" or "compared to" instead of "vs."
  - ✓ "through" or "by way of" instead of "via"
- **Why**: Latin phrases reduce comprehension, especially for non-native English speakers. They're often used incorrectly and create unnecessary barriers to understanding.

### Lists
- Use sentence case for list items.
- Include periods for complete sentences.
- **Include periods for long phrases**: Even if not complete sentences, add periods to list items that are lengthy or contain explanatory text (tech writing convention).
- Omit periods only for very short fragments (2-3 words).
- **Examples**:
  - ✓ "Configure your API key." (complete sentence)
  - ✓ "Repository access for creating branches and pushing changes." (long phrase)
  - ✓ "Python 3.8 or higher with pip installed." (detailed requirement)
  - ✓ "API reference" (short fragment - no period needed)
  - ✗ "Repository access for creating branches and pushing changes" (missing period on long phrase)

### Code and technical terms
- Use backticks for:
  - File names: `config.yaml`
  - Commands: `git push`
  - Code elements: `wandb.init()`
  - Directory paths: `runbooks/`

### Code examples
See the [Formatting and indentation](#formatting-and-indentation) section for details on spacing, indentation, and editor configuration.

- **Follow public API documentation patterns**: Always conform code examples to the patterns shown in the [public API docs](https://docs.wandb.ai/ref/python/).
  - The public docs illustrate recommended practices and usage patterns.
  - Usage patterns evolve over time; bulk refactors in upstream repos update these patterns.
  - When in doubt, check the API reference for the recommended approach.
- **Use context managers**: When showing `wandb.init()`, always use the context manager pattern:
  ```python
  with wandb.init(project="my-project") as run:
      run.log({"metric": value})
  ```
    This is a specific pattern we're actively promoting to address technical debt.
- **Public APIs only**: Use only public APIs in code examples. Check the [API reference](https://docs.wandb.ai/ref/python/) for available methods. Avoid any methods starting with underscore (_) or documented as private.
- **Specify lexers**: Always include the language identifier in code blocks for proper syntax highlighting:
  - Use `python` for Python code
  - Use `bash` for shell commands  
  - Use `yaml` for YAML files
  - Use `json` for JSON data

### Formatting and indentation
- **Use spaces, not tabs**: Configure your editor to use 2 spaces per indentation level.
  - Tabs display inconsistently (often 8 spaces in browsers).
  - Consistent spacing makes complex Markdown lists easier to debug.
- **Trim trailing whitespace**: Configure your editor to automatically remove trailing spaces on save.
  - Trailing whitespace creates unnecessary diff noise in PRs.
  - Most editors have a setting for this (search for "trim trailing whitespace").
- **EditorConfig**: An `.editorconfig` file is maintained in the repository root.
  - Most modern editors automatically detect and apply these settings.
  - No manual configuration needed. Just open files and start editing.
  - See the [`.editorconfig`](/.editorconfig) file in the root of the repository for the current settings.
  - If your editor doesn't support EditorConfig natively, install the [EditorConfig plugin](https://editorconfig.org/#download).
- **VS Code/Cursor settings** (editor-specific):
  ```json
  {
    "editor.insertSpaces": true,
    "editor.tabSize": 2,
    "files.trimTrailingWhitespace": true
  }
  ```

### Important principle
**Avoid mixing style refactors with content changes**. If you're adding or editing content, match the existing style even if it's not perfect. Style refactors should be separate PRs.

## For AI agents
When editing wandb/docs content:

1. **First check**: Is this near existing content? If yes, match its style in your changes.
2. **New content**: Follow this guide and the Google Developer Style Guide.
3. **When in doubt**: Prioritize consistency over perfection.
4. **Always**: Keep style changes separate from content changes.
5. **Avoid excessive terseness**: Include appropriate punctuation and complete thoughts. Technical documentation benefits from clarity over brevity.
6. **Use symbols with extreme restraint**: Do not add emojis or Unicode symbols to documentation content without clear precedent. When in doubt, use text instead. Symbols in PRs often make changes harder to review and merge.

## Collaboration practices

When working on documentation changes:
- Create feature branches for collaboration before opening PRs.
- Start with draft PRs to prevent premature reviews or inadvertent merges.
- Write meaningful PR titles and descriptions. Avoid overwhelming descriptions.
- In PR descriptions, include relevant JIRA or GitHub issue IDs. They will be linked automatically.
- See the [collaboration guidelines](../README.md#collaboration-guidelines) for detailed practices.
