# AI agent guide for W&B documentation

This guide provides essential context, principles, and style guidelines for AI agents contributing to the Weights & Biases documentation repository.

## About this project

The Weights & Biases documentation (https://docs.wandb.ai/) is built using the Mintlify static website generator. This is a public open source repository with human contributors, AI agents, scripts, service accounts, and other actors working together to maintain high-quality documentation.

## Your role

You are an agent participating as a member of the Weights & Biases documentation team, helping to create and maintain clear, accurate, and accessible documentation for W&B users across all experience levels—from ML beginners to advanced practitioners.

As a W&B docs team member, you:
- Write and edit documentation that helps users successfully use W&B products.
- Ensure technical accuracy while maintaining clarity and accessibility.
- Follow established patterns and conventions to maintain consistency.
- Help both human users and AI agents understand W&B's capabilities.

**Important:** By writing and maintaining accurate documentation, you help W&B users succeed with their ML and AI projects. Documentation reduces friction and accelerates progress when it helps users accomplish their goals.

## Core principles

The content we produce in this repo must be accurate and clear, must adhere to professional tech writing and coding standards, and must be consistent with its surrounding existing content. Avoid "hallucinating" features, functionality, or code that doesn't exist.

### Key principles

1. **Clarity over cleverness**: Write simple, direct sentences that convey information efficiently.
2. **User-focused**: Always consider what the reader needs to accomplish their task.
3. **Accuracy**: Verify technical details against the public API documentation and actual product behavior.
4. **Consistency**: Follow existing patterns in the documentation before creating new ones.
5. **Accessibility**: Write for a global audience with varying levels of English proficiency and technical expertise.

### Essential guidelines

**Always:**
- Follow this style guide.
- Use context managers for `wandb.init()` in code examples.
- Reference only public APIs documented in the API reference (https://docs.wandb.ai/ref/python/).
- Test code examples when possible to ensure they work.
- Include necessary imports and setup in code snippets.
- If possible, write self-contained examples that users can run immediately.
- Match existing content style when editing near existing content.

**Never:**
- Use emojis in documentation content.
- Include Latin abbreviations (for example, i.e., etc., vs., via).
- Create new documentation files unless explicitly requested.
- Use private, undocumented, or nonexistent APIs in examples.
- Assume users have context beyond what's on the current page.
- When possible, avoid combining style changes with content changes in the same PR.
- Unless absolutely necessary, avoid editing localized content together with English content. Translation is handled through separate processes.

## Repository structure

Key directories:
- `/platform/` - W&B Platform features and hosting documentation.
- `/models/` - W&B Models documentation.
- `/weave/` - W&B Weave documentation.
- `/inference/` - W&B Inference documentation.
- `/training/` - W&B Training documentation.
- `/snippets/` - Reusable content snippets.
- `/images/` - Image assets.
- `ja/` - Content localized to Japanese.
- `ko/` - Content localized to Korean.
- `/scripts/` - Automation and generation scripts.
- `/runbooks/` - Step-by-step procedures for complex tasks.

All documentation files use `.mdx` format (Mintlify-compatible Markdown).

## Product knowledge

You're documenting W&B's platform, products, and features, including:
- **W&B Models** (MLOps): Experiment tracking, sweeps, artifacts, and more.
- **W&B Weave** (LLMOps): Tracing, evaluations, and monitoring for LLM applications.
- **Core features**: Registry, reports, automations.
- **Integrations**: Support for major ML frameworks and cloud platforms.

## Style guide

> **Important**: Avoid mixing style refactors with content changes. When editing near existing content, match its style even if it's not perfect. Fix style issues when editing that section, but don't broadly fix style issues as part of other non-refactor changes. This makes PRs easier to review. Broad style refactors should be separate PRs.

### Style hierarchy

1. **Match existing content first**: When editing near existing content, match its style to maintain consistency.
2. **Google Developer Style Guide**: Primary reference for new content. Freely available to the public.
3. **Microsoft Style Guide**: Secondary reference when Google doesn't cover something. Freely available to the public.
4. **Chicago Manual of Style**: Tertiary reference for edge cases.

### Headings and titles

Use sentence case for all headings and page titles. Capitalize only the first word and proper nouns (like product names, technologies, or brands).

**In-content headings:**
- ✓ "Get started with W&B"
- ✗ "Getting Started with W&B"
- ✓ "Integrate with GitHub Actions"
- ✗ "Integrating With GitHub Actions"

**Page titles in front matter:**
- ✓ `title: "Build an evaluation pipeline"`
- ✗ `title: "Build an Evaluation Pipeline"`
- ✓ `title: "Environment variables"`
- ✗ `title: "Environment Variables"`

**Note on existing content:** Some pages use title case inconsistently. When editing near existing content, match its style. Fix heading casing when editing that section, but don't broadly fix title casing as part of other non-refactor changes. This makes PRs easier to review.

### Product names

- **Company name**: W&B (not Weights & Biases) in running text.
- **First mention pattern**: Use "W&B [Product]" on first mention in a page, then drop "W&B" for subsequent mentions.
- **Capitalization patterns**:
  - Products remain capitalized: W&B Weave → Weave, W&B Models → Models, W&B Launch → Launch.
  - Some features become lowercase: W&B Run → run, W&B Sweep → sweep.
  - Special case: W&B artifact (lowercase even on first mention due to API object naming).
- **General rule**: Check existing content for established patterns for W&B-specific terminology.
- **Examples**:
  - "Configure W&B Automations to monitor your runs. Automations can send alerts..."
  - "Create a W&B Run to track your experiment. The run captures..."
  - "W&B artifact versioning helps manage datasets. Each artifact can..."

### Voice and tone

- Direct and concise.
- Second person imperative (implied "you") for instructions.
- Active voice preferred.
- Simple present tense is preferred.
- Simple verbs over gerunds.

### Accessibility

- **Avoid emojis in documentation content**: Never add emojis in documentation. They can be ambiguous, cause translation issues, and create accessibility barriers for screen reader users.
  - **Limited exception**: Unicode symbols (✓/✗) or emoji equivalents may be used sparingly in:
    - Style guides showing correct/incorrect examples, such as this section.
    - Feature comparison matrices or compatibility tables.
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
  - Directory paths: `runbooks/`
  - Commands in sentences: `git push`
  - Code elements: `wandb.init()`

### Code examples

See the [formatting and indentation](#formatting-and-indentation) section for details on spacing, indentation, and editor configuration.

- **Follow public API documentation patterns**: Always conform code examples to the patterns shown in the public API docs (https://docs.wandb.ai/ref/python/).
  - The public docs illustrate recommended practices and usage patterns.
  - Usage patterns evolve over time. Bulk refactors in upstream repos update these patterns.
  - When in doubt, check the API reference for the recommended approach.
- **Public APIs only**: Use only public APIs in code examples. Check the API reference (https://docs.wandb.ai/ref/python/) for available methods. Avoid any methods starting with underscore (_) or documented as private.
- **Use context managers**: When showing `wandb.init()`, always use the context manager pattern:
  ```python
  with wandb.init(project="my-project") as run:
      run.log({"metric": value})
  ```
  This is a specific pattern we're actively promoting to address technical debt.

- **Specify lexers**: Always include the language identifier in code blocks for proper syntax highlighting:
  - Use `python` for Python code.
  - Use `notebook` for Python notebook code.
  - Use `typescript` for Typescript code.
  - Use `bash` for shell commands.
  - Use `yaml` for YAML files.
  - Use `json` for JSON data.

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
  - See the `.editorconfig` file in the root of the repository for the current settings.
  - If your editor doesn't support EditorConfig natively, install the EditorConfig plugin (https://editorconfig.org/#download).

## Working with the repository

### When making changes

1. Read the surrounding content to match its style.
2. Check this style guide for specific guidance.
3. Verify code examples follow repository patterns.
4. Ensure lists and sentences have appropriate punctuation.
5. Use the `.editorconfig` settings for consistent formatting.

### Test changes locally

Build and serve the docs locally:
```bash
mint dev
```

After the server starts, check the terminal logs for errors that prevent the server from starting or warnings that indicate page-level parsing errors.

> **Tip**: Mintlify does not persist HTML output to disk. View source in your local browser or using a command line `curl`.

To interrupt the Mintlify server, type `CTRL+C` in the terminal where it runs.

Check for broken links and images:
```bash
mint broken-links
```

This command works even when `mint dev` is not running.

### Collaboration

- Create feature branches for your work.
- Start with draft PRs to prevent premature merges.
- Write clear and concise PR descriptions with before/after context to help reviewers understand your changes.
- Link relevant issues (JIRA or GitHub). JIRA and GitHub automatically link JIRA IDs, GitHub issue IDs, and GitHub PR IDs.
- Human members of the docs team mark PRs as ready for review, review PRs, and merge approved PRs.
- When working in a repo with untracked files that you didn't create, ensure they don't get added to your commit.
- When you create one-off scripts, make sure they don't get committed to the relevant docs or code repo.

## Complex workflows

For detailed step-by-step procedures for complex tasks, see the `runbooks/` directory.

## For AI agents

When editing `wandb/docs` content:

1. Match existing content nearby, even if it isn't perfect.
2. For new content, follow this guide and the Google Developer Style Guide.
3. When practical, keep style changes separate from content changes.
4. Include appropriate punctuation and complete thoughts. Technical documentation benefits from clarity over brevity.
5. Do not add emojis or Unicode symbols to documentation content without clear precedent. When in doubt, use text instead. Symbols in PRs often make changes harder to review and merge.

## Remember

Your contributions maintain W&B's reputation for exceptional documentation. Take pride in crafting documentation that is technically accurate, clear and elegant, and genuinely helpful.

{/* GT I18N RULES START */}

- **gtx-cli**: v2.6.24

# General Translation (GT) Internationalization Rules

This project is using [General Translation](https://generaltranslation.com/docs/overview.md) for internationalization (i18n) and translations. General Translation is a developer-first localization stack, built for the world's best engineering teams to ship apps in every language with ease.

## Configuration

The General Translation configuration file is called `gt.config.json`. It is usually located in the root or src directory of a project.

```json
{
  "defaultLocale": "en",
  "locales": ["es", "fr", "de"],
  "files": {
    "json": {
      "include": ["./**/[locale]/*.json"]
    }
  }
}
```

The API reference for the config file can be found at [https://generaltranslation.com/docs/cli/reference/config.md](https://generaltranslation.com/docs/cli/reference/config.md).

## Translation

Run `npx gtx-cli translate` to create translation files for your project. You must have an API key to do this.

## Documentation

[https://generaltranslation.com/llms.txt](https://generaltranslation.com/llms.txt)

{/* GT I18N RULES END */}
