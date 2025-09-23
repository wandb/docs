# W&B documentation project rules

This file defines the core rules and conventions for the W&B documentation repository.

> **Tip**: To help save context tokens, agents do not automatically load prompts like these project rules. To ask your agent to load and use these rules, prompt it with: "Read `.cursor/rules.md` to understand the mandatory rules for this project."

## Project overview

This is the official repository repository for W&B (Weights & Biases) documentation, including:
- **[W&B Models](https://docs.wandb.ai/guides/models/)** (MLOps): Experiment tracking, sweeps, artifacts, and more.
- **[W&B Weave](https://weave-docs.wandb.ai/)** (LLMOps): Tracing, evaluations, and monitoring for LLM applications.
- **[W&B Inference](https://docs.wandb.ai/guides/inference/)** (Foundation Model APIs): Access to open-source foundation models through an OpenAI-compatible API.
- **Core features** of W&B: Registry, reports, automations.
- **Integrations**: Support for major ML frameworks and cloud platforms.

## Repository structure

- `/content/`: Main documentation content organized by language (`en`, `ja`, `ko`). In general, only English content. Separate processes exist for updating translations.
- `/assets/`: Images and other static resources.
- `/.ai/`: Natural language resources for AI agents (human-readable format).
- `/.cursor/`: Prompts and configurations in Cursor format (this directory).

## Core documentation principles

<documentation_principles>

1. **Clarity over cleverness**: Write simple, direct sentences that convey information efficiently.
2. **User-focused**: Always consider what the reader needs to accomplish their task.
3. **Accuracy**: Verify technical details against the public API documentation and actual product behavior.
4. **Consistency**: Follow existing patterns in the documentation before creating new ones.
5. **Accessibility**: Write for a global audience with varying levels of English proficiency and technical expertise.

</documentation_principles>

## Mandatory rules

<mandatory_rules>

1. **Always check existing content style BEFORE editing** - Read the file and nearby files first
2. **Never use emojis in documentation content** - They cause accessibility and translation issues
3. **Never include Latin abbreviations** (e.g., i.e., etc., vs., via) - Use plain English equivalents
4. **Never create new documentation files unless explicitly requested**
5. **Never mix style changes with content changes in the same PR**
6. **Never use private or undocumented APIs in examples**
7. **Always use context managers for `wandb.init()` in code examples**
8. **Always reference only public APIs documented in the API reference**
9. **Always match existing content style when editing near existing content**

### The prime directive: Match existing style

When editing documentation, your first priority is consistency with the existing corpus. Before making any changes:
1. **Read the entire file** you're editing.
2. **Check nearby files** in the same directory.
3. **Identify established patterns** for formatting, terminology, and structure.
4. **Match those patterns** even if they differ from this style guide.

This approach ensures:
- Reviewers see consistent changes.
- The documentation maintains a unified voice.
- Style evolution happens deliberately, not accidentally.

</mandatory_rules>

## Style guidelines

<style_rules>


### Headings
- Use sentence case for all headings (capitalize only the first word and proper nouns).
- Examples:
  - ✓ "Get started with W&B"
  - ✗ "Getting Started with W&B"

### Product names
- **Company name**: W&B (not Weights & Biases) in running text.
- **First mention pattern**: Use "W&B [Product]" on first mention, then drop "W&B" for subsequent mentions.
- **Capitalization patterns**:
  - Products remain capitalized: W&B Weave → Weave, W&B Models → Models
  - Some features become lowercase: W&B Run → run, W&B Sweep → sweep
  - Special case: W&B artifact (lowercase even on first mention)

### Voice and tone
- Direct and concise.
- Second person ("you") for instructions.
- Active voice preferred.
- Present tense for descriptions.
- Simple verbs over gerunds.

### Punctuation
- Use straight quotes and apostrophes (not curly/smart quotes).
- Avoid exclamation points, ellipses, semicolons, em/en dashes.
- Include periods for complete sentences and long phrases in lists.

</style_rules>

## Code examples

<code_example_rules>

1. **Follow public API documentation patterns** - Check https://docs.wandb.ai/ref/python/ for recommended usage
2. **Always use context managers** for `wandb.init()`:
   ```python
   with wandb.init(project="my-project") as run:
       run.log({"metric": value})
   ```
3. **Specify language identifiers** in code blocks (python, bash, yaml, json)
4. **Include necessary imports and setup** - Examples should be self-contained and runnable
5. **Test code examples when possible** to ensure they work

</code_example_rules>

## Formatting standards

<formatting_rules>

- **Use spaces, not tabs**: 2 spaces per indentation level.
- **Trim trailing whitespace**: Remove trailing spaces on all lines.
- **EditorConfig**: The repository includes an `.editorconfig` file that enforces these settings.
- **Markdown details**: See [`style.md` Format specification](style.md#format-specification) for complete formatting conventions including emphasis, links, and code blocks.

</formatting_rules>

## Collaboration practices

<collaboration_rules>

1. **Create feature branches** for all work
2. **Start with draft PRs** to prevent premature reviews or inadvertent merges
3. **Write clear PR descriptions** including:
   - What changed and why
   - Links to relevant JIRA or GitHub issues
   - Before/after comparison when applicable
4. **Wait for all tests to pass** before marking PR as ready for review
5. **Let humans merge PRs** - AI agents should not merge PRs

</collaboration_rules>

## Security guidelines

<security_rules>

- Never include sensitive information (API keys, passwords, internal URLs).
- Be cautious about exposing infrastructure details.
- Review all contributions for potential security risks.
- Some runbooks require W&B employee access - clearly mark these.

</security_rules>

## File handling

<file_handling_rules>

- **Prefer editing existing files** over creating new ones
- **Match surrounding style** when editing existing content
- **Separate style refactors** from content changes
- **Avoid bulk style changes** without explicit approval

</file_handling_rules>

## Important context

<context>

- This is a public repository, but some runbooks require W&B employee access.
- The repository uses Hugo as its static site generator.
- Documentation is available in multiple languages (English, Japanese, Korean).
- PR previews are generated using Cloudflare Pages.
- The main branch is protected and requires review before merge.

</context>
