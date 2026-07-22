# AI agent guide for W&B documentation

This guide provides essential context, principles, and repository conventions for AI agents contributing to the Weights & Biases documentation repository. **Style rules live in the `style-guide` skill** (see [Style](#style)), not in this file.

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

The content we produce in this repo must be accurate and clear, and must adhere to professional tech writing and coding standards. Avoid "hallucinating" features, functionality, or code that doesn't exist.

### Key principles

1. **Clarity over cleverness**: Write simple, direct sentences that convey information efficiently.
2. **User-focused**: Always consider what the reader needs to accomplish their task.
3. **Accuracy**: Verify technical details against the public API documentation and actual product behavior.
4. **Consistency**: Follow the documentation's existing structure and information architecture rather than inventing new patterns. Style comes from the `style-guide` skill.
5. **Accessibility**: Write for a global audience with varying levels of English proficiency and technical expertise.

### Essential guidelines

**Always:**
- Use the `style-guide` skill for all style decisions (see [Style](#style)).
- Test code examples when possible to ensure they work.
- Include necessary imports and setup in code snippets.
- If possible, write self-contained examples that users can run immediately.

**Never:**
- Create new documentation files unless explicitly requested.
- Use private, undocumented, or nonexistent APIs in examples.
- When possible, avoid combining style changes with content changes in the same PR.
- Unless absolutely necessary, avoid editing localized content together with English content. Translation is handled through separate processes.

## Repository structure

Key directories:
- `/platform/` - W&B Platform features and hosting documentation.
- `/models/` - W&B Models documentation.
- `/weave/` - W&B Weave documentation.
- `/inference/` - Serverless Inference documentation.
- `/serverless-training/` - Serverless Training documentation (Serverless RL and Serverless SFT subcategories; formerly Serverless RL / W&B Training).
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

## Style

> **Required: the `style-guide` skill must be installed.**
> This file no longer contains inline style rules — they live in the `style-guide` skill's pass files (`1-context-pass.md` through `7-polish-pass.md`). Before editing any documentation content, confirm you can read those pass files. If you cannot locate them, STOP and tell the user:
>
> "The style-guide skill is required but not installed. Install it from https://github.com/coreweave/docs-skills (see the README Setup section), for example by cloning the repo and symlinking `skills/style-guide` into `~/.claude/skills/`. Skill source: https://github.com/coreweave/docs-skills/tree/main/skills/style-guide"
>
> Do not apply style rules from memory and do not proceed without the skill — the inline rules have been removed from this file.

To check or apply style, read the relevant pass file(s) and use them as guidance — do not run the skill itself. Apply rules to the section you're editing. You have the whole file available, so consult it when a rule needs document-wide context (for example, first-use abbreviation expansion or terminology consistency). Don't broadly re-style or restructure beyond your change. Broad style refactors belong in separate PRs.

### W&B-specific conventions not in the skill

- **Python SDK functions**: Module-level functions in the Python SDK are listed in the [Global Functions overview](/models/ref/python/functions).
- **`.editorconfig`**: An `.editorconfig` file in the repository root enforces indentation and whitespace automatically. Most editors apply it with no configuration. If yours doesn't support it natively, install the EditorConfig plugin (https://editorconfig.org/#download).
- **Consistent language tab labels**: When an example offers multiple languages — in a `<CodeGroup>` or across `<Tab title="...">` blocks — label every tab with the same canonical name everywhere: `Python`, `TypeScript` (never `Typescript`), `Bash`. In a `<CodeGroup>`, give each fence a lowercase lexer **and** that canonical title (e.g. a `python` fence titled `Python`); never leave a language fence untitled. The reader's **Python/TypeScript** choice carries from page to page via `code-group-language-persist.js` (repo root), which matches those two labels case-insensitively — so inconsistent casing of them silently resets it. Other labels like `Bash` are only for in-page consistency and are intentionally not persisted across pages. Don't add a competing per-page persistence script.
- **ARIA chat examples**: When a task can be delegated to ARIA end-to-end in the W&B app, add a compact chat example as a third content modality alongside code examples and UI click sequences. Use **ARIA** as the public-facing product name in prose and in fence titles.

  Default format: a user prompt and a concise ARIA response. Don't include reasoning or thinking steps unless the page specifically needs them for clarity; longer walkthroughs belong on [ARIA overview](/aria/overview). Place chat examples in the first section where the ARIA-delegable task appears, not at the top of the page unless the whole page is about chatting with ARIA.

  Import `/snippets/AriaChatBubbles.jsx` and render the component with `prompt` and `response` props:

  ```mdx
  import { AriaChatBubbles } from '/snippets/AriaChatBubbles.jsx';

  <AriaChatBubbles
    prompt="Your user prompt here"
    response="A concise example ARIA response"
  />
  ```
- **Live report embeds**: To show a live, interactive W&B report on a page, import `/snippets/WandbReport.jsx` and render it with `src` (the report URL), `title` (an accessible description), and optional `height` (500–800; default 640):

  ```mdx
  import { WandbReport } from '/snippets/WandbReport.jsx';

  <WandbReport
    src="https://wandb.ai/ENTITY/PROJECT/reports/Slug--VmlldzoXXXXXXX"
    title="Accessible description of the report"
    height={700}
  />
  ```

  Rules for every embed:
  - **Always pair it with prose and a plain Markdown link** to the same report in the surrounding text, stating what the reader should take from it. Agents, the llms.txt export, and the translation pipeline read MDX source, where the iframe is opaque; the link is also the fallback wherever third-party frames are blocked. This is enforced by CI (`scripts/report-embeds/check_embeds.py`).
  - **Register the report** in `scripts/report-embeds/registry.yaml` — see [`scripts/report-embeds/README.md`](scripts/report-embeds/README.md) for the fields and workflow.
  - **The report must be viewable by anonymous visitors**: a report in a public project, or one shared via a view-only link (`Share` → "anyone with the link can view"; see [View-only report links](/models/reports/cross-project-reports#view-only-report-links)). The URL — including any `?accessToken=` — ships in public source and git history, so treat the report as public forever and never embed sensitive data.
  - **Regular reports only, not Fully Connected articles.** FC articles keep their full blog chrome in a frame and look broken. Regular reports render in a slim embed view.
  - **Keep it skinny and sparse**: prefer purpose-built reports (ideally a single panel grid), one or two per page maximum. Each iframe boots the full W&B app, is a fixed height with inner scroll (no auto-resize), and renders its own light theme regardless of the docs dark-mode toggle — so article-length reports show as a tall scroll region, not the whole page at once.
  - English sources only — never add embeds to `ja/`, `ko/`, or `fr/` copies.

## Working with the repository

### When making changes

1. Read the surrounding content for context.
2. Apply the `style-guide` skill for style (see [Style](#style)).
3. Verify code examples follow repository patterns.
4. Use the `.editorconfig` settings for consistent formatting.

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

1. For new content, follow the `style-guide` skill (see [Style](#style)).
2. When practical, keep style changes separate from content changes.

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
