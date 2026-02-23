# CLI Reference Intro Snippets

This directory contains introductory content for W&B CLI commands that gets included in the auto-generated CLI reference pages.

## Purpose

The CLI reference pages (`models/ref/cli/`) are auto-generated from the W&B CLI source code using `scripts/cli-docs-generator.py`. However, some commands benefit from additional context, examples, or explanations beyond what's in the CLI help text. This directory holds that supplementary content.

## How It Works

1. **Snippet files** are created here with the naming pattern: `wandb-{command}.mdx`
   - Example: `wandb-login.mdx`, `wandb-sync.mdx`, `wandb-verify.mdx`
   - For nested commands, use the full command with dashes: `wandb-artifact-get.mdx`

2. **Generation script** automatically detects these snippets and:
   - Adds an import statement at the top of the generated page
   - Includes the snippet content right after the front matter, before the Usage section
   - For pages without snippets, adds a commented-out template showing how to add one

3. **Generated page structure**:
   ```markdown
   ---
   title: wandb login
   ---
   import WandbLogin from "/snippets/_includes/cli/wandb-login.mdx";

   <WandbLogin/>

   ## Usage
   [auto-generated content from CLI]
   ```

## Adding Intro Content for a Command

To add introductory content for a command:

1. Create a new snippet with the correct naming: `wandb-{command-name}.mdx`.
2. Write the intro content in MDX format (can include markdown, code blocks, etc.).
3. In the generated CLI reference for the command, uncomment the lines to include and use the snippet. No need to regenerate all of the references.
4. Next time the references are generated, the generator will automatically detect and include your snippet.


## Guidelines

- Keep content concise and focused on what users need to know before the command details
- Use code blocks for examples
- Avoid duplicating information that's already in the auto-generated command options/arguments
- Test locally after adding a snippet to ensure it renders correctly
