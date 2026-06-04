# CLI reference intro snippets

This directory contains introductory content for W&B CLI commands that's included in the auto-generated CLI reference pages. This README is for docs contributors who want to add or update intro content for a CLI command reference page.

## Purpose

The `scripts/cli-docs-generator.py` script auto-generates the CLI reference pages in `models/ref/cli/` from the W&B CLI source code. However, some commands benefit from additional context, examples, or explanations beyond what's in the CLI help text. This directory holds that supplementary content.

## How it works

The generation process works as follows:

1. Create **snippet files** here with the naming pattern `wandb-[COMMAND].mdx`.
   - Example: `wandb-login.mdx`, `wandb-sync.mdx`, `wandb-verify.mdx`
   - For nested commands, use the full command with dashes: `wandb-artifact-get.mdx`

2. The **generation script** automatically detects these snippets and:
   - Adds an import statement at the top of the generated page.
   - Includes the snippet content immediately after the front matter, before the Usage section.
   - For pages without snippets, adds a commented-out template that shows how to add one.

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

## Add intro content for a command

Follow these steps to add introductory content that appears above the auto-generated reference for a CLI command.

1. Create a new snippet with the correct naming, `wandb-[COMMAND-NAME].mdx`. The generator uses this filename to match the snippet to its command.
2. Write the intro content in MDX format, which can include Markdown and code blocks.
3. In the generated CLI reference for the command, uncomment the lines to include and use the snippet. You don't need to regenerate all of the references.
4. The next time you generate the references, the generator automatically detects and includes your snippet.

Once you complete these steps, the snippet appears on the command's reference page immediately above the Usage section, and persists across subsequent regenerations.

## Guidelines

Apply the following guidelines when authoring snippet content to keep intro sections consistent across CLI reference pages.

- Keep content concise and focused on what users need to know before the command details.
- Use code blocks for examples.
- Snippets must avoid duplicating information that's already in the auto-generated command options or arguments.
- After you add a snippet, test locally to ensure it renders correctly.
