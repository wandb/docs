# Weights & Biases Documentation

The Weights & Biases Docs ([https://docs.wandb.ai/](https://docs.wandb.ai/)) is built using Mintlify, a static website generator. The high level overview of the doc writing process is:

1. Edit the desired files
2. Create a pull request proposing your changes
3. Confirm changes don’t break the docs, which will be tested by CI
4. Respond to feedback from the W&B docs team and CI checks

After this, someone from the docs team will merge the PR and it will go live in a matter of minutes!

## Quickstart

This section shows how to edit a page or report a bug from within your browser without cloning the repository or installing anything locally. To edit multiple files or build the documentation locally, refer to the [Prerequisites](#prerequisites) and the following sections.

### Edit a page

1. To edit a page you're reading on docs.wandb.com, scroll to the bottom of the page and click **Edit page** to open the Markdown file in the GitHub editor.

   To edit a page from https://github.com/wandb/docs, browse to or search for the page, then click the pencil icon to open the Markdown file in the GitHub editor.
1. Edit the page, then click **Commit changes**. In the dialog, choose to create a new branch, then specify:
  - A name for the branch
  - A commit message that describes the change. By default, this becomes the pull request title.
  - An optional extended description. By default, this becomes the pull request body.
1. Click **Propose change**. A new branch is created with the commit you just created. A new dialog opens where you can create a pull request.
1. Optionally edit the pull request's title and description. Markdown is allowed. You can refer to a PR or issue by number or URL, and you can refer to a JIRA issue by its ID.
1. Click **Create pull request**. A member of @docs-team reviews your changes, provides feedback, and works with you to merge the change.

### Report a bug

If you work for Weights & Biases, file a doc JIRA, using this template: https://wandb.atlassian.net/secure/CreateIssue!default.jspa?project=DOCS.

{/*
To report a bug on a page you're reading on docs.wandb.com:
1. Scroll to the bottom of the page and click **Report issue**.
1. Provide a title and optionally edit the description, then click **Create**.
*/}

To report a bug from https://github.com/wandb/docs:
1. Click the **Issues** tab.
1. Click **New issue**. Optionally select a template, then click **Create**.
1. Provide a title and a description. If applicable, include the URL of the page with the bug. Click **Create**.

## Prerequisites

The `mint` CLI is required for local builds. See https://www.mintlify.com/docs/installation. Install it from `npx` or from Homebrew.


### macOS

After cloning this repo:
1. `cd` into the clone.
2. Create a working branch:
    ```shell
    git checkout -b my_working_branch origin/main
    ```
3. Build and serve the docs locally:
    ```shell
    mint dev
    ```
    By default, content is served from `https://localhost:3000/`.
4. Check for broken links and images:
  ```shell
  mint broken-links
  ```
5. Push your branch to `origin`:
    ```shell
    git push origin my_working_branch
    ```
6. The first time you push a new branch, the terminal output includes a link to create a PR. Click the link and create a draft PR. CI tests will run.
7. When you are satisfied with your PR and all tests pass, click **Ready for review** to convert the draft PR to a reviewable PR. A member of the W&B docs team will review your changes and give feedback.
8. When all feedback is addressed and the PR is approved, the reviewer will merge the PR.


## Exiting `mint dev`

To interrupt the Mintlify server, type `CTRL+C` in the terminal where it is running.

## Mintlify compatibility
Mintlify processes Markdown files in `.mdx` format, which has stricter requirements than `.md`. If a file contains incompatible Markdown, `mint dev` will fail and `mint broken-links` will show errors. For example, HTML comments (`<!-- like this -->`) are not supported. Instead, use this syntax for comments: `{/* like this */}`.

For general guidelinest, refer to the **Create content** section of the [Mintlify documentation](https://www.mintlify.com/docs/). For example:
- [Format text](https://www.mintlify.com/docs/create/text)
- [Lists and tables](https://www.mintlify.com/docs/create/list-table)
- [Images and embeds](https://www.mintlify.com/docs/create/image-embeds)
- [Callouts](https://www.mintlify.com/docs/components/callouts)
- [Cards](https://www.mintlify.com/docs/components/cards)
- [Tabs](https://www.mintlify.com/docs/components/tabs)
This list of links is not exhaustive. Check the Mintlify documentation for full details.

## How to edit the docs locally

1. Navigate to your local clone this repo and pull the latest changes from main:
    ```bash
    git pull origin main
    ```
2. Create a feature branch off of `main`.
    ```bash
    git checkout -b <your-feature-branch>
    ```
3. After installing the prerequsites documented above, start a local preview of the docs.
    ```bash
    mint dev
    ```
    This will return the localhost URL and port number where you can preview your changes to the docs as you make them (e.g. `https://localhost:3000`).
4. Make your changes on the new branch.
5. Check your changes are rendered correctly. Any time you save a file, the preview automatically updates.
7. Commit the changes to the branch.
    ```bash
    git add .
    git commit -m 'Useful commit message.'
    ```
8. Push the branch to GitHub.
    ```bash
    git push origin <your-feature-branch>
    ```
9. Open a pull request from the new branch to the original repo.

## What files do I edit?
You can edit most `.mdx` files in the repo directly. Content in a few directories is generated from upstream code. Edit the code in the source repo, not in `wandb/docs`. After your upstream change is merged and released, when the relevant content is next generated, your change will be included.

| Directory | Source repo |
|-----------|-------------|
| `/ref/python/`| [`wandb/wandb`](https://github.com/wandb/wandb) |
| `/models/ref/cli/` | [`wandb/wandb`](https://github.com/wandb/wandb)|
| `/weave/api-reference/` | [`wandb/weave`](https://github.com/wandb/weave) |
| `/weave/cookbooks/` | [`wandb/weave`](https://github.com/wandb/weave) |
| `/weave/reference/` | [`wandb/weave`](https://github.com/wandb/weave) |

## AI resources

The `AGENTS.md` file contains guidance and style conventions specifically designed for AI agents working with this repository. This includes:

- Project context and your role as a W&B docs team member
- Core principles for writing clear, accurate documentation
- Comprehensive style guide with examples
- Code formatting standards and best practices
- Collaboration guidelines

For complex, multi-step procedures, see the `runbooks/` directory which contains step-by-step instructions for tasks like testing GitHub Actions changes.

If you're using an AI agent to help with documentation tasks, provide `AGENTS.md` as context to ensure consistent, high-quality contributions.

## License

The source for this documentation is offered under the Apache 2.0 license. 

## Notices

- [LICENSE](LICENSE)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt)

## Attributions

- This project uses Mintlify. [License](https://github.com/mintlify)
- A dependency of Docsy is the `caniuse-lite` package, offered under CC-BY-4.0. [License](https://github.com/browserslist/caniuse-lite/blob/main/LICENSE)
- Another dependency of Docsy is Font Awesome Free, offered under the CC-BY-4.0, SIL OFL 1.1, and MIT licenses. [License notice](https://github.com/FortAwesome/Font-Awesome/blob/master/LICENSE.txt)
- This site is built using Hugo, a static site generator. [License](https://github.com/gohugoio/hugo/blob/master/LICENSE)
