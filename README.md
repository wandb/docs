# Weights & Biases Documentation [Japanese]

[![Build the docs](https://github.com/wandb/docodile/actions/workflows/build_docs.yml/badge.svg)](https://github.com/wandb/docodile/actions/workflows/build_docs.yml)

The Weights & Biases Docs ([https://docs.wandb.ai/](https://docs.wandb.ai/)) is built using [Docusaurus 2](https://docusaurus.io/), a static website generator built with React. The high level overview of the doc writing process is:
<div align='center'>Edit markdown → confirm changes don’t break the docs → create a pull request for review.</div>  

&nbsp;

## Install dependencies

Satisfy the following dependencies to create, build, and locally serve W&B Docs on your local machine:

- (Recommended) Install [`nvm`](https://github.com/nvm-sh/nvm) to manage your node.js versions.
- Install [Node.js](https://nodejs.org/en/download/) version 18.0.0.
  ```node
  nvm install 18.0.0
  ```
- Install Yarn. It is recommended to install Yarn through the [npm package manager](http://npmjs.org/), which comes bundled with [Node.js](https://nodejs.org/) when you install it on your system.
  ```yarn
  npm install --global yarn
  ```
- Install an IDE (e.g. VS Studio) or Text Editor (e.g. Sublime)

## Install W&B Doc app
1. After you have forked and cloned wandb/docodile, change directory to `wandb/docodile` and install the W&B Doc app dependencies with `yarn`:

```bash
cd docodile

yarn install
```

2. Install pre-commit hooks to ensure that the code snippets in the docs are formatted correctly:

```bash
pip install pre-commit
pre-commit install
# pre-commit run blacken-docs --all-files  # to run manually on all files
```

&nbsp;

## How to edit the docs locally
> ⚠️ Make sure that you push changes to the `japanese_docs` branch and not `main`. ⚠️

&nbsp;


1. Navigate to your local GitHub repo of `docodile`
```bash
cd docodile
```

2. Checkout a branch bashed off of `japanese_docs` branch and track `japanese_docs`:
```bash
git checkout -b your-feature-branch origin/japanese_docs
```

3. Make changes to the appropriate markdown files.
4. Build a local version of the site to make sure changes don't break the docs:

```bash
bash ./scripts/build-local-docs.sh
```

If the build is succesful, go to the next step.

5. Commit your changes:
```bash
git add -u
git commit -m "Your commit message"
```

6. Push to `japanese_docs` branch on remote:
```bash
git push origin your-feature-branch:japanese_docs
```


&nbsp;

## What files do I edit?

There are two types of docs in our Docs site: Developer Guide and the API Reference Guide.

### Developer Guide

All markdown files for the [W&B Developer Guide](https://docs.wandb.ai/) are stored in:

```bash
docodile/docs/guides/
```

Each Chapter in the Developer Guide has its own folder in `docodile/docs/guides` . For example, the Artifacts Chapter ([https://docs.wandb.ai/guides/artifacts](https://docs.wandb.ai/guides/artifacts)) has a chapter called “artifacts”:

```bash
# Directory with Artifacts markdown content
docodile/docs/guides/artifacts
```

The PR you create will get reviewed and (if approved) merged by the Docs Team.

### API Reference Guide

All markdown files for the [W&B API Reference Guide](https://docs.wandb.ai/ref) are stored in:

```bash
docodile/docs/ref
```

The markdown files are generated from docstrings in wandb/wandb. Modify the docstring from the appropriate Python Class, function, or CLI definition to to update the public-facing documentation API.

Once you are done, create a pull request from wandb/wandb. The PR you create will get reviewed and (if approved) merged by the SDK Team. The Docs are updated when the W&B SDK Team makes an W&BSDK Release. SDK Releases occur about every 2-4 weeks.

&nbsp;
