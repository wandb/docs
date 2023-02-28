# Weights & Biases Documentation
The Weights & Biase Docs ([https://docs.wandb.ai/](https://docs.wandb.ai/)) is built using [Docusaurus 2](https://docusaurus.io/), a static website generator built with React. The high level overview of the doc writing process is:

<div align='center'>Edit markdown → confirm changes don’t break the docs → create a pull request for review.</div>  

&nbsp;

## Docusaurus dependencies
Satisfy the following dependencies to create, build, and locally serve W&B Docs on your local machine:

- Install [Node.js](https://nodejs.org/en/download/) version 16.14 or above.
- Install Yarn. It is recommended to install Yarn through the [npm package manager](http://npmjs.org/), which comes bundled with [Node.js](https://nodejs.org/) when you install it on your system.
- An IDE (e.g. VS Studio) or Text Editor (e.g. Sublime)

&nbsp;

## Install W&B Doc dependencies
After you have forked and cloned wandb/docodile, change directory to `wandb/docodile` and install the W&B Doc app dependencies with `yarn`:

```bash
cd docodile

yarn install
```

Test that you can build the docs locally with:

```bash
yarn start
```

This will return the port number where you can preview your changes to the docs. A full list of yarn commands can be found in `docusaurus/package.json`

&nbsp;

## How to edit the docs locally

1. Navigate to your local GitHub repo of `docodile` and pull the latest changes from main:
```bash
cd docodile
git pull origin main
```
2. Create a feature branch off of `main`.
```bash
git checkout -b <your-feature-branch>
```
3. In a new terminal, start a local preview of the docs with `yarn start`.
```bash
yarn start
```
This will return the port number where you can preview your changes to the docs.  

4. Make your changes on the new branch.   
5. Check your changes are rendered correctly.  
6. Build the static files of your website for production with `yarn build`.
```bash
yarn build
```
7. Commit the changes to the branch.
```bash
git commit -m 'Useful commit message.'
```
8. Push the branch to GitHub.
```bash
git push origin <your-feature-branch>
```
9. Open a pull request from the new branch to the original repo.


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


