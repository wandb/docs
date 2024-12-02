# Weights & Biases Documentation

The Weights & Biases Docs ([https://docs.wandb.ai/](https://docs.wandb.ai/)) is built using Docsy, a technical documentation theme for Hugo, a static website generator. The high level overview of the doc writing process is:

<div align='center'>Edit markdown → confirm changes don’t break the docs → create a pull request for review.</div>

From there, someone from the Docs Team will review the PR and merge it. 

## Get set up

Run these commands from the root of your local repo clone:

```
brew install go
brew install hugo
brew install npm
brew install nvm
nvm install 20
nvm use 20
npm install
```

## Running the website locally

```bash
hugo server
```

## Exiting `hugo server`

Hit `CTRL+C` in the terminal that is showing `hugo` activity to interrupt the server and exit to the terminal prompt.

## Editing style

Style overrides are in `/assets/scss/_variables_project.scss`. Here we can override all the styles that ship with the Docsy theme. O

## Troubleshooting

As you run the website locally, you may run into the following error:

```console
$ hugo server
WARN 2023/06/27 16:59:06 Module "project" is not compatible with this Hugo version; run "hugo mod graph" for more information.
Start building sites …
hugo v0.101.0-466fa43c16709b4483689930a4f9ac8add5c9f66+extended windows/amd64 BuildDate=2022-06-16T07:09:16Z VendorInfo=gohugoio
Error: Error building site: "C:\Users\foo\path\to\docsy-example\content\en\_index.md:5:1": failed to extract shortcode: template for shortcode "blocks/cover" not found
Built in 27 ms
```

This error occurs if you are running an outdated version of Hugo. As of docsy theme version `v0.7.0`, hugo version `0.110.0` or higher is required.
See this [section](https://www.docsy.dev/docs/get-started/docsy-as-module/installation-prerequisites/#install-hugo) of the user guide for instructions on how to install Hugo.

Or you may be confronted with the following error:

```console
$ hugo server

INFO 2021/01/21 21:07:55 Using config file:
Building sites … INFO 2021/01/21 21:07:55 syncing static files to /
Built in 288 ms
Error: Error building site: TOCSS: failed to transform "scss/main.scss" (text/x-scss): resource "scss/scss/main.scss_9fadf33d895a46083cdd64396b57ef68" not found in file cache
```

This error occurs if you have not installed the extended version of Hugo.
See this [section](https://www.docsy.dev/docs/get-started/docsy-as-module/installation-prerequisites/#install-hugo) of the user guide for instructions on how to install Hugo.

Or you may encounter the following error:

```console
$ hugo server

Error: failed to download modules: binary with name "go" not found
```

This error occurs if you have not installed the `go` programming language on your system.
See this [section](https://www.docsy.dev/docs/get-started/docsy-as-module/installation-prerequisites/#install-go-language) of the user guide for instructions on how to install `go`.


[alternate dashboard]: https://app.netlify.com/sites/goldydocs/deploys
[deploys]: https://app.netlify.com/sites/docsy-example/deploys
[Docsy user guide]: https://docsy.dev/docs
[Docsy]: https://github.com/google/docsy
[example.docsy.dev]: https://example.docsy.dev
[Hugo theme module]: https://gohugo.io/hugo-modules/use-modules/#use-a-module-for-a-theme
[Netlify]: https://netlify.com
[Docker Compose documentation]: https://docs.docker.com/compose/gettingstarted/


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
6. Build the static files of your website:

```bash
yarn docusaurus build
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