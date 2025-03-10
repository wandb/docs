# Weights & Biases Documentation

The Weights & Biases Docs ([https://docs.wandb.ai/](https://docs.wandb.ai/)) is built using Docsy, a technical documentation theme for Hugo, a static website generator. The high level overview of the doc writing process is:

1. Edit the desired files
2. Create a pull request proposing your changes
3. Confirm changes don’t break the docs, which will be tested by CI
4. Respond to feedback from the W&B docs team and CI checks

After this, someone from the docs team will merge the PR and it will go live in a matter of minutes!

## Prerequisites

A current version of NodeJS is required; ideally, something newer than version 20. If you still need to use an old version of node for other projects, we suggest using `nvm` and setting up version 20 using that, which you can swap into with the `use` command:

```
nvm install 20
nvm use 20
```

### macOS

After cloning this repo, `cd` into your local clone directory and these commands:

```
brew install go
brew install hugo
brew install npm
npm install
hugo mod get -u
```

The last lines critical, as it downloads Hugo, the [Docsy](https://docsy.dev) module for Hugo, and the dependencies of each.

## Running the website locally

```bash
hugo server
```

## Exiting `hugo server`

Hit `CTRL+C` in the terminal that is showing `hugo` activity to interrupt the server and exit to the terminal prompt.

## Hugo and Docsy shortcodes

- We use Docsy's `alert` shortcode for admonitions. The alert `color` determines the admonition type. Refer to [alerts](https://www.docsy.dev/docs/adding-content/shortcodes/#alert) for details. Examples:
    ```markdown
    {{% alert %}}
    Only **public** reports are viewable when embedded.
    {{% /alert %}}
    ```
    ```markdown
    {{% alert title="Undo changes to your workspace" %}}
    Select the undo button (arrow that points left) to undo any unwanted changes.
    {{% /alert %}}
    ```
    ```markdown
    {{% alert title="Warning" color="warning" %}}
    This is a warning.
    {{% /alert %}}
    ```
- We use Docsy's `tabpane` and `tab` shortcodes for tabbed content. Refer to [tabpane](https://www.docsy.dev/docs/adding-content/shortcodes/#tabpane) for details. Example:
    ```markdown
    {{< tabpane text=true >}}
      {{% tab header="GitHub repository dispatch" value="github" %}}
        ... Markdown contents ...
      {{% /tab %}}

      {{% tab header="Microsoft Teams notification" value="microsoft"%}}
        ... Markdown contents ...
      {{% /tab %}}
    {{< /tabpane >}}
    ```
- We use a custom `img` shortcode for images. It is implemented in `layouts/shortcodes/img.html`.  Examples:
    ```markdown
    {{< img src="/images/app_ui/automated_workspace.svg" >}}
    ```
    ```markdown
    {{< img src="/images/app_ui/automated_workspace.svg" alt="automated workspace icon" >}}
    ```
    ```markdown
    {{< img src="/images/app_ui/automated_workspace.svg" alt="automated workspace icon" width="32px" >}}
    ```
    ```markdown
    {{< img src="/images/app_ui/demo_make_a_custom_chart_bar_chart.gif" alt="Creating a custom bar chart showing accuracy across runs in a project" max-width="90%" >}}
    ```
- We use a custom `ctabutton` shortcode to link to Colab notebooks. It is implemented in `layouts/shortcodes/cta-button.html`. Examples:
    ```markdown
    {{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W&B_Sweeps_with_XGBoost.ipynb" >}}
    ```
- We use a custom `prism` shortcode to load titled code examples from `static/` within the docs repo. Example for a file stored in `static/webhook_test.sh`:
    ```markdown
    {{< prism file="/webhook_test.sh" title="webhook_test.sh">}}{{< /prism >}}
    ```

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
hugo server
```

This will return the localhost URL and port number where you can preview your changes to the docs as you make them (e.g. `https://localhost:1313`).

4. Make your changes on the new branch.
5. Check your changes are rendered correctly. Any time you save a file, the preview should automatically update.
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

There are two types of docs in our Docs site: Developer Guide and the API Reference Guide.

### Developer Guide

All markdown files for the [W&B Developer Guide](https://docs.wandb.ai/) are stored in:

```bash
content/guides/
```

The PR you create will get reviewed and (if approved) merged by the Docs Team.

### API Reference Guide

All markdown files for the [W&B API Reference Guide](https://docs.wandb.ai/ref) are stored in:

```bash
content/ref
```

The markdown files are generated from docstrings in https://github.com/wandb/wandb. Modify the docstring from the appropriate Python Class, function, or CLI definition to update the public-facing documentation API.

Once you are done, create a pull request from https://github.com/wandb/wandb. The PR you create will get reviewed and (if approved) merged by the SDK Team. The Docs are updated when the W&B SDK Team makes an W&BSDK Release. SDK Releases occur about every 2-4 weeks.

## License

The source for this documentation is offered under the Apache 2.0 license. 

## Notices

- [LICENSE](LICENSE)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [THIRD-PARTY-NOTICES.txt](THIRD-PARTY-NOTICES.txt)

## Attributions

- This project uses Docsy, a Hugo theme by Google. [License](https://github.com/google/docsy/blob/main/LICENSE)
- A dependency of Docsy is the `caniuse-lite` package, offered under CC-BY-4.0. [License](https://github.com/browserslist/caniuse-lite/blob/main/LICENSE)
- Another dependency of Docsy is Font Awesome Free, offered under the CC-BY-4.0, SIL OFL 1.1, and MIT licenses. [License notice](https://github.com/FortAwesome/Font-Awesome/blob/master/LICENSE.txt)
- This site is built using Hugo, a static site generator. [License](https://github.com/gohugoio/hugo/blob/master/LICENSE)
