---
menu:
  default:
    identifier: spacy
    parent: integrations
title: spaCy
weight: 410
---
[spaCy](https://spacy.io) is a popular "industrial-strength" NLP library: fast, accurate models with a minimum of fuss. As of spaCy v3, Weights and Biases can now be used with [`spacy train`](https://spacy.io/api/cli#train) to track your spaCy model's training metrics as well as to save and version your models and datasets. And all it takes is a few added lines in your configuration.

## Sign up and create an API key

An API key authenticates your machine to W&B. You can generate an API key from your user profile.

{{% alert %}}
For a more streamlined approach, you can generate an API key by going directly to [https://wandb.ai/authorize](https://wandb.ai/authorize). Copy the displayed API key and save it in a secure location such as a password manager.
{{% /alert %}}

1. Click your user profile icon in the upper right corner.
1. Select **User Settings**, then scroll to the **API Keys** section.
1. Click **Reveal**. Copy the displayed API key. To hide the API key, reload the page.

## Install the `wandb` library and log in

To install the `wandb` library locally and log in:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. Set the `WANDB_API_KEY` [environment variable]({{< relref "/guides/models/track/environment-variables.md" >}}) to your API key.

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. Install the `wandb` library and log in.



    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

## Add the `WandbLogger` to your spaCy config file

spaCy config files are used to specify all aspects of training, not just logging -- GPU allocation, optimizer choice, dataset paths, and more. Minimally, under `[training.logger]` you need to provide the key `@loggers` with the value `"spacy.WandbLogger.v3"`, plus a `project_name`. 

{{% alert %}}
For more on how spaCy training config files work and on other options you can pass in to customize training, check out [spaCy's documentation](https://spacy.io/usage/training).
{{% /alert %}}

```python
[training.logger]
@loggers = "spacy.WandbLogger.v3"
project_name = "my_spacy_project"
remove_config_values = ["paths.train", "paths.dev", "corpora.train.path", "corpora.dev.path"]
log_dataset_dir = "./corpus"
model_log_interval = 1000
```

| Name                   | Description                                                                                                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project_name`         | `str`. The name of the W&B Project. The project will be created automatically if it doesn’t exist yet.                                                                                                    |
| `remove_config_values` | `List[str]` . A list of values to exclude from the config before it is uploaded to W&B. `[]` by default.                                                                                                                                                     |
| `model_log_interval`   | `Optional int`. `None` by default. If set, [model versioning]({{< relref "/guides/core/registry/model_registry/" >}}) with [Artifacts]({{< relref "/guides/core/artifacts/" >}})will be enabled. Pass in the number of steps to wait between logging model checkpoints. `None` by default. |
| `log_dataset_dir`      | `Optional str`. If passed a path, the dataset will be uploaded as an Artifact at the beginning of training. `None` by default.                                                                                                            |
| `entity`               | `Optional str` . If passed, the run will be created in the specified entity                                                                                                                                                                                   |
| `run_name`             | `Optional str` . If specified, the run will be created with the specified name.                                                                                                                                                                               |

## Start training

Once you have added the `WandbLogger` to your spaCy training config you can run `spacy train` as usual.

{{< tabpane text=true >}}

{{% tab header="Command Line" value="cli" %}}

```python
python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```python
python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}

{{% tab header="Python notebook" value="notebook" %}}

```notebook
!python -m spacy train \
    config.cfg \
    --output ./output \
    --paths.train ./train \
    --paths.dev ./dev
```

{{% /tab %}}
{{< /tabpane >}}

When training begins, a link to your training run's [W&B page]({{< relref "/guides/models/track/runs/" >}}) will be output which will take you to this run's experiment tracking [dashboard]({{< relref "/guides/models/track/workspaces.md" >}}) in the Weights & Biases web UI.