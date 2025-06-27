---
description: Overview of W&B Artifacts, how they work, and how to get started using them.
menu:
  default:
    identifier: artifacts
    parent: core
title: Artifacts
url: guides/artifacts
cascade:
- url: guides/artifacts/:filename
weight: 1
---


{{< cta-button productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifact_fundamentals.ipynb" >}}

Use W&B Artifacts to track and version data as the inputs and outputs of your [W&B Runs]({{< relref "/guides/models/track/runs/" >}}). For example, a model training run might take in a dataset as input and produce a trained model as output. You can log hyperparameters, metadata, and metrics to a run, and you can use an artifact to log, track, and version the dataset used to train the model as input and another artifact for the resulting model checkpoints as output.

## Use cases
You can use artifacts throughout your entire ML workflow as inputs and outputs of [runs]({{< relref "/guides/models/track/runs/" >}}). You can use datasets, models, or even other artifacts as inputs for processing.

{{< img src="/images/artifacts/artifacts_landing_page2.png" >}}

| Use Case               | Input                       | Output                       |
|------------------------|-----------------------------|------------------------------|
| Model Training         | Dataset (training and validation data)     | Trained Model                |
| Dataset Pre-Processing | Dataset (raw data)          | Dataset (pre-processed data) |
| Model Evaluation       | Model + Dataset (test data) | [W&B Table]({{< relref "/guides/models/tables/" >}})                        |
| Model Optimization     | Model                       | Optimized Model              |


{{% alert %}}
The proceeding code snippets are meant to be run in order.
{{% /alert %}}

## Create an artifact

Create an artifact with four lines of code:
1. Create a [W&B run]({{< relref "/guides/models/track/runs/" >}}).
2. Create an artifact object with the [`wandb.Artifact`]({{< relref "/ref/python/sdk/classes/artifact" >}}) API.
3. Add one or more files, such as a model file or dataset, to your artifact object.
4. Log your artifact to W&B.

For example, the proceeding code snippet shows how to log a file called `dataset.h5` to an artifact called `example_artifact`:

```python
import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()

# Logs the artifact version "my_data" as a dataset with data from dataset.h5
```

- The `type` of the artifact affects how it appears in the W&B platform. If you do not specify a `type`, it defaults to `unspecified`.
- Each label of the dropdown represents a different `type` parameter value. In the above code snippet, the artifact's `type` is `dataset`.

{{% alert %}}
See the [track external files]({{< relref "./track-external-files.md" >}}) page for information on how to add references to files or directories stored in external object storage, like an Amazon S3 bucket. 
{{% /alert %}}

## Download an artifact
Indicate the artifact you want to mark as input to your run with the [`use_artifact`]({{< relref "/ref/python/sdk/classes/run#use_artifact" >}}) method.

Following the preceding code snippet, this next code block shows how to use the `training_dataset` artifact: 

```python
artifact = run.use_artifact(
    "training_dataset:latest"
)  # returns a run object using the "my_data" artifact
```
This returns an artifact object.

Next, use the returned object to download all contents of the artifact:

```python
datadir = (
    artifact.download()
)  # downloads the full `my_data` artifact to the default directory.
```

{{% alert %}}
You can pass a custom path into the `root` [parameter]({{< relref "/ref/python/sdk/classes/artifact" >}}) to download an artifact to a specific directory. For alternate ways to download artifacts and to see additional parameters, see the guide on [downloading and using artifacts]({{< relref "./download-and-use-an-artifact.md" >}}).
{{% /alert %}}


## Next steps
* Learn how to [version]({{< relref "./create-a-new-artifact-version.md" >}}) and [update]({{< relref "./update-an-artifact.md" >}}) artifacts.
* Learn how to trigger downstream workflows or notify a Slack channel in response to changes to your artifacts with [automations]({{< relref "/guides/core/automations/" >}}).
* Learn about the [registry]({{< relref "/guides/core/registry/" >}}), a space that houses trained models.
* Explore the [Python SDK]({{< relref "/ref/python/sdk/classes/artifact" >}}) and [CLI]({{< relref "/ref/cli/wandb-artifact/" >}}) reference guides.
