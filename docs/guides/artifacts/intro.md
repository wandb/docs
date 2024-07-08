---
slug: /guides/artifacts
description: >-
  Overview of what W&B Artifacts are, how they work, and how to get started
  using W&B Artifacts.
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Artifacts

<CTAButtons productLink="https://wandb.ai/wandb/arttest/artifacts/model/iv3_trained/5334ab69740f9dda4fed/lineage" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb"/>

Use W&B Artifacts to track and version data as the inputs and outputs of your [W&B Runs](../runs/intro.md). For example, a model training run might take in a dataset as input and produce a trained model as output. In addition to logging hyperparameters, metadata and metrics to a run, you can use an artifact to log, track and version the dataset used to train the model as input and another artifact for the resulting model checkpoints as outputs.

## Use cases
You can use artifacts throughout your entire ML workflow as inputs and outputs of [runs](../runs/intro.md). You can use datasets, models, or even other artifacts as inputs for processing.

![](/images/artifacts/artifacts_landing_page2.png)

| Use Case               | Input                       | Output                       |
|------------------------|-----------------------------|------------------------------|
| Model Training         | Dataset (training and validation data)     | Trained Model                |
| Dataset Pre-Processing | Dataset (raw data)          | Dataset (pre-processed data) |
| Model Evaluation       | Model + Dataset (test data) | [W&B Table](../tables/intro.md)                        |
| Model Optimization     | Model                       | Optimized Model              |


## Create an artifact

Create an artifact with four lines of code:
1. Create a [W&B Run](../runs/intro.md).
2. Create an artifact object with the [`wandb.Artifact`](../../ref/python/artifact.md) API.
3. Add one or more files, such as a model file or dataset, to your artifact object. In this example, you'll add a single file.
4. Log your artifact to W&B.


```python
run = wandb.init(project = "artifacts-example", job_type = "add-dataset")
run.log_artifact(data = "./dataset.h5", name = "my_data", type = "dataset" ) # Logs the artifact version "my_data" as a dataset with data from dataset.h5
```

:::tip
See the [track external files](./track-external-files.md) page for information on how to add references to files or directories stored in external object storage, like an Amazon S3 bucket. 
:::

## Download an artifact
Indicate the artifact you want to mark as input to your run with the [`use_artifact`](../../ref/python/run.md#use_artifact) method, which returns an artifact object:

```python
artifact = run.use_artifact("my_data:latest") #returns a run object using the "my_data" artifact
```

Then, use the returned object to download all contents of the artifact:

```python
datadir = artifact.download() #downloads the full "my_data" artifact to the default directory.
```

:::tip
You can pass a custom path into the `root` [parameter](../../ref/python/artifact.md) to download an artifact to a specific directory. For alternate ways to download artifacts and to see additional parameters, see the guide on [downloading and using artifacts](./download-and-use-an-artifact.md)
:::

## Next steps
* Learn how to [version](./create-a-new-artifact-version.md), [update](./update-an-artifact.md), or [delete](./delete-artifacts.md) artifacts.
* Learn how to trigger downstream workflows in response to changes to your artifacts with [artifact automation](./project-scoped-automations.md).
* Learn about the [model registry](../model_registry/intro.md), a space that houses trained models.
* Explore the [Python SDK](../../ref/python/artifact.md) and [CLI](../../ref/cli/wandb-artifact/README.md) reference guides.
