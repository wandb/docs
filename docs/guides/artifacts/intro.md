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

<CTAButtons productLink="https://github.com/wandb/docodile" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb"/>

Use W&B Artifacts to track and version any serialized data as the inputs and outputs of your [W&B Runs](../runs/intro.md). For example, a model training run might take in a dataset as input and trained model as output. In addition to logging hyper-parameters and metadata to a run, you can use an artifact to log the dataset used to train the model as input and the resulting model checkpoints as outputs. You will always be able answer the question “what version of my dataset was this model trained on”.

In summary, with W&B Artifacts, you can:
* [View where a model came from, including data it was trained on](./explore-and-traverse-an-artifact-graph.md).
* [Version every dataset change or model checkpoint](./create-a-new-artifact-version.md).
* [Easily reuse models and datasets across your team](./download-and-use-an-artifact.md).

![](/images/artifacts/artifacts_landing_page2.png)


The diagram above demonstrates how you can use artifacts throughout your entire ML workflow; as inputs and outputs of [runs](../runs/intro.md). 

## How it works

Create an artifact with four lines of code:
1. Create a [W&B run](../runs/intro.md).
2. Create an artifact object with the [`wandb.Artifact`](../../ref/python/artifact.md) API.
3. Add one or more files, such as a model file or dataset, to your artifact object. 
4. Log your artifact to W&B.


```python showLineNumbers
run = wandb.init(project="artifacts-example", job_type='add-dataset')
artifact = wandb.Artifact(name='my_data', type='dataset')
artifact.add_file(local_path='./dataset.h5') # Add dataset to artifact
run.log_artifact(artifact) # Logs the artifact version "my_data:v0"
```

:::tip
The preceding code snippet, and the [colab linked on this page](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb), show how to track files by uploading them to W&B. See the [track external files](./track-external-files.md) page for information on how to add references to files or directories that are stored in external object storage (for example, in an Amazon S3 bucket). 
:::

## How to get started

Depending on your use case, explore the following resources to get started with W&B Artifacts:

* If this is your first time using W&B Artifacts, we recommend you go through the [Artifacts Colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb#scrollTo=fti9TCdjOfHT).
* Read the [artifacts walkthrough](./artifacts-walkthrough.md) for a step-by-step outline of the W&B Python SDK commands you could use to create, track, and use a dataset artifact.
* Explore this chapter to learn how to:
  * [Construct an artifact](./construct-an-artifact.md) or a [new artifact version](./create-a-new-artifact-version.md)
  * [Update an artifact](./update-an-artifact.md)
  * [Download and use an artifact](./download-and-use-an-artifact.md).
  * [Delete artifacts](./delete-artifacts.md).
* Explore the [Python SDK Artifact APIs](../../ref/python/artifact.md) and [Artifact CLI Reference Guide](../../ref/cli/wandb-artifact/README.md).
