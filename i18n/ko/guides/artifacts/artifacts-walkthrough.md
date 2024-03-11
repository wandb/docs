---
description: >-
  Artifacts quickstart shows how to create, track, and use a dataset artifact
  with W&B.
displayed_sidebar: default
---

# Walkthrough

<head>
  <title>Walkthrough</title>
</head>


The proceeding walkthrough demonstrates the major W&B Python SDK commands used to create, track, and use a dataset artifact from [W&B Runs](../runs/intro.md).

## 1. Log into W&B

Import the W&B library and log in to W&B. You will need to sign up for a free W&B account if you have not done so already.

```python
import wandb

wandb.login()
```

## 2. Initialize a run

Use the [`wandb.init()`](../../ref/python/init.md) API to generate a background process to sync and log data as a W&B Run. Provide a project name and a job type:

```python
# Create a W&B Run. Here we specify 'dataset' as the job type since this example
# shows how to create a dataset artifact.
run = wandb.init(project="artifacts-example", job_type="upload-dataset")
```

## 3. Create an artifact object

Create an artifact object with the [`wandb.Artifact()`](../../ref/python/artifact.md) API. Provide a name for the artifact and a description of the file type for the `name` and `type` parameters, respectively.

For example, the following code snippet demonstrates how to create an artifact called `‘bicycle-dataset’` with a `‘dataset’` label:

```python
artifact = wandb.Artifact(name="bicycle-dataset", type="dataset")
```

For more information about how to construct an artifact, see [Construct artifacts](./construct-an-artifact.md).

## Add the dataset to the artifact

Add a file to the artifact. Common file types include models and datasets. The following example adds a dataset named `dataset.h5` that is saved locally on our machine to the artifact:

```python
# Add a file to the artifact's contents
artifact.add_file(local_path="dataset.h5")
```

Replace the filename `dataset.h5` in the preceding code snippet with the path to the file you want to add to the artifact.

## 4. Log the dataset

Use the W&B run objects `log_artifact()` method to both save your artifact version and declare the artifact as an output of the run.

```python
# Save the artifact version to W&B and mark it
# as the output of this run
run.log_artifact(artifact)
```

A `'latest'` alias is created by default when you log an artifact. For more information about artifact aliases and versions, see [Create a custom alias](./create-a-custom-alias.md) and [Create new artifact versions](./create-a-new-artifact-version.md), respectively.

## 5. Download and use the artifact

The following code example demonstrates the steps you can take to use an artifact you have logged and saved to the W&B servers.

1. First, initialize a new run object with **`wandb.init()`.**
2. Second, use the run objects [`use_artifact()`](../../ref/python/run.md#use_artifact) method to tell W&B what artifact to use. This returns an artifact object.
3. Third, use the artifacts [`download()`](../../ref/python/artifact.md#download) method to download the contents of the artifact.

```python
# Create a W&B Run. Here we specify 'training' for 'type'
# because we will use this run to track training.
run = wandb.init(project="artifacts-example", job_type="training")

# Query W&B for an artifact and mark it as input to this run
artifact = run.use_artifact("bicycle-dataset:latest")

# Download the artifact's contents
artifact_dir = artifact.download()
```

Alternatively, you can use the Public API (`wandb.Api`) to export (or update data) data already saved in a W&B outside of a Run. See [Track external files](./track-external-files.md) for more information.
