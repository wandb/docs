---
slug: /guides/data-and-model-versioning
description: Dataset versioning, model versioning, pipeline tracking with flexible and lightweight building blocks
---

# Data and model versioning

Use W&B Artifacts for dataset versioning, model versioning, and tracking dependencies and results across machine learning pipelines. Think of an artifact as a versioned folder of data. You can store entire datasets directly in artifacts, or use artifact references to point to data in other systems like S3, GCP, or your own system.

## Artifacts Quickstart

The easiest way to log an artifact is passing a path to your data files. Remember to also specify a name and an artifact type.

```
wandb.log_artifact(file_path, name='new_artifact', type='my_dataset') 
```

This will create a new artifact in your project's workspace:

![](</images/data_model_versioning/artifacts_quickstart.png>)

### Log a new version

If you log again, we'll checksum the artifact, identify that something changed, and track the new version. If nothing changes, we don't re-upload any data or create a new version.

```
artifact = wandb.Artifact('new_artifact', type='my_dataset')
artifact.add_dir('nature_100/')
run.log_artifact(artifact)
```

![In your Artifact page, click on the Compare button to see a new folder appears in the new version](</images/data_model_versioning/artifacts_page_compare.png>)

### Use your artifact

In a separate run, you can retrieve and download a specific version of an artifact to a local path:

```
artifact = run.use_artifact('user_name/project_name/new_artifact:v1', type='my_dataset')
artifact_dir = artifact.download()
```

### [![](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/artifacts-quickstart)

Looking for a longer example with real model training? Try our [Guide to W&B Artifacts](https://wandb.ai/wandb/arttest/reports/Guide-to-W-B-Artifacts--VmlldzozNTAzMDM).

![](</images/data_model_versioning/keras_example.png>)

## How it works

Using our Artifacts API, you can log artifacts as outputs of W&B runs, or use artifacts as input to runs.

![](</images/data_model_versioning/simple_artifact_diagram.png>)

Since a run can use another run’s output artifact as input, artifacts and runs together form a directed graph. You don’t need to define pipelines ahead of time. Just use and log artifacts, and we’ll stitch everything together.

Here's an [example artifact](https://app.wandb.ai/shawn/detectron2-11/artifacts/model/run-1cxg5qfx-model/4a0e3a7c5bff65ff4f91/graph) where you can see the summary view of the DAG, as well as the zoomed-out view of every execution of each step and every artifact version.


## Artifacts resources

Learn more about using artifacts for data and model versioning:

1. [Dataset Versioning](dataset-versioning.md)
2. [Model Versioning](model-versioning.md)

## Video tutorial for W&B Artifacts

Follow along with our [tutorial video](http://wandb.me/artifacts-video) and [interactive colab](http://wandb.me/artifacts-colab) and learn how to track your machine learning pipeline with W&B Artifacts.

<!-- {% embed url="http://wandb.me/artifacts-video" %} -->
