---
description: Answers to frequently asked question about W&B Artifacts.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Artifacts FAQs

<head>
  <title>Frequently Asked Questions About Artifacts</title>
</head>

The proceeding questions are commonly asked questions about [W&B Artifacts](#questions-about-artifacts) and [W&B Artifact workflows](#questions-about-artifacts-workflows).

## Questions about Artifacts

### Who has access to my artifacts?

Artifacts inherit the access to their parent project:

* If the project is private, then only members of the project's team have access to its artifacts.
* For public projects, all users have read access to artifacts but only members of the project's team can create or modify them.
* For open projects, all users have read and write access to artifacts.

## Questions about Artifacts workflows

This section describes workflows for managing and editing Artifacts. Many of these workflows use [the W&B API](../track/public-api-guide.md), the component of [our client library](../../ref/python/README.md) which provides access to data stored with W&B.

### How do I log an artifact to an existing run?

Occasionally, you may want to mark an artifact as the output of a previously logged run. In that scenario, you can [reinitialize the old run](../runs/resuming.md) and log new artifacts to it as follows:

```python
with wandb.init(id="existing_run_id", resume="allow") as run:
    artifact = wandb.Artifact("artifact_name", "artifact_type")
    artifact.add_file("my_data/file.txt")
    run.log_artifact(artifact)
```

### How do I set a retention or expiration policy on my artifact?

If you have artifacts that are subject to data privacy regulations such as dataset artifacts containing PII, or want to schedule the deletion of an artifact version to manage your storage, you can set a TTL (time-to-live) policy. Learn more in [this](./ttl.md) guide. 

### How can I find the artifacts logged or consumed by a run? How can I find the runs that produced or consumed an artifact?

W&B automatically tracks the artifacts a given run has logged as well as the artifacts a given run has used and uses the information to construct an artifact graph -- a bipartite, directed, acyclic graph whose nodes are runs and artifacts, like [this one](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph) (click "Explode" to see the full graph).

You can walk this graph programmatically with [the Public API](../../ref/python/public-api/README.md), starting from either a run or an artifact.

<Tabs
  defaultValue="from_artifact"
  values={[
    {label: 'From an Artifact', value: 'from_artifact'},
    {label: 'From a Run', value: 'from_run'},
  ]}>
  <TabItem value="from_artifact">

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# Walk up the graph from an artifact:
producer_run = artifact.logged_by()
# Walk down the graph from an artifact:
consumer_runs = artifact.used_by()

# Walk down the graph from a run:
next_artifacts = consumer_runs[0].logged_artifacts()
# Walk up the graph from a run:
previous_artifacts = producer_run.used_artifacts()
```

  </TabItem>
  <TabItem value="from_run">

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# Walk down the graph from a run:
produced_artifacts = run.logged_artifacts()
# Walk up the graph from a run:
consumed_artifacts = run.used_artifacts()

# Walk up the graph from an artifact:
earlier_run = consumed_artifacts[0].logged_by()
# Walk down the graph from an artifact:
consumer_runs = produced_artifacts[0].used_by()
```

  </TabItem>
</Tabs>

### How do I best log models from runs in a sweep?

One effective pattern for logging models in a [sweep](../sweeps/intro.md) is to have a model artifact for the sweep, where the versions will correspond to different runs from the sweep. More concretely, you would have:

```python
wandb.Artifact(name="sweep_name", type="model")
```

### How do I find an artifact from the best run in a sweep?

You can use the following code to retrieve the artifacts associated with the best performing run in a sweep:

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```

### How do I save code?‌

Use `save_code=True` in `wandb.init` to save the main script or notebook where you’re launching the run. To save all your code to a run, version code with Artifacts. Here’s an example:

```python
code_artifact = wandb.Artifact(type="code")
code_artifact.add_file("./train.py")
wandb.log_artifact(code_artifact)
```

### Using artifacts with multiple architectures and runs?

There are many ways in which you can think of _version_ a model. Artifacts provides a you a tool to implement model versioning as you see fit. One common pattern for projects that explore multiple model architectures over a number of runs is to separate artifacts by architecture. As an example, one could do the following:

1. Create a new artifact for each different model architecture. You can use `metadata` attribute of artifacts to describe the architecture in more detail (similar to how you would use `config` for a run).
2. For each model, periodically log checkpoints with `log_artifact`. W&B will automatically build a history of those checkpoints, annotating the most recent checkpoint with the `latest` alias so you can refer to the latest checkpoint for any given model architecture using `architecture-name:latest`

## Reference Artifact FAQs


### How can I fetch these Version IDs and ETags in W&B?

If you've logged an artifact reference with W&B and if the versioning is enabled on your buckets then the version IDs can be seen in the S3 UI. To fetch these version IDs and ETags in W&B, you can fetch the artifact and then get the corresponding manifest entries. For example:

```python
artifact = run.use_artifact("my_table:latest")
for entry in artifact.manifest.entries.values():
    versionID = entry.extra.get("versionID")
    etag = manifest_entry.extra.get("etag")
```
