---
description: Create and view lineage maps, audit a collection's history
menu:
  default:
    parent: registry
    identifier: lineage
title: Lineage maps and audit history
weight: 8
---

Use lineage maps to visualize a linked artifact's history. Audit a collection's history to track changes made to artifacts in that collection.

## Lineage maps

Within a collection in the W&B Registry, you can view a history of the artifacts that an ML experiment uses. This history is called a _lineage graph_.

{{% alert color="info" %}}
You can also view lineage graphs for artifacts you log to W&B that are not part of a collection.
{{% /alert %}}

Lineage graphs show:
* If a run used an artifact as an input.
* If a run created an artifact as an output.

In other words, lineage graphs show the input and output of a run. 


For example, the following image shows a typical lineage graph for artifacts created and used throughout an ML experiment:

{{< img src="/images/registry/registry_lineage.png" alt="Registry lineage" >}}

From left to right, the image shows:
1. Multiple runs log the `split_zoo_dataset:v4` artifact.
2. The "rural-feather-20" run uses the `split_zoo_dataset:v4` artifact for training.
3. The output of the "rural-feather-20" run is a model artifact called `zoo-ylbchv20:v0`.
4. A run called "northern-lake-21" uses the model artifact `zoo-ylbchv20:v0` to evaluate the model.


### Track the input of a run

Mark an artifact as the input (or dependency) of a run with the [`wandb.Run.use_artifact()`]({{< relref "/ref/python/experiments/run/#method-runuse_artifact" >}}) method. Specify the name of the artifact and an optional alias to reference a specific version of that artifact. The name of the artifact is in the format `<artifact_name>:<version>` or `<artifact_name>:<alias>`.

Replace values enclosed in angle brackets (`< >`) with your values:

```python
import wandb

# Initialize a run
with wandb.init(entity="<entity>", project="<project>") as run:
  # Get artifact, mark it as a dependency
  artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```


### Track the output of a run

Use [`wandb.Run.log_artifact()`]({{< relref "/ref/python/experiments/run.md#log_artifact" >}}) to declare an artifact as an output of a run. First, create an artifact with the [`wandb.Artifact()`]({{< relref "/ref/python/experiments/artifact/#wandb.Artifact" >}}) constructor. Then, log the artifact as an output of the run with `wandb.Run.log_artifact()`.

Replace values enclosed in angle brackets (`< >`) with your values:

```python
import wandb

# Initialize a run
with wandb.init(entity="<entity>", project="<project>") as run:
  
  # Create an artifact
  artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
  artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

  # Log the artifact as an output of the run
  run.log_artifact(artifact_or_path = artifact)
```

For more information on about creating artifacts, see [Create an artifact]({{< relref "guides/core/artifacts/construct-an-artifact.md" >}}).


### View lineage graphs in a collection

View the lineage of an artifact linked to a collection in the W&B Registry.

1. Navigate to the W&B Registry.
2. Select the collection that contains the artifact.
3. From the dropdown, select the artifact version you want to view its lineage graph.
4. Select the "Lineage" tab.
5. Select a node to view detailed information about the run or artifact.


The following image shows the expanded detailed view of a run (`rural-feather-20`) when you select a node in the lineage graph:

{{< img src="/images/registry/lineage_expanded_node.png" alt="Expanded lineage node" >}}

The following image shows the expanded detailed view of an artifact (`zoo-ylbchv20:v0`) when you select an artifact node in the lineage graph:

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="Expanded artifact node details" >}}

## Audit a collection's history

View actions that members of your organization take on that collection. You can view:

- If an alias was added or removed from an artifact version.
- If an artifact version was added or removed from a collection.

For both actions, you can view the user that performed the action and the date the action occurred.

To view a collection's action history:

1. Navigate to the W&B Registry.
2. Select the collection you want to view its action history.
3. Select the dropdown menu next to the collection name.
4. Select the **Action History** option.