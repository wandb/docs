---
description: Create a lineage map in the W&B Registry.
menu:
  default:
    parent: registry
    identifier: lineage
title: Create and view lineage maps
weight: 8
---

Within a collection in the W&B Registry, you can view a history of the artifacts that an ML experiment uses. This history is called a _lineage graph_.

{{% pageinfo color="info" %}}
You can also view lineage graphs for artifacts you log to W&B that are not part of a collection.
{{% /pageinfo %}}

Lineage graphs can show the specific run that logs an artifact. In addition, lineage graphs can also show which run used an artifact as an input. In other words, lineage graphs can show the input and output of a run. 


For example, the proceeding image shows artifacts created and used throughout an ML experiment:

{{< img src="/images/registry/registry_lineage.png" alt="Registry lineage" >}}

From left to right, the image shows:
1. Multiple runs log the `split_zoo_dataset:v4` artifact.
2. The "rural-feather-20" run uses the `split_zoo_dataset:v4` artifact for training.
3. The output of the "rural-feather-20" run is a model artifact called `zoo-ylbchv20:v0`.
4. A run called "northern-lake-21" uses the model artifact `zoo-ylbchv20:v0` to evaluate the model.


## Track the input of a run

Mark an artifact as an input or dependency of a run with the `wandb.init.use_artifact` API.

The proceeding code snippet shows how to use the `use_artifact`. Replace values enclosed in angle brackets (`< >`) with your values:

```python
import wandb

# Initialize a run
run = wandb.init(project="<project>", entity="<entity>")

# Get artifact, mark it as a dependency
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```


## Track the output of a run

Use ([`wandb.init.log_artifact`]({{< relref "/ref/python/sdk/classes/run#log_artifact" >}})) to declare an artifact as an output of a run.

The proceeding code snippet shows how to use the `wandb.init.log_artifact` API. Ensure to replace values enclosed in angle brackets (`< >`) with your values:

```python
import wandb

# Initialize a run
run = wandb.init(entity  "<entity>", project = "<project>",)
artifact = wandb.Artifact(name = "<artifact_name>", type = "<artifact_type>")
artifact.add_file(local_path = "<local_filepath>", name="<optional-name>")

# Log the artifact as an output of the run
run.log_artifact(artifact_or_path = artifact)
```

For more information on about creating artifacts, see [Create an artifact]({{< relref "guides/core/artifacts/construct-an-artifact.md" >}}).


## View lineage graphs in a collection

View the lineage of an artifact linked to a collection in the W&B Registry.

1. Navigate to the W&B Registry.
2. Select the collection that contains the artifact.
3. From the dropdown, click the artifact version you want to view its lineage graph.
4. Select the "Lineage" tab.


Once you are in an artifact's lineage graph page, you can view additional information about any node in that lineage graph. 
 
Select a run node to view that run's details, such as the run's ID, the run's name, the run's state, and more. As an example, the proceeding image shows information about the `rural-feather-20` run:

{{< img src="/images/registry/lineage_expanded_node.png" alt="Expanded lineage node" >}}

Select an artifact node to view that artifact's details, such as its full name, type, creation time, and associated aliases.

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="Expanded artifact node details" >}}