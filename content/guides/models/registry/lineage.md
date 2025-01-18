---
description: Create a lineage map in the W&B Registry.
menu:
  default:
    parent: registry
    identifier: lineage
title: Create and view lineage maps
weight: 8
---

Within a collection in the W&B Registry, you can view a history of the artifacts that were created and used throughout an ML experiment. This history is called a _lineage graph_.

{{% alert %}}
You can also view lineage graphs for artifacts you log to W&B that are not part of a collection.
{{% /alert %}}

<!-- Lineage graphs can show the run that logged an artifact as well as artifacts used by a specific run. -->

Lineage graphs can show the specific run that logs an artifact. In addition, lineage graphs can also show which run used an artifact as an input. In other words, lineage graphs can show the input and output of a run. 


For example, the proceeding image shows artifacts created and used throughout an ML experiment:

{{< img src="/images/registry/registry_lineage.png" alt="" >}}

From left to right, the image shows.
1. Multiple runs log the "split_zoo_dataset:v4" artifact.
2. The "rural-feather-20" run uses the "split_zoo_dataset:v4" artifact for training.
3. The output of the "rural-feather-20" run is a model artifact called "zoo-ylbchv20:v0".
4. A run called "northern-lake-21" uses the model artifact "zoo-ylbchv20:v0" to evaluate the model.


## Track the input of a run

Declare an artifact as an input to a W&B run with the `wandb.init.use_artifact` API. The `wandb.init.use_artifact` API marks the artifact as a dependency of the run.

The proceeding code snippet shows how to use the `use_artifact`. Ensure to replace values enclosed in angle brackets (`< >`) with your values:

```python
import wandb

# Initialize a run
run = wandb.init(project="<project>", entity="<entity>")

# Get artifact, mark it as a dependency
artifact = run.use_artifact(artifact_or_name="<name>", aliases="<alias>")
```

Once you have retrieved your artifact, you can use that artifact to (for example), evaluate the performance of a model. 


## Track the output of a run


Track the output of a run by logging an artifact with the `wandb.init.log` API. The `wandb.init.log` API logs the artifact as an output of the run.

Use ([`wandb.init.log_artifact`]({{< relref "/ref/python/run.md#log_artifact" >}})) to declare an artifact as an output of a run.

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

For more information on how to create, construct artifacts, see [Create an artifact]({{< relref "guides/core/artifacts/construct-an-artifact.md" >}}).


## View lineage graphs in a collection

View the lineage of an artifact linked to a collection in the W&B Registry.

1. Navigate to the W&B Registry.
2. Select the collection that contains the artifact.
3. From the dropdown, click the artifact version you want to view its lineage graph.
4. Select the "Lineage" tab.


Once you are in an artifact's lineage graph page, you can view additional information about any node in that lineage graph. 
 
If you select a run node, you can view the run's ID, the run's name, the run's state, and more. As an example, the proceeding image shows information about the "rural-feather-20" run:

{{< img src="/images/registry/lineage_expanded_node.png" alt="" >}}

If you select an artifact node, you can view that artifact's full name, the artifact's type, aliases associated with the artifact, when the artifact was created, and more.

{{< img src="/images/registry/lineage_expanded_artifact_node.png" alt="" >}}