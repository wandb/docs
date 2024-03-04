---
description: Traverse automatically created direct acyclic W&B Artifact graphs.
displayed_sidebar: default
---

# Explore and traverse artifact graphs

<head>
    <title>Explore direct acyclic W&B Artifact graphs.</title>
</head>

W&B automatically tracks the artifacts a given run logged as well as the artifacts a given run used. Explore the lineage of an artifact with the W&B App UI or programmatically.


## Lineage

When selecting an artifact in the **Artifacts** tab, you will be able to see your artifact's lineage. This graph view shows a general overview of your pipeline. 

To view an artifact graph:

1. Navigate to your project in the W&B App UI
2. Choose the artifact icon on the left panel.
3. Select **Lineage**.

### Navigating lineage 
The lineage graph is generated based on the `type` you provide when you create runs and artifacts. 

The artifact or run type you provide is located above its name, withA artifacts represented by blue icona and runs represented by green icons. The input and output of a run or artifact is depicted in the graph with arrows. 

:::info
You can view the type and the name of artifact in both the left sidebar and in the **Lineage** tab. 
:::


For a more detailed view, click on the arrow on any individual artifact or run to get more information on a particular object.

### Artifact clusters

When a level of the graph has five or more runs or artifacts, a cluster will be created. A cluster has a search bar to find specific versions of runs or artifact and an individual node can be pulled from a cluster to continue investigating the lineage of a node inside a cluster. Clicking on a node will open a preview drawer with an overview of the node.

## Traverse a graph programmatically 
You can also navigate a graph usuing the [W&B API]((../../ref/python/public-api/api.md)). 

### Traverse from an artifact

Create an artifact object, and provide the name of the project, artifact and alias of the artifact:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("project/artifact:alias")
```

Use the artifact object's [`logged_by`](../../ref/python/artifact.md#logged_by) and [`used_by`](../../ref/python/artifact.md#used_by) methods to walk the graph from the artifact:

```python
# Walk up and down the graph from an artifact:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```

### Traverse from a run

Create an artifact object with the W&B Public API ([wandb.Api.Run](../../ref/python/public-api/run.md)). Provide the name of the entity, project, and run ID:

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
```

Use the [`logged_artifacts`](../../ref/python/public-api/run.md#logged_artifacts) and [`used_artifacts`](../../ref/python/public-api/run.md#used_artifacts) methods to walk the graph from a given run:

```python
# Walk up and down the graph from a run:
logged_artifacts = run.logged_artifacts()
used_artifacts = run.used_artifacts()
```
