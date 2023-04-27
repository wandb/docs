---
description: Traverse automatically created direct acyclic W&B Artifact graphs.
displayed_sidebar: ja
---

# Explore and traverse artifact graphs

<head>
    <title>Explore direct acyclic W&B Artifact graphs.</title>
</head>

Weights & Biases automatically tracks the artifacts a given run logged as well as the artifacts a given run used. Explore the lineage of an artifact with the W&B App UI or programmatically.


## Traverse an artifact with the W&B App UI

The graph view shows a general overview of your pipeline. 

To view an artifact graph:

1. Navigate to your project in the W&B App UI
2. Choose the artifact icon on the left panel.
3. Select **Lineage**.

The `type` you provide when you create runs and artifacts are used to create the graph. The input and output of a run or artifact is depicted in the graph with arrows. Artifacts are represented by blue rectangles and Runs are represented by green rectangles. 



The artifact type you provide is located in the dark blue header next to the **ARTIFACT** label. The name of the artifact, along with the artifact version, is shown in the light blue region underneath the **ARTIFACT** label.

The job type you provide when you initialized a run is located next to the **RUN** label. The W&B run name is located in the light green region underneath the **RUN** label. 

:::info
You can view the type and the name of artifacts both in the left sidebar and in the **Lineage** tab. 
:::



For example, in the proceeding image, an artifact was defined with a type called "raw_dataset" (pink square). The name of the artifact is called "MNIST_raw" (pink line). The artifact was then used for training. The name of the training run is called "vivid-snow-42". That run then produced a "model" artifact (orange square) named "mnist-19pofeku".


![DAG view of artifacts, runs used for an experiment.](/images/artifacts/example_dag_with_sidebar.png)


For a more detailed view, select the **Explode** toggle on the upper left hand side of the dashboard. The expanded graph shows details of every run and every artifact in the project that was logged. Try it yourself on this [example Graph page](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/v0/lineage).


## Traverse an artifact programmatically 

Create an artifact object with the W&B Public API ([wandb.Api](https://docs.wandb.ai/ref/python/public-api/api)). Provide the name of the project, artifact and alias of the artifact:

```python
import wandb

api = wandb.Api()

artifact = api.artifact('project/artifact:alias')
```

Use the artifact objects [`logged_by`](https://docs.wandb.ai/ref/python/public-api/artifact#logged\_by) and [`used_by`](https://docs.wandb.ai/ref/python/public-api/artifact#used\_by) methods to walk the graph from the artifact:

```python
# Walk up and down the graph from an artifact:
producer_run = artifact.logged_by()
consumer_runs = artifact.used_by()
```

#### Traverse from a run

Create an artifact object with the W&B Public API ([wandb.Api.Run](https://docs.wandb.ai/ref/python/public-api/run)). Provide the name of the entity, project, and run ID:

```python
import wandb

api = wandb.Api()

artifact = api.run('entity/project/run_id')
```

Use the [`logged_artifacts`](https://docs.wandb.ai/ref/python/public-api/run#logged\_artifacts) and [`used_artifacts`](https://docs.wandb.ai/ref/python/public-api/run#used\_artifacts) methods to walk the graph from a given run:

```python
# Walk up and down the graph from a run:
logged_artifacts = run.logged_artifacts()
used_artifacts = run.used_artifacts()
```
