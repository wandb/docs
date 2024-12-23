---
title: "How can I find the artifacts logged or consumed by a run? How can I find the runs that produced or consumed an artifact?"
toc_hide: true
type: docs
tags:
   - artifacts
---
W&B tracks artifacts logged by each run and those used by each run to construct an artifact graph. This graph is a bipartite, directed, acyclic graph with nodes representing runs and artifacts. An example can be viewed [here](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph) (click "Explode" to expand the graph).

Use the Public API to navigate the graph programmatically, starting from either an artifact or a run.

{{< tabpane text=true >}}
{{% tab "From an Artifact" %}}

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

{{% /tab %}}
{{% tab "From a Run" %}}

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

{{% /tab %}}
{{% /tabpane %}}