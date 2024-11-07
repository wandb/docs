---
title: "How can I find the artifacts logged or consumed by a run? How can I find the runs that produced or consumed an artifact?"
displayed_sidebar: support
tags:
   - artifacts
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

W&B tracks artifacts logged by each run and those used by each run to construct an artifact graph. This graph is a bipartite, directed, acyclic graph with nodes representing runs and artifacts. An example can be viewed [here](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph) (click "Explode" to expand the graph).

Use the Public API to navigate the graph programmatically, starting from either an artifact or a run.

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