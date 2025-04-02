---
title: How can I find the artifacts logged or consumed by a run? How can I find the
  runs that produced or consumed an artifact?
menu:
  support:
    identifier: ja-support-kb-articles-find_artifacts_logged_consumed_run_find
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

W&B は、各 run によって記録された Artifacts と、各 run が Artifacts グラフを構築するために使用する Artifacts を追跡します。このグラフは、run と Artifacts を表すノードを持つ、二部グラフ、有向非巡回グラフです。例は[こちら](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph)で確認できます（グラフを展開するには「Explode」をクリックしてください）。

Public API を使用して、Artifacts または run のいずれかから開始して、プログラムでグラフをナビゲートします。

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
