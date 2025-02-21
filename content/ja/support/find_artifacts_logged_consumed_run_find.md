---
title: How can I find the artifacts logged or consumed by a run? How can I find the
  runs that produced or consumed an artifact?
menu:
  support:
    identifier: ja-support-find_artifacts_logged_consumed_run_find
tags:
- artifacts
toc_hide: true
type: docs
---

W&B は、各 run によって記録されたアーティファクトと、各 run が使用するアーティファクトを追跡し、アーティファクトのグラフを構築します。このグラフは、run とアーティファクトを表すノードを持つ二部、方向性、非巡回グラフです。例は [こちら](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph) で見ることができます（「Explode」をクリックするとグラフが展開されます）。

Public API を使用して、アーティファクトまたは run からプログラム的にグラフをナビゲートします。

{{< tabpane text=true >}}
{{% tab "From an Artifact" %}}

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# アーティファクトからグラフを上に移動:
producer_run = artifact.logged_by()
# アーティファクトからグラフを下に移動:
consumer_runs = artifact.used_by()

# run からグラフを下に移動:
next_artifacts = consumer_runs[0].logged_artifacts()
# run からグラフを上に移動:
previous_artifacts = producer_run.used_artifacts()
```

{{% /tab %}}
{{% tab "From a Run" %}}

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# run からグラフを下に移動:
produced_artifacts = run.logged_artifacts()
# run からグラフを上に移動:
consumed_artifacts = run.used_artifacts()

# アーティファクトからグラフを上に移動:
earlier_run = consumed_artifacts[0].logged_by()
# アーティファクトからグラフを下に移動:
consumer_runs = produced_artifacts[0].used_by()
```

{{% /tab %}}
{{% /tabpane %}}