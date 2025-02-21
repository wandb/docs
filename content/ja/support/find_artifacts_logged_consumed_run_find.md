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

W&B は、各 run によってログに記録された Artifacts と、各 run で使用された Artifacts を追跡して、アーティファクトグラフを構築します。このグラフは、run とアーティファクトを表すノードを持つ、二部グラフ、有向非巡回グラフです。例は[こちら](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph)で確認できます（グラフを展開するには「Explode」をクリックしてください）。

Public API を使用して、アーティファクトまたは run のいずれかから開始して、グラフをプログラムでナビゲートします。

{{< tabpane text=true >}}
{{% tab "From an Artifact" %}}

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# アーティファクトからグラフを上にたどる:
producer_run = artifact.logged_by()
# アーティファクトからグラフを下にたどる:
consumer_runs = artifact.used_by()

# run からグラフを下にたどる:
next_artifacts = consumer_runs[0].logged_artifacts()
# run からグラフを上にたどる:
previous_artifacts = producer_run.used_artifacts()
```

{{% /tab %}}
{{% tab "From a Run" %}}

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# run からグラフを下にたどる:
produced_artifacts = run.logged_artifacts()
# run からグラフを上にたどる:
consumed_artifacts = run.used_artifacts()

# アーティファクトからグラフを上にたどる:
earlier_run = consumed_artifacts[0].logged_by()
# アーティファクトからグラフを下にたどる:
consumer_runs = produced_artifacts[0].used_by()
```

{{% /tab %}}
{{% /tabpane %}}
