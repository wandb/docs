---
title: >-
  run によってログされた、または使用された Artifacts をどのように見つけることができますか？Artifacts を生成または使用した run
  をどのように見つけられますか？
menu:
  support:
    identifier: ja-support-kb-articles-find_artifacts_logged_consumed_run_find
support:
  - artifacts
toc_hide: true
type: docs
url: /ja/support/:filename
---
W&B は、各 run によってログされた Artifacts と、artifact graph を構築するために各 run で使用された Artifacts を追跡します。このグラフは、run と Artifacts を表すノードを持つ二部グラフで、有向非巡回グラフです。例は [こちら](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deebdd5/graph) で見ることができます（グラフを展開するには「Explode」をクリックしてください）。

Public API を使用して、Artifacts または run からプログラム的にグラフをナビゲートします。

{{< tabpane text=true >}}
{{% tab "Artifacts から" %}}

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# アーティファクトからグラフを上方向にたどる:
producer_run = artifact.logged_by()
# アーティファクトからグラフを下方向にたどる:
consumer_runs = artifact.used_by()

# run からグラフを下方向にたどる:
next_artifacts = consumer_runs[0].logged_artifacts()
# run からグラフを上方向にたどる:
previous_artifacts = producer_run.used_artifacts()
```

{{% /tab %}}
{{% tab "Run から" %}}

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# run からグラフを下方向にたどる:
produced_artifacts = run.logged_artifacts()
# run からグラフを上方向にたどる:
consumed_artifacts = run.used_artifacts()

# アーティファクトからグラフを上方向にたどる:
earlier_run = consumed_artifacts[0].logged_by()
# アーティファクトからグラフを下方向にたどる:
consumer_runs = produced_artifacts[0].used_by()
```

{{% /tab %}}
{{% /tabpane %}}