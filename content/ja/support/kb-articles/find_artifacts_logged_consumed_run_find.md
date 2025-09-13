---
title: run でログされた、または消費された Artifacts を見つけるにはどうすればよいですか？ Artifact を生成または消費した runs を見つけるにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-find_artifacts_logged_consumed_run_find
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

W&B は、各 Run がログした Artifact と各 Run が使用した Artifact を追跡し、Artifact グラフを構築します。このグラフは、ノードが Run と Artifact を表す、二部・有向・非巡回グラフです。サンプルは [こちら](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph) で確認できます (グラフを展開するには "Explode" をクリック)。

Public API を使って、Artifact または Run を起点にプログラムからグラフをたどれます。

{{< tabpane text=true >}}
{{% tab "Artifact から" %}}

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# Artifact からグラフを上流にたどる:
producer_run = artifact.logged_by()
# Artifact からグラフを下流にたどる:
consumer_runs = artifact.used_by()

# Run からグラフを下流にたどる:
next_artifacts = consumer_runs[0].logged_artifacts()
# Run からグラフを上流にたどる:
previous_artifacts = producer_run.used_artifacts()
```

{{% /tab %}}
{{% tab "Run から" %}}

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# Run からグラフを下流にたどる:
produced_artifacts = run.logged_artifacts()
# Run からグラフを上流にたどる:
consumed_artifacts = run.used_artifacts()

# Artifact からグラフを上流にたどる:
earlier_run = consumed_artifacts[0].logged_by()
# Artifact からグラフを下流にたどる:
consumer_runs = produced_artifacts[0].used_by()
```

{{% /tab %}}
{{% /tabpane %}}