---
title: ある run がログしたり利用した Artifacts をどうやって見つけることができますか？また、ある Artifact を作成または利用した run
  をどうやって見つけることができますか？
menu:
  support:
    identifier: ja-support-kb-articles-find_artifacts_logged_consumed_run_find
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

W&B は、各 run によってログされた Artifacts と、各 run で使用された Artifacts を追跡し、アーティファクトグラフを構築します。このグラフは二部、かつ有向非循環グラフで、ノードは run と Artifacts を表します。実際の例は [こちら](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph) で確認できます（「Explode」をクリックしてグラフを展開してください）。

Public API を使うことで、プログラムからこのグラフを辿ることができます。始点は Artifact か run のどちらからでも構いません。

{{< tabpane text=true >}}
{{% tab "Artifact からたどる場合" %}}

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# Artifact からグラフを上方向にたどる:
producer_run = artifact.logged_by()
# Artifact からグラフを下方向にたどる:
consumer_runs = artifact.used_by()

# run からグラフを下方向にたどる:
next_artifacts = consumer_runs[0].logged_artifacts()
# run からグラフを上方向にたどる:
previous_artifacts = producer_run.used_artifacts()
```

{{% /tab %}}
{{% tab "Run からたどる場合" %}}

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# run からグラフを下方向にたどる:
produced_artifacts = run.logged_artifacts()
# run からグラフを上方向にたどる:
consumed_artifacts = run.used_artifacts()

# Artifact からグラフを上方向にたどる:
earlier_run = consumed_artifacts[0].logged_by()
# Artifact からグラフを下方向にたどる:
consumer_runs = produced_artifacts[0].used_by()
```

{{% /tab %}}
{{% /tabpane %}}