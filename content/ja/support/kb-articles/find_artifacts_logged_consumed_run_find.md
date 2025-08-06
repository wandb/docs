---
title: run がログしたまたは利用したアーティファクトを見つけるにはどうすればよいですか？また、あるアーティファクトを生成または利用した run を見つけるにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
---

W&B は、各 Run によってログされた Artifacts と、各 Run で使用された Artifacts を追跡し、アーティファクトグラフを構築します。このグラフは二部グラフで、有向非巡回グラフ（DAG）となっており、ノードは Runs と Artifacts を表します。例は[こちら](https://wandb.ai/shawn/detectron2-11/artifacts/dataset/furniture-small-val/06d5ddd4deeb2a6ebdd5/graph)（「Explode」をクリックするとグラフが展開されます）。

Public API を使えば、アーティファクトまたは Run のどちらかからプログラムによってこのグラフを辿ることができます。

{{< tabpane text=true >}}
{{% tab "アーティファクトから辿る" %}}

```python
api = wandb.Api()

artifact = api.artifact("project/artifact:alias")

# アーティファクトからグラフを上方向に辿る:
producer_run = artifact.logged_by()
# アーティファクトからグラフを下方向に辿る:
consumer_runs = artifact.used_by()

# Run からグラフを下方向に辿る:
next_artifacts = consumer_runs[0].logged_artifacts()
# Run からグラフを上方向に辿る:
previous_artifacts = producer_run.used_artifacts()
```

{{% /tab %}}
{{% tab "Run から辿る" %}}

```python
api = wandb.Api()

run = api.run("entity/project/run_id")

# Run からグラフを下方向に辿る:
produced_artifacts = run.logged_artifacts()
# Run からグラフを上方向に辿る:
consumed_artifacts = run.used_artifacts()

# アーティファクトからグラフを上方向に辿る:
earlier_run = consumed_artifacts[0].logged_by()
# アーティファクトからグラフを下方向に辿る:
consumer_runs = produced_artifacts[0].used_by()
```

{{% /tab %}}
{{% /tabpane %}}