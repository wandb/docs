---
title: サマリー メトリクスをログする
menu:
  default:
    identifier: ja-guides-models-track-log-log-summary
    parent: log-objects-and-media
---

トレーニング 中に時間とともに変化する値に加えて、モデル や前処理ステップを要約する単一の値を追跡することも重要です。こうした情報は W&B の Run の `summary` 辞書にログしてください。Run の summary 辞書は numpy 配列、PyTorch テンソル、または TensorFlow テンソルを扱えます。値がこれらの型のいずれかの場合、テンソル全体をバイナリファイルとして保存し、summary オブジェクトには min、mean、variance、percentiles などのハイレベルなメトリクスを保存します。

`wandb.Run.log()` で最後にログされた値は、W&B の Run の summary 辞書として自動的に設定されます。summary メトリクスの辞書を変更すると、以前の値は失われます。

以下のコードスニペットは、W&B にカスタムの summary メトリクスを渡す方法を示します。

```python
import wandb
import argparse

with wandb.init(config=args) as run:
  best_accuracy = 0
  for epoch in range(1, args.epochs + 1):
      test_loss, test_accuracy = test()
      if test_accuracy > best_accuracy:
          run.summary["best_accuracy"] = test_accuracy
          best_accuracy = test_accuracy
```

トレーニング 完了後でも、既存の W&B の Run の summary 属性を更新できます。[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使って summary 属性を更新します:

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary メトリクスをカスタマイズする

カスタム summary メトリクスは、`run.summary` にトレーニング の最良ステップでの モデル の性能を記録するのに役立ちます。例えば、最終値ではなく、最大の精度や最小の損失を記録したい場合があります。

デフォルトでは、summary は履歴の最終値を採用します。summary メトリクスをカスタマイズするには、`define_metric` に `summary` 引数を渡します。指定できる値は次のとおりです:

* `"min"`
* `"max"`
* `"mean"`
* `"best"`
* `"last"`
* `"none"`

`"best"` は、オプションの `objective` 引数を `"minimize"` または `"maximize"` に設定した場合にのみ使用できます。 

次の例では、loss と accuracy の最小値・最大値を summary に追加します。

```python
import wandb
import random

random.seed(1)

with wandb.init() as run:
    # loss の最小値・最大値を summary に記録
    run.define_metric("loss", summary="min")
    run.define_metric("loss", summary="max")

    # accuracy の最小値・最大値を summary に記録
    run.define_metric("acc", summary="min")
    run.define_metric("acc", summary="max")

    for i in range(10):
        log_dict = {
            "loss": random.uniform(0, 1 / (i + 1)),
            "acc": random.uniform(1 / (i + 1), 1),
        }
        run.log(log_dict)
```

## summary メトリクスを表示する

summary の値は、各 Run の Overview ページや Project の Runs テーブルで確認できます。

{{< tabpane text=true >}}
{{% tab header="Run Overview" value="overview" %}}

1. W&B アプリに移動します。
2. Workspace タブを選択します。
3. Runs の一覧から、summary の値をログした対象の Run 名をクリックします。
4. Overview タブを選択します。
5. Summary セクションで summary の値を確認します。

{{< img src="/images/track/customize_summary.png" alt="Run の Overview" >}}

{{% /tab %}}
{{% tab header="Run Table" value="run table" %}}

1. W&B アプリに移動します。
2. Runs タブを選択します。
3. Runs テーブルでは、summary 値の名前に対応した列に summary の値が表示されます。

{{% /tab %}}

{{% tab header="W&B Public API" value="api" %}}

W&B Public API を使って、特定の Run の summary の値を取得できます。 

以下は、W&B Public API と pandas を使って、特定の Run にログされた summary の値を取得する方法の一例です。

```python
import wandb
import pandas

entity = "<your-entity>"
project = "<your-project>"
run_name = "<your-run-name>" # summary の値を含む Run の名前

all_runs = []

for run in api.runs(f"{entity}/{project_name}"):
    print("Fetching details for run: ", run.id, run.name)
    run_data = {
              "id": run.id,
              "name": run.name,
              "url": run.url,
              "state": run.state,
              "tags": run.tags,
              "config": run.config,
              "created_at": run.created_at,
              "system_metrics": run.system_metrics,
              "summary": run.summary,
              "project": run.project,
              "entity": run.entity,
              "user": run.user,
              "path": run.path,
              "notes": run.notes,
              "read_only": run.read_only,
              "history_keys": run.history_keys,
              "metadata": run.metadata,
          }
    all_runs.append(run_data)
  
# DataFrame に変換
df = pd.DataFrame(all_runs)

# 列名（run）に基づいて行を取得し、辞書に変換
df[df['name']==run_name].summary.reset_index(drop=True).to_dict()
```

{{% /tab %}}
{{< /tabpane >}}