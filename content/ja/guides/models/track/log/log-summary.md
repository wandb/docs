---
title: Log summary metrics
menu:
  default:
    identifier: ja-guides-models-track-log-log-summary
    parent: log-objects-and-media
---

トレーニング中に時間とともに変化する値に加えて、モデルや前処理ステップを要約する単一の値を追跡することも重要です。この情報を W&B の Run の `summary` 辞書に記録します。Run の summary 辞書は、numpy 配列、PyTorch テンソル、または TensorFlow テンソルを処理できます。値がこれらの型のいずれかである場合、バイナリファイルにテンソル全体を保持し、min、平均、分散、パーセンタイルなどの高度な メトリクス を summary オブジェクトに格納します。

`wandb.log` で最後に記録された値は、W&B Run の summary 辞書として自動的に設定されます。summary メトリクス 辞書が変更されると、前の値は失われます。

次の コードスニペット は、カスタム summary メトリクス を W&B に提供する方法を示しています。
```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

トレーニングが完了した後、既存の W&B Run の summary 属性を更新できます。[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使用して、summary 属性を更新します。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary メトリクス のカスタマイズ

カスタム summary メトリクス は、`wandb.summary` でトレーニングの最適なステップでモデルのパフォーマンスをキャプチャするのに役立ちます。たとえば、最終値の代わりに、最大精度または最小損失値をキャプチャしたい場合があります。

デフォルトでは、summary は履歴からの最終値を使用します。summary メトリクス をカスタマイズするには、`define_metric` で `summary` 引数に渡します。これは、次の値を受け入れます。

* `"min"`
* `"max"`
* `"mean"`
* `"best"`
* `"last"`
* `"none"`

`"best"` は、オプションの `objective` 引数を `"minimize"` または `"maximize"` に設定した場合にのみ使用できます。

次の例では、損失と精度の最小値と最大値を summary に追加します。

```python
import wandb
import random

random.seed(1)
wandb.init()

# 損失の最小値と最大値のsummary値
wandb.define_metric("loss", summary="min")
wandb.define_metric("loss", summary="max")

# 精度に対する最小値と最大値のsummary値
wandb.define_metric("acc", summary="min")
wandb.define_metric("acc", summary="max")

for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

## summary メトリクス の表示

run の **Overview** ページまたは project の runs テーブルで summary 値を表示します。

{{< tabpane text=true >}}
{{% tab header="Run Overview" value="overview" %}}

1. W&B App に移動します。
2. **Workspace** タブを選択します。
3. runs のリストから、summary 値を記録した run の名前をクリックします。
4. **Overview** タブを選択します。
5. **Summary** セクションで summary 値を表示します。

{{< img src="/images/track/customize_summary.png" alt="W&B に記録された run の Overview ページ。UI の右下隅には、Summary メトリクス セクション内の機械学習モデルの精度と損失の最小値と最大値が表示されます。" >}}

{{% /tab %}}
{{% tab header="Run Table" value="run table" %}}

1. W&B App に移動します。
2. **Runs** タブを選択します。
3. runs テーブル内で、summary 値の名前に基づいて、列内の summary 値を表示できます。

{{% /tab %}}

{{% tab header="W&B Public API" value="api" %}}

W&B Public API を使用して、run の summary 値を取得できます。

次の コード例 は、W&B Public API と pandas を使用して、特定の run に記録された summary 値を取得する方法の 1 つを示しています。

```python
import wandb
import pandas

entity = "<your-entity>"
project = "<your-project>"
run_name = "<your-run-name>" # summary 値を持つ run の名前

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
  
# DataFrameに変換
df = pd.DataFrame(all_runs)

# カラム名 (run) に基づいて行を取得し、dictionary に変換します。
df[df['name']==run_name].summary.reset_index(drop=True).to_dict()
```

{{% /tab %}}
{{< /tabpane >}}
