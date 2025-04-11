---
title: ログサマリーメトリクス
menu:
  default:
    identifier: ja-guides-models-track-log-log-summary
    parent: log-objects-and-media
---

時間とともに変化する値に加えて、モデルや前処理ステップを要約する単一の値を追跡することも重要です。この情報を W&B Run の `summary` 辞書にログします。Run の summary 辞書は numpy 配列、PyTorch テンソル、TensorFlow テンソルを扱うことができます。値がこれらのタイプのいずれかの場合、バイナリファイルにテンソル全体を保存し、メトリクスを summary オブジェクトに保存します。たとえば最小値、平均、分散、パーセンタイルなどです。

最後に `wandb.log` でログされた値は、自動的に W&B Run の summary 辞書に設定されます。summary メトリクス辞書が変更されると、以前の値は失われます。

次のコードスニペットは、W&B にカスタムの summary メトリクスを提供する方法を示しています。
```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

トレーニングが完了した後、既存の W&B Run の summary 属性を更新することができます。[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使用して、summary 属性を更新してください。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary メトリクスをカスタマイズする

カスタム summary メトリクスは、トレーニングにおける最良のステップでのモデルのパフォーマンスを `wandb.summary` にキャプチャするのに便利です。たとえば、最終的な値の代わりに、最大精度や最小損失値をキャプチャしたいかもしれません。

デフォルトでは、summary は履歴からの最終的な値を使用します。summary メトリクスをカスタマイズするには、`define_metric` の中に `summary` 引数を渡します。以下の値を受け付けます。

* `"min"`
* `"max"`
* `"mean"`
* `"best"`
* `"last"`
* `"none"`

`"best"` を使用するには、任意の `objective` 引数を `"minimize"` または `"maximize"` に設定する必要があります。

次の例は、損失と精度の最小値と最大値を summary に追加する方法を示しています。

```python
import wandb
import random

random.seed(1)
wandb.init()

# 損失の最小値および最大値を summary に追加
wandb.define_metric("loss", summary="min")
wandb.define_metric("loss", summary="max")

# 精度の最小値および最大値を summary に追加
wandb.define_metric("acc", summary="min")
wandb.define_metric("acc", summary="max")

for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

## summary メトリクスを閲覧する

Run の **Overview** ページまたはプロジェクトの runs テーブルで summary 値を表示することができます。

{{< tabpane text=true >}}
{{% tab header="Run Overview" value="overview" %}}

1. W&B アプリに移動します。
2. **Workspace** タブを選択します。
3. runs のリストから、summary 値をログした run の名前をクリックします。
4. **Overview** タブを選択します。
5. **Summary** セクションで summary 値を表示します。

{{< img src="/images/track/customize_summary.png" alt="W&Bにログされた run の概要ページ。UIの右下隅には summary メトリクスセクション内の機械学習モデルの精度と損失の最小値と最大値が表示されています。" >}}

{{% /tab %}}
{{% tab header="Run Table" value="run table" %}}

1. W&B アプリに移動します。
2. **Runs** タブを選択します。
3. runs テーブル内で、summary 値の名前に基づいて列内の summary 値を表示することができます。

{{% /tab %}}

{{% tab header="W&B Public API" value="api" %}}

W&B Public API を使用して、run の summary 値を取得することができます。

次のコード例は、W&B Public API と pandas を使用して特定の run にログされた summary 値を取得する方法の一例を示しています。

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
  
# DataFrame に変換  
df = pd.DataFrame(all_runs)

# 列名（run）に基づいて行を取得し、辞書に変換
df[df['name']==run_name].summary.reset_index(drop=True).to_dict()
```

{{% /tab %}}
{{< /tabpane >}}