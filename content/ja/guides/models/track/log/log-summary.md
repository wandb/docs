---
title: サマリー メトリクスをログする
menu:
  default:
    identifier: log-summary
    parent: log-objects-and-media
---

トレーニング中に変化する値とは別に、モデルや前処理ステップを要約する単一の値を記録することも重要です。この情報は W&B Run の `summary` 辞書にログします。Run の summary 辞書は numpy 配列、PyTorch テンソル、または TensorFlow テンソルを扱うことができます。これらの型の値が指定された場合、テンソル全体をバイナリファイルとして保存し、min、mean、variance、パーセンタイルなどの高レベルメトリクスが summary オブジェクトに記録されます。

`wandb.Run.log()` で最後にログした値が、自動的に W&B Run の summary 辞書として設定されます。summary メトリクス辞書を変更した場合、以前の値は失われます。

以下のコードスニペットは、カスタム summary メトリクスを W&B に記録する方法を示しています。

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

トレーニング終了後でも、既存の W&B Run の summary 属性を更新できます。[W&B Public API]({{< relref "/ref/python/public-api/" >}}) を使用して summary 属性を編集できます。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary メトリクスをカスタマイズ

カスタム summary メトリクスを使うことで、`run.summary` にトレーニングで最も良いステップのモデルパフォーマンスを記録できます。たとえば、最終値ではなく最大 accuracy や最小 loss 値を記録したい場合などに便利です。

デフォルトでは、summary には履歴の最終値が使われます。summary メトリクスをカスタマイズするには、`define_metric` で `summary` 引数を指定します。指定できる値は以下のとおりです。

* `"min"`
* `"max"`
* `"mean"`
* `"best"`
* `"last"`
* `"none"`

`"best"` を使う場合は、オプションの `objective` 引数を `"minimize"` または `"maximize"` に設定する必要があります。

以下の例は、loss と accuracy の min と max の値を summary に追加しています。

```python
import wandb
import random

random.seed(1)

with wandb.init() as run:
    # loss の最小値と最大値を summary に追加
    run.define_metric("loss", summary="min")
    run.define_metric("loss", summary="max")

    # accuracy の最小値と最大値を summary に追加
    run.define_metric("acc", summary="min")
    run.define_metric("acc", summary="max")

    for i in range(10):
        log_dict = {
            "loss": random.uniform(0, 1 / (i + 1)),
            "acc": random.uniform(1 / (i + 1), 1),
        }
        run.log(log_dict)
```

## summary メトリクスの表示

run の **Overview** ページ、またはプロジェクトの runs テーブルで summary 値を見ることができます。

{{< tabpane text=true >}}
{{% tab header="Run Overview" value="overview" %}}

1. W&B App にアクセスします。
2. **Workspace** タブを選択します。
3. run 一覧から、summary 値を記録した run の名前をクリックします。
4. **Overview** タブを選択します。
5. **Summary** セクションで summary 値を確認できます。

{{< img src="/images/track/customize_summary.png" alt="Run overview" >}}

{{% /tab %}}
{{% tab header="Run Table" value="run table" %}}

1. W&B App にアクセスします。
2. **Runs** タブを選択します。
3. runs テーブルでは、summary 値が summary 名に対応したカラムで表示されます。

{{% /tab %}}

{{% tab header="W&B Public API" value="api" %}}

W&B Public API を使って run の summary 値を取得することもできます。

以下のコード例は、W&B Public API および pandas を使って特定の run の summary 値を取得する方法を示しています。

```python
import wandb
import pandas

entity = "<your-entity>"
project = "<your-project>"
run_name = "<your-run-name>" # summary 値を記録した run の名前

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

# run というカラム名で該当行を取得し、辞書型に変換
df[df['name']==run_name].summary.reset_index(drop=True).to_dict()
```

{{% /tab %}}
{{< /tabpane >}}