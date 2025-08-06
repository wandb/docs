---
title: サマリーメトリクスをログする
menu:
  default:
    identifier: ja-guides-models-track-log-log-summary
    parent: log-objects-and-media
---

トレーニング中に時間とともに変化する値だけでなく、モデルや前処理ステップを要約するシングルな値を記録することも重要です。この情報は W&B Run の `summary` 辞書に記録できます。Run の summary 辞書は numpy 配列、PyTorch テンソル、TensorFlow テンソルなどを扱うことができます。もし値がこれらの型の場合、テンソル全体はバイナリファイルとして保存され、高レベルなメトリクス（min、mean、variance、パーセンタイル等）が summary オブジェクトに記録されます。

`wandb.Run.log()` で最後に記録した値は自動的に W&B Run の summary 辞書として設定されます。summary メトリクスの辞書が修正された場合、以前の値は失われます。

以下のコードスニペットは、W&B にカスタム summary メトリクスを登録する方法を示しています:

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

トレーニングが終わった後でも、既存の W&B Run の summary 属性を更新できます。[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使って summary 属性を更新してください。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary メトリクスのカスタマイズ

カスタム summary メトリクスは、`run.summary` でトレーニング中のベストのモデル性能などを記録したい場合に便利です。例えば最終値ではなく最大の accuracy や最小の loss を記録したい場面などに有効です。

デフォルトでは、summary には履歴から最終値が使われます。summary メトリクスをカスタマイズするには、`define_metric` の `summary` 引数を指定します。指定できる値は次の通りです。

* `"min"`
* `"max"`
* `"mean"`
* `"best"`
* `"last"`
* `"none"`

`"best"` を使う場合は、オプションの `objective` 引数を `"minimize"` もしくは `"maximize"` に設定する必要があります。

以下の例では、loss と accuracy の min/max 値を summary に追加します。

```python
import wandb
import random

random.seed(1)

with wandb.init() as run:
    # loss の min/max summary 値
    run.define_metric("loss", summary="min")
    run.define_metric("loss", summary="max")

    # accuracy の min/max summary 値
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

run の **Overview** ページやプロジェクトの runs テーブルで summary 値を確認できます。

{{< tabpane text=true >}}
{{% tab header="Run Overview" value="overview" %}}

1. W&B App にアクセスします。
2. **Workspace** タブを選択します。
3. runs の一覧から、summary 値を記録している run 名をクリックします。
4. **Overview** タブを選択します。
5. **Summary** セクションで summary 値を確認できます。

{{< img src="/images/track/customize_summary.png" alt="Run overview" >}}

{{% /tab %}}
{{% tab header="Run Table" value="run table" %}}

1. W&B App にアクセスします。
2. **Runs** タブを選択します。
3. runs テーブル内で、summary 値を各カラム名ごとに確認できます。

{{% /tab %}}

{{% tab header="W&B Public API" value="api" %}}

W&B Public API を使って run の summary 値を取得することも可能です。

以下のコード例は、W&B Public API と pandas を使って特定の run に記録された summary 値を取得する方法の一例です。

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
  
# DataFrame へ変換  
df = pd.DataFrame(all_runs)

# カラム名（run）で行を取得し、辞書へ変換
df[df['name']==run_name].summary.reset_index(drop=True).to_dict()
```

{{% /tab %}}
{{< /tabpane >}}