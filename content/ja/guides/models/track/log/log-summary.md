---
title: Log summary metrics
menu:
  default:
    identifier: ja-guides-models-track-log-log-summary
    parent: log-objects-and-media
---

トレーニング中に時間とともに変化する値に加えて、モデルや前処理のステップを要約する単一の値を追跡することも重要です。この情報を W&B の Run の `summary` 辞書に記録します。Run の summary 辞書は、NumPy 配列、PyTorch テンソル、または TensorFlow テンソルを処理できます。値がこれらのタイプのいずれかである場合、テンソル全体をバイナリファイルに保持し、最小、平均、分散、パーセンタイルなどの高レベルのメトリクスを summary オブジェクトに格納します。

`wandb.log` で記録された最後の値は、W&B の Run で summary 辞書として自動的に設定されます。summary メトリクス辞書が変更されると、以前の値は失われます。

次のコードスニペットは、カスタム summary メトリクスを W&B に提供する方法を示しています。

```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.run.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

トレーニングが完了した後、既存の W&B の Run の summary 属性を更新できます。[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使用して、summary 属性を更新します。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## summary メトリクスのカスタマイズ

カスタムメトリクスの summary は、`wandb.summary` でのトレーニングの最後のステップではなく、最適なステップでモデルのパフォーマンスをキャプチャするのに役立ちます。たとえば、最終的な値の代わりに、最大精度または最小損失値をキャプチャすることができます。

Summary メトリクスは、`define_metric` の `summary` 引数を使用して制御できます。この引数は、`"min"`、`"max"`、`"mean"`、`"best"`、`"last"`、および `"none"` の値を受け入れます。`"best"` パラメータは、`"minimize"` および `"maximize"` の値を受け入れるオプションの `objective` 引数と組み合わせてのみ使用できます。以下は、履歴からの最終値を使用するデフォルトの summary の振る舞いではなく、損失の最小値と精度の最大値を summary にキャプチャする例です。

```python
import wandb
import random

random.seed(1)
wandb.init()
# 最小値に関心のあるメトリクスを定義します
wandb.define_metric("loss", summary="min")
# 最大値に関心のあるメトリクスを定義します
wandb.define_metric("acc", summary="max")
for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

以下は、Project Page のワークスペースのサイドバーにある固定された列に表示される、最小値と最大値の summary の結果です。

{{< img src="/images/track/customize_sumary.png" alt="Project Page Sidebar" >}}
