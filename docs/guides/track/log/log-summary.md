---
displayed_sidebar: default
---


# ログサマリーメトリクス

トレーニング中に時間とともに変化する値に加えて、モデルや前処理ステップを要約する単一の値を追跡することも重要です。この情報を W&B Run の `summary` 辞書にログします。Run の summary 辞書は、numpy 配列、PyTorch テンソル、または TensorFlow テンソルを処理できます。値がこれらのタイプのいずれかの場合、テンソル全体をバイナリファイルに保存し、サマリーオブジェクトに min、mean、variance、95th percentile などの高レベルのメトリクスを保存します。

最後に `wandb.log` でログされた値は、自動的に W&B Run の summary 辞書として設定されます。もしサマリーメトリクス辞書が変更された場合、以前の値は失われます。

以下のコードスニペットは、W&B にカスタムサマリーメトリクスを提供する方法を示しています。
```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
    test_loss, test_accuracy = test()
    if test_accuracy > best_accuracy:
        wandb.run.summary["best_accuracy"] = test_accuracy
        best_accuracy = test_accuracy
```

トレーニングが完了した後に、既存の W&B Run のサマリー属性を更新できます。[W&B パブリック API](../../../ref/python/public-api/README.md)を使用して、サマリー属性を更新します。

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## カスタムサマリーメトリクス

カスタムメトリクスサマリーは、トレーニングの最後のステップではなく、最良のステップでのモデル性能をキャプチャするのに役立ちます。例えば、最終的な値ではなく、最大精度や最小損失値をキャプチャしたい場合があります。

サマリーメトリクスは `define_metric` の `summary` 引数を使用して制御できます。`summary` 引数は、`"min"`, `"max"`, `"mean"`, `"best"`, `"last"` および `"none"` の値を受け入れます。`"best"` パラメータは、`"minimize"` と `"maximize"` の値を受け入れるオプションの `objective` 引数と組み合わせて使用することができます。ここに、履歴の最終的な値を使用するデフォルトサマリ―の振る舞いの代わりに、損失の最小値と精度の最大値をキャプチャする例を示します。

```python
import wandb
import random

random.seed(1)
wandb.init()
# 最小値に興味のあるメトリクスを定義する
wandb.define_metric("loss", summary="min")
# 最大値に興味のあるメトリクスを定義する
wandb.define_metric("acc", summary="max")
for i in range(10):
    log_dict = {
        "loss": random.uniform(0, 1 / (i + 1)),
        "acc": random.uniform(1 / (i + 1), 1),
    }
    wandb.log(log_dict)
```

以下は、プロジェクトページワークスペースのサイドバーでピン留めされた列に表示される最小および最大サマリー値の例です。

![Project Page Sidebar](/images/track/customize_sumary.png)