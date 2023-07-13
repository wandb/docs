---
displayed_sidebar: ja
---
# ログサマリーメトリクス

トレーニング中に時間の経過とともに変化する値に加えて、モデルや前処理ステップを要約する単一の値をトラッキングすることが重要であることがよくあります。W&B Runの `summary` 辞書にこの情報をログに記録します。Runのサマリ辞書では、numpy配列、PyTorchテンソル、TensorFlowテンソルを扱うことができます。値がこれらの型のいずれかである場合、テンソル全体をバイナリファイルに永続化し、サマリーオブジェクトに最小値、平均値、分散、95パーセンタイルなどの高レベルのメトリクスを格納します。

`wandb.log` でログに記録された最後の値が、W&B Runのサマリー辞書に自動的に設定されます。サマリーメトリクス辞書が変更されると、前の値が失われます。

以下のコードスニペットは、W&Bにカスタムサマリーメトリックを提供する方法を示しています。

```python
wandb.init(config=args)

best_accuracy = 0
for epoch in range(1, args.epochs + 1):
  test_loss, test_accuracy = test()
  if (test_accuracy > best_accuracy):
    wandb.run.summary["best_accuracy"] = test_accuracy
    best_accuracy = test_accuracy
```

トレーニングが完了した後、既存のW&B Runのサマリ属性を更新することができます。[W&B Public API](../../../ref/python/public-api/README.md) を使用してサマリ属性を更新します：

```python
api = wandb.Api()
run = api.run("username/project/run_id")
run.summary["tensor"] = np.random.random(1000)
run.summary.update()
```

## サマリーメトリックスのカスタマイズ

カスタムメトリックサマリーは、`wandb.summary` の最適なステップでのモデルのパフォーマンスをキャプチャするために役立ちます。最後のステップではなく、たとえば、最大の精度や最小の損失値をキャプチャしたい場合があります。
要約メトリクスは、`define_metric`の`summary`引数を使って制御できます。これは、次の値を受け入れます：`"min"`、`"max"`、`"mean"`、`"best"`、`"last"`、そして`"none"`。`"best"`パラメータは、オプションの`objective`引数と併せて使用することができます。`objective`引数は`"minimize"`と`"maximize"`を受け入れます。以下は、デフォルトの要約振る舞いではなく、要約で損失の最小値と精度の最大値を取得する例です。

```python

import wandb

import random

random.seed(1)

wandb.init()

# define a metric we are interested in the minimum of

wandb.define_metric("loss", summary="min")

# define a metric we are interested in the maximum of

wandb.define_metric("acc", summary="max")

for i in range(10):

  log_dict = {

      "loss": random.uniform(0,1/(i+1)),

      "acc": random.uniform(1/(i+1),1),

  }

  wandb.log(log_dict)

```

以下は、結果として得られる最小値と最大値の要約が、プロジェクトページのワークスペースのサイドバーにある固定された列にどのように表示されるかを示しています。

![プロジェクトページのサイドバー](/images/track/customize_sumary.png)