---
title: MosaicML Composer
description: ニューラルネットワークを学習させる最先端のアルゴリズム
menu:
  default:
    identifier: ja-guides-integrations-composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer) は、ニューラルネットワークのトレーニングをより良く、より速く、より低コストで行うためのライブラリです。ニューラルネットワークのトレーニングを高速化し汎化性能を高める最先端の手法を多数含み、さらに多様な拡張の _組み合わせ_ を簡単にするオプションの [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API も提供します。

W&B は、ML の実験をログするための軽量なラッパーを提供します。ですが両者を自分で組み合わせる必要はありません。W&B は [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) を通じて Composer ライブラリに直接統合されています。

## W&B へのログ記録を開始する

```python
from composer import Trainer
from composer.loggers import WandBLogger

trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="インタラクティブなダッシュボード" >}}

## Composer の `WandBLogger` を使う

Composer ライブラリは、`Trainer` 内で [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) クラスを使って W&B にメトリクスをログします。Logger をインスタンス化して `Trainer` に渡すだけです。

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## Logger の引数

以下は `WandbLogger` の主なパラメータです。完全な一覧と説明は [Composer のドキュメント](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html) を参照してください。

| パラメータ                       | 説明                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B Project 名 (str, 任意)
| `group`                   | W&B のグループ名 (str, 任意)
| `name`                   | W&B Run 名。未指定の場合は State.run_name が使われます (str, 任意)
| `entity`                   | W&B Entity 名（例: あなたのユーザー名や W&B Team 名）(str, 任意)
| `tags`                   | W&B のタグ (List[str], 任意)
| `log_artifacts`                 | チェックポイントを W&B にログするかどうか。デフォルト: `false` (bool, 任意)|
| `rank_zero_only`         | ランク 0 のプロセスでのみログするかどうか。Artifacts をログする場合は、すべてのランクでログすることを強く推奨します。ランク ≥1 の Artifacts は保存されず、重要な情報が失われる可能性があります。例えば Deepspeed ZeRO 使用時、すべてのランクの Artifacts がなければチェックポイントからの復元が不可能になります。デフォルト: `True` (bool, 任意)
| `init_kwargs`                   | `wandb.init()` に渡す引数（wandb の `config` など）。`wandb.init()` が受け付ける引数は、[`wandb.init()` のパラメータ]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) を参照してください。

一般的な使い方の例は次のとおりです:

```
init_kwargs = {"notes":"Testing higher learning rate in this experiment", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 予測サンプルをログする

[Composer の Callbacks](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) システムを使って、`WandBLogger` 経由で W&B にいつログするかを制御できます。次の例では、検証画像と予測のサンプルをログします。

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """バッチごとに予測を計算し、self.data に保存します"""
        
        if state.timer.epoch == state.max_duration: # 最後の検証エポックで
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        with wandb.init() as run:
            "wandb.Table を作成してログします"
            columns = ['image', 'ground truth', 'prediction']
            table = wandb.Table(columns=columns, data=self.data[:self.num_samples])
            run.log({'sample_table':table}, step=int(state.timer.batch))         
...

trainer = Trainer(
    ...
    loggers=[WandBLogger()],
    callbacks=[LogPredictions()]
)
```