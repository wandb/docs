---
title: MosaicML Composer
description: 最新のアルゴリズムでニューラルネットワークをトレーニング
menu:
  default:
    identifier: composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer) は、ニューラルネットワークのトレーニングをより良く、速く、そして安く行うためのライブラリです。最先端の手法が多数含まれており、ニューラルネットワークのトレーニングの高速化や汎化性能の向上を実現します。さらに、さまざまな強化手法を手軽に組み合わせられる [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API も用意されています。

W&B は、機械学習実験のログを手軽に記録できる軽量なラッパーを提供しています。ですが、Composer ライブラリ内ですでに W&B へのログ記録が組み込まれているため、自分で両者を組み合わせる必要はありません。[WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) を使えば、すぐに使い始められます。

## W&B へのログ記録をはじめる

```python
from composer import Trainer
from composer.loggers import WandBLogger

trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="Interactive dashboards" >}}

## Composer の `WandBLogger` を使う

Composer ライブラリは `Trainer` 内で [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) クラスを使い、メトリクスを W&B にログします。ロガーをインスタンス化し、それを `Trainer` に渡すだけで簡単に利用できます。

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## Logger の引数

`WandbLogger` で設定できる主なパラメータを以下に示します。全てのリストや詳細は [Composer のドキュメント](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html) をご覧ください。

| パラメータ                       | 説明                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B の Project 名 (str, オプション)
| `group`                   | W&B の group 名 (str, オプション)
| `name`                   |  W&B の Run 名。未指定の場合は State.run_name が使われます (str, オプション)
| `entity`                   | W&B の entity 名（例: ユーザー名や W&B Team の名前）(str, オプション)
| `tags`                   | W&B のタグ (List[str], オプション)
| `log_artifacts`                 | チェックポイントを wandb に記録するかどうか。デフォルト: `false` (bool, オプション)|
| `rank_zero_only`         | ランク 0 のプロセスのみログするかどうか。Artifacts の記録時はすべてのランクで記録するのが推奨です。ランク1以上の Artifacts は保存されず、必要な情報が失われる可能性があります。例として、Deepspeed ZeRO を利用している場合、すべてのランクの Artifact がなければチェックポイントからの復元は不可能になります。デフォルト: `True` (bool, オプション)
| `init_kwargs`                   | `wandb.init()` に渡すパラメータ（例: wandb の `config` など）。受け付けるパラメータの詳細は [`wandb.init()` のパラメータ]({{< relref "/ref/python/sdk/functions/init.md" >}}) を参照してください。

典型的な使い方の例：

```
init_kwargs = {"notes":"この実験で高めの学習率をテスト中", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 予測サンプルのログ

[Composer のコールバック](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) システムを利用すれば、どのタイミングで W&B にログするかも柔軟に制御できます。以下の例では、検証時の画像とその予測結果のサンプルをログしています。

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """バッチごとの予測を計算し、self.data に保存します"""
        
        if state.timer.epoch == state.max_duration: # 最後の val エポック時
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