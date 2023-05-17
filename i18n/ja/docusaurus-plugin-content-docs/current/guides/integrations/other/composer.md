---
slug: /guides/integrations/composer
description: State of the art algorithms to train your neural networks
displayed_sidebar: default
---

# MosaicML Composer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://wandb.me/composer)

[Composer](https://github.com/mosaicml/composer)は、ニューラルネットワークをより良く、より速く、より安価にトレーニングするためのライブラリです。ニューラルネットワークのトレーニングを加速し、一般化を向上させる最先端の多くの手法を提供し、さまざまな強化機能を簡単に組み合わせるためのオプションの[Trainer](https://docs.mosaicml.com/en/v0.5.0/trainer/using\_the\_trainer.html) APIが含まれています。

W&Bは、ML実験のログを記録するための軽量なラッパーを提供します。しかし、自分で両方を組み合わせる必要はありません。Weights & Biasesは、[WandBLogger](https://docs.mosaicml.com/en/latest/api\_reference/composer.loggers.wandb\_logger.html#composer-loggers-wandb-logger)を介してComposerライブラリに直接組み込まれています。

## コード2行でW&Bにログを記録する

```python
from composer import Trainer
from composer.loggers import WandBLogger
﻿
wandb_logger = WandBLogger(init_params=init_params)
trainer = Trainer(..., logger=wandb_logger)
```

![どこからでもアクセスできるインタラクティブなダッシュボードなど！](@site/static/images/integrations/n6P7K4M.gif)

## Composerの`WandBLogger`を使用する

Composerライブラリには、`Trainer`と一緒に使用して、指標をWeights and Biasesにログを記録する[WandBLogger](https://docs.mosaicml.com/en/latest/api\_reference/composer.loggers.wandb\_logger.html#composer-loggers-wandb-logger)クラスがあります。ロガーをインスタンス化して`Trainer`に渡すだけです。

```
wandb_logger = WandBLogger()
trainer = Trainer(logger=wandb_logger)
```
### Logger引数

以下は、WandbLoggerでよく使用されるパラメータのいくつかです。全てのリストと説明については、[Composerのドキュメント](https://docs.mosaicml.com/en/latest/api\_reference/composer.loggers.wandb\_logger.html#composer-loggers-wandb-logger)を参照してください。

| パラメータ                        | 説明                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `init_params`                   | `wandb.init`に渡すパラメータ、例えばwandbの`project`、`entity`、`name`や`config`などです。[こちら](https://docs.wandb.ai/ref/python/init)で`wandb.init`が受け入れる完全なリストを参照してください。                                                                                                                                                                                   |
| `log_artifacts`                 | wandbにチェックポイントをログするかどうか                                                                                                                                                                                                                                                                                                                                       |
| `log_artifacts_every_n_batches` | アーティファクトをアップロードする間隔です。`log_artifacts=True`の場合のみ適用されます。                                                                                                                                                                                                                                                                                        |
| `rank_zero_only`                | ランクゼロプロセスでのみログするかどうかです。wandbに`artifacts`をログする場合、すべてのランクでログすることを強くお勧めします。ランク≥1のアーティファクトは保存されず、重要な情報が破棄される可能性があります。例えば、DeepSpeed ZeROを使用する場合、すべてのランクのアーティファクトがなければ、チェックポイントから復元することは不可能です（デフォルト：`False`） |

典型的な使用方法は以下の通りです。

```
init_params = {"project":"composer", 
               "name":"imagenette_benchmark",
               "config":{"arch":"Resnet50",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_params=init_params)
```

### 予測サンプルのログ

[Composerのコールバック](https://docs.mosaicml.com/en/latest/trainer/callbacks.html)システムを使用して、WandBLoggerを介してWeights & Biasesにログするタイミングを制御できます。この例では、検証画像と予測のサンプルをログします。

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):

    def __init__(self, num_samples=100, seed=1234):

        super().__init__()

        self.num_samples = num_samples

        self.data = []

        

    def eval_batch_end(self, state: State, logger: Logger):

        """バッチごとに予測を計算し、self.dataに格納する"""

        

        if state.timer.epoch == state.max_duration: #最後のvalエポックに

            if len(self.data) < self.num_samples:

                n = self.num_samples

                x, y = state.batch_pair

                outputs = state.outputs.argmax(-1)

                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]

                self.data += data

            

    def eval_end(self, state: State, logger: Logger):

        "wandb.Tableを作成し、ログに記録する"

        columns = ['image', 'ground truth', 'prediction']

        table = wandb.Table(columns=columns, data=self.data[:self.num_samples])

        wandb.log({'sample_table':table}, step=int(state.timer.batch))         

...



trainer = Trainer(

    ...

    loggers=[WandBLogger()],

    callbacks=[LogPredictions()]

)

```