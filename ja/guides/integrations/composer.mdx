---
title: MosaicML Composer
description: 最先端のアルゴリズムでニューラルネットワークをトレーニングする
menu:
  default:
    identifier: ja-guides-integrations-composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer) は、ニューラルネットワークをより良く、より速く、より安価にトレーニングするためのライブラリです。ニューラルネットワークのトレーニングを加速し、一般化能力を向上させるための最新のメソッドが多数含まれており、多様な強化を容易に組み合わせるためのオプションの [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API も用意されています。

W&B は、あなたの ML 実験をログするための軽量なラッパーを提供します。しかし、自分でそれらを組み合わせる必要はありません：W&B は [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) を介して Composer ライブラリに直接組み込まれています。

## W&B へのログの開始

```python
from composer import Trainer
from composer.loggers import WandBLogger
﻿
trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="インタラクティブなダッシュボードはどこからでもアクセス可能で、さらに多くの機能があります！" >}}

## Composer の `WandBLogger` を使用する

Composer ライブラリは、`Trainer` 内の [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) クラスを使用して、Weights & Biases へのメトリクスをログします。ロガーをインスタンス化し、それを `Trainer` に渡すだけです。

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## ロガーの引数

WandbLogger のパラメータは以下です。完全な一覧と説明については [Composer のドキュメント](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html) を参照してください

| パラメータ                       | 説明                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B プロジェクト名 (str, optional)
| `group`                   | W&B グループ名 (str, optional)
| `name`                   |  W&B run 名。指定されていない場合は State.run_name が使用されます (str, optional)
| `entity`                   | W&B エンティティ名。ユーザー名や W&B チーム名など (str, optional)
| `tags`                   | W&B タグ (List[str], optional)
| `log_artifacts`                 | チェックポイントを wandb にログするかどうか。デフォルト: `false` (bool, optional)|
| `rank_zero_only`         | ランクゼロのプロセスでのみログするかどうか。アーティファクトをログする場合、すべてのランクでログすることが強く推奨されます。ランク 1 以上のアーティファクトは保存されないため、関連する情報が失われる可能性があります。例えば、Deepspeed ZeRO を使用する場合、すべてのランクからのアーティファクトがなければチェックポイントから復元することはできません。デフォルト: `True` (bool, optional)
| `init_kwargs`                   | `wandb.init` に渡すパラメータ、`config` など。このリストについては完全な一覧を[こちら]({{< relref path="/ref/python/init" lang="ja" >}}) から確認できます。                                                                                                                                                                                   

典型的な使用法は次のとおりです：

```
init_kwargs = {"notes":"この実験での学習率の向上をテストしています", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 予測サンプルをログする

[Composer のコールバック](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) システムを使用して、WandBLogger を通じて Weights & Biases へのログを制御できます。この例では、バリデーション画像と予測のサンプルがログされています：

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """バッチごとの予測を計算し、それを self.data に保存します"""
        
        if state.timer.epoch == state.max_duration: # 最後のバリデーションエポックで
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        "wandb.Table を作成してログします"
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