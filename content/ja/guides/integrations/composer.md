---
title: MosaicML Composer
description: 最新のアルゴリズムでニューラルネットワークをトレーニングしましょう。
menu:
  default:
    identifier: ja-guides-integrations-composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer) は、ニューラルネットワークをより良く、より速く、より安くトレーニングするためのライブラリです。ニューラルネットワークのトレーニングを加速し、一般化を改善するための最先端のメソッドが多数含まれており、多様な強化を簡単に _構成_ できるオプションの [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API も備えています。

W&B は、ML 実験をログに記録するための軽量なラッパーを提供します。しかし、それらを自分で組み合わせる必要はありません。W&B は [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) を介して Composer ライブラリに直接組み込まれています。

## W&B へのログ記録を始める

```python
from composer import Trainer
from composer.loggers import WandBLogger
﻿
trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="どこでもアクセス可能なインタラクティブなダッシュボード、およびその他多数！" >}}

## Composer の `WandBLogger` を使用する

Composer ライブラリは `Trainer` 内で [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) クラスを使用して、Weights & Biases にメトリクスをログに記録します。ロガーをインスタンス化して `Trainer` に渡すだけで簡単に利用できます。

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## Logger 引数

WandBLogger のパラメータは以下の通りです。詳細と説明は [Composer のドキュメント](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html)を参照してください。

| Parameter                       | Description                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B プロジェクト名 (str, オプション)
| `group`                   | W&B グループ名 (str, オプション)
| `name`                   |  W&B run 名。指定されていない場合は State.run_name が使用されます (str, オプション)
| `entity`                   | ユーザー名や W&B Team 名のような W&B entity 名 (str, オプション)
| `tags`                   | W&B タグ (List[str], オプション)
| `log_artifacts`                 | チェックポイントを wandb にログするかどうか、デフォルト: `false` (bool, オプション)|
| `rank_zero_only`         | ランクゼロプロセスでのみログするかどうか。アーティファクトをログする際には全ランクでのログを強く推奨します。ランク ≥1 のアーティファクトは保存されず、関連情報が破棄される可能性があります。例えば、Deepspeed ZeRO を使用する場合、全ランクのアーティファクトがないとチェックポイントからの復元が不可能です。デフォルト: `True` (bool, オプション)
| `init_kwargs`                   | `wandb.init` に渡すパラメータ。wandb `config` など。[こちらを参照]({{< relref path="/ref/python/init" lang="ja" >}})して、`wandb.init` が受け入れる全リストを確認してください。                                                                                                                                                                                   

通常の使用方法の例：

```
init_kwargs = {"notes":"高い学習率における実験のテスト", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 予測サンプルをログする

[Composer の Callbacks](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) システムを使用して、WandBLogger を介して Weights & Biases にいつログを記録するかを制御できます。この例では、検証画像と予測のサンプルがログに記録されます。

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """バッチごとに予測を計算し、自身のデータに格納する"""
        
        if state.timer.epoch == state.max_duration: #最後の検証エポックで
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        "wandb.Table を作成してログする"
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