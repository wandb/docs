---
title: MosaicML Composer
description: 最新のアルゴリズムでニューラルネットワークをトレーニング
menu:
  default:
    identifier: ja-guides-integrations-composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer) は、ニューラルネットワークをより効率的かつ低コストでトレーニングするためのライブラリです。最先端のトレーニング高速化や汎化性能向上のためのさまざまなメソッドが含まれており、さらに多様な改良を簡単に"合成"できる便利な [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API も用意されています。

W&B は、機械学習実験のログを簡単に記録できる軽量ラッパーを提供しています。しかも、Composer ライブラリにはすでに [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) を通じて W&B が直接組み込まれているので、特別な統合作業は必要ありません。

## W&B へのログ記録を開始する

```python
from composer import Trainer
from composer.loggers import WandBLogger

trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="インタラクティブなダッシュボード" >}}

## Composer の `WandBLogger` を使う

Composer ライブラリは `Trainer` の中で [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) クラスを利用し、W&B へメトリクスを記録します。ロガーを生成して `Trainer` に渡すだけで、セットアップは完了です。

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## Logger の引数

`WandBLogger` で利用できる主なパラメータは下記の通りです。すべての詳細については [Composer ドキュメント](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html) も参照してください。

| パラメータ                       | 説明                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `project`                 | W&B Project 名（str, オプション）
| `group`                   | W&B group 名（str, オプション）
| `name`                   |  W&B Run 名。指定しない場合、State.run_name が利用されます（str, オプション）
| `entity`                   | W&B entity 名（あなたのユーザー名や W&B Team 名など）（str, オプション）
| `tags`                   | W&B tags（List[str], オプション）
| `log_artifacts`                 | チェックポイントを wandb に記録するかどうか（デフォルト: `false`）（bool, オプション）|
| `rank_zero_only`         | 0番プロセスのみでログを記録するかどうか。Artifacts を記録する場合は、全ランクで記録することを強く推奨します。rank≥1からの Artifacts は保存されず、重要な情報が失われる可能性があります。例：Deepspeed ZeRO 利用時、すべてのランクの Artifacts がないとチェックポイント復元ができません（デフォルト: `True`）（bool, オプション）
| `init_kwargs`                   | `wandb.init()` に渡すパラメータ（例：wandb `config` など）。受け付けるパラメータの詳細は [`wandb.init()` のパラメータ]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) をご確認ください。

使用例：

```
init_kwargs = {"notes":"この実験で高い学習率をテスト中", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 予測サンプルのログ出力

Composer の [Callbacks](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) システムを活用して、`WandBLogger` 経由でいつ W&B にログを送信するかを管理できます。下記の例では、バリデーション画像とその予測結果のサンプルをログしています。

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """バッチごとに予測値を計算し、self.data に格納します"""
        
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