---
title: MosaicML Composer
description: 最先端のアルゴリズムでニューラルネットワークをトレーニング
menu:
  default:
    identifier: ja-guides-integrations-composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer) は、ニューラルネットワークのトレーニングをより良く、より速く、より安価にするためのライブラリです。ニューラルネットワークのトレーニングを加速させ、汎化性能を向上させるための多くの最先端の メソッド が含まれています。また、オプションの [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) APIを使用すると、さまざまな拡張機能を簡単に _構成_ できます。

Weights & Biases は、ML 実験 の ログ を記録するための軽量なラッパーを提供します。ただし、2つを自分で組み合わせる必要はありません。Weights & Biases は、[WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) を介して Composer ライブラリに直接組み込まれています。

## Weights & Biases への ログ の記録を開始する

```python
from composer import Trainer
from composer.loggers import WandBLogger
﻿
trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="インタラクティブな ダッシュボード にどこからでもアクセス可能!" >}}

## Composer の `WandBLogger` の使用

Composer ライブラリは、`Trainer` の [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) クラスを使用して、 メトリクス を Weights & Biases に ログ します。ロガーをインスタンス化して `Trainer` に渡すのと同じくらい簡単です。

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## ロガーの 引数

WandbLogger の パラメータ については、完全なリストと説明について [Composer のドキュメント](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html) を参照してください。

| パラメータ | 説明 |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project` | Weights & Biases の プロジェクト 名 (str, optional)
| `group` | Weights & Biases の グループ 名 (str, optional)
| `name` | Weights & Biases の run 名。指定されていない場合、State.run_name が使用されます (str, optional)
| `entity` | Weights & Biases の エンティティ 名 ( ユーザー 名または Weights & Biases の Teams 名など) (str, optional)
| `tags` | Weights & Biases の タグ (List[str], optional)
| `log_artifacts` | チェックポイント を wandb に ログ するかどうか、デフォルト: `false` (bool, optional)|
| `rank_zero_only` | ランク 0 の プロセス でのみ ログ を記録するかどうか。Artifacts を ログ に記録する場合は、すべてのランクで ログ に記録することを強くお勧めします。ランク ≥1 からの Artifacts は保存されず、関連情報が破棄される可能性があります。たとえば、Deepspeed ZeRO を使用する場合、すべてのランクからの Artifacts がないと チェックポイント から復元することは不可能です。デフォルト: `True` (bool, optional)
| `init_kwargs` | wandb `config` などの `wandb.init` に渡す パラメータ [完全なリストについては、こちら]({{< relref path="/ref/python/init" lang="ja" >}}) `wandb.init` が受け入れます

一般的な使用法は次のとおりです。

```
init_kwargs = {"notes":"この 実験 でより高い学習率をテストする", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 予測 サンプル の ログ

[Composer の Callbacks](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) システムを使用して、WandBLogger 経由で Weights & Biases に ログ を記録するタイミングを制御できます。この例では、検証画像と 予測 の サンプル が ログ に記録されます。

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """バッチ ごとに 予測 を計算し、self.data に保存します"""
        
        if state.timer.epoch == state.max_duration: #最後の val エポック で
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        "wandb.Table を作成して ログ に記録します"
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