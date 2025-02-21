---
title: MosaicML Composer
description: 最先端のアルゴリズムで ニューラルネットワーク を学習
menu:
  default:
    identifier: ja-guides-integrations-composer
    parent: integrations
weight: 230
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb" >}}

[Composer](https://github.com/mosaicml/composer) は、ニューラルネットワークのトレーニングをより良く、より速く、より安価にするためのライブラリです。ニューラルネットワークのトレーニングを加速させ、汎化性能を向上させるための多くの最先端の メソッド が含まれており、オプションの [Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) API と組み合わせることで、さまざまな拡張機能を簡単に _構成_ できます。

Weights & Biases は、ML の 実験管理 を ログ するための軽量なラッパーを提供します。W&B は [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) を介して Composer ライブラリに直接組み込まれているため、自分で 2 つを組み合わせる必要はありません。

## W&B への ログ の開始

```python
from composer import Trainer
from composer.loggers import WandBLogger
﻿
trainer = Trainer(..., logger=WandBLogger())
```

{{< img src="/images/integrations/n6P7K4M.gif" alt="インタラクティブな ダッシュボード にどこからでもアクセス可能！" >}}

## Composer の `WandBLogger` の使用

Composer ライブラリは、`Trainer` の [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) クラスを使用して、メトリクス を Weights & Biases に ログ します。ロガーをインスタンス化して `Trainer` に渡すのと同じくらい簡単です。

```python
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

## ロガーの 引数

WandbLogger の パラメータ については、完全なリストと説明について [Composer のドキュメント](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html) を参照してください。

| パラメータ                       | 説明                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&B の プロジェクト 名 (str, optional)
| `group`                   | W&B の グループ 名 (str, optional)
| `name`                   | W&B の run 名。指定されていない場合、State.run_name が使用されます (str, optional)
| `entity`                   | W&B の エンティティ 名。ユーザー名または W&B の Team 名など (str, optional)
| `tags`                   | W&B の タグ (List[str], optional)
| `log_artifacts`                 | チェックポイント を wandb に ログ するかどうか、デフォルト: `false` (bool, optional)|
| `rank_zero_only`         | ランク 0 の プロセス でのみ ログ するかどうか。アーティファクト を ログ する場合、すべてのランクで ログ することを強くお勧めします。ランク 1 以上の アーティファクト は保存されず、関連情報が破棄される可能性があります。たとえば、Deepspeed ZeRO を使用する場合、すべてのランクからの アーティファクト がないと チェックポイント から復元することは不可能です。デフォルト: `True` (bool, optional)
| `init_kwargs`                   | wandb `config` など、`wandb.init` に渡す パラメータ [`wandb.init` が受け入れる完全なリストについてはこちら]({{< relref path="/ref/python/init" lang="ja" >}})を参照してください。                                                                                                                                                     


一般的な使用法は次のとおりです。

```
init_kwargs = {"notes":"この 実験 でより高い学習率をテスト", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

## 予測 サンプル の ログ

[Composer の コールバック](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) システムを使用して、WandBLogger を介して Weights & Biases への ログ のタイミングを制御できます。この例では、検証画像と 予測 の サンプル が ログ されます。

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """バッチ ごとの 予測 を計算し、self.data に格納します"""
        
        if state.timer.epoch == state.max_duration: # 最後の検証 エポック で
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        "wandb.Table を作成して ログ します"
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