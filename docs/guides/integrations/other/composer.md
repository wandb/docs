---
description: 最新のアルゴリズムでニューラルネットワークを訓練する
slug: /guides/integrations/composer
displayed_sidebar: default
---


# MosaicML Composer

[**Colabノートブックで試す →**](https://github.com/wandb/examples/blob/master/colabs/mosaicml/MosaicML_Composer_and_wandb.ipynb)

[Composer](https://github.com/mosaicml/composer)は、ニューラルネットワークをより良く、より速く、より安くトレーニングするためのライブラリです。ニューラルネットワークのトレーニングを高速化し、汎化性能を向上させる最先端のメソッドを多数含んでおり、多様な強化を簡単に組み合わせられる[Trainer](https://docs.mosaicml.com/projects/composer/en/stable/trainer/using_the_trainer.html) APIも提供しています。

W&Bは、あなたのML実験をログするための軽量なラッパーを提供します。しかし、これらを自分で組み合わせる必要はありません。W&BはComposerライブラリに直接組み込まれており、[WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts)を通じて使用できます。

## 1行のコードでW&Bにログを開始

```python
from composer import Trainer
from composer.loggers import WandBLogger
﻿
trainer = Trainer(..., logger=WandBLogger())
```

![どこでもアクセスできるインタラクティブなダッシュボードなど！](@site/static/images/integrations/n6P7K4M.gif)

## Composerの `WandBLogger` の使用

Composerライブラリは `Trainer` 内で [WandBLogger](https://docs.mosaicml.com/projects/composer/en/stable/trainer/file_uploading.html#weights-biases-artifacts) クラスを使用して、Weights & Biases にメトリクスをログします。ロガーをインスタンス化し、それを `Trainer` に渡すだけで簡単です。

```
wandb_logger = WandBLogger(project="gpt-5", log_artifacts=True)
trainer = Trainer(logger=wandb_logger)
```

### ロガーの引数

WandbLoggerのパラメータは以下の通りです。完全なリストと詳細については、[Composerのドキュメント](https://docs.mosaicml.com/projects/composer/en/stable/api_reference/generated/composer.loggers.WandBLogger.html) を参照してください。

| パラメータ                       | 説明                                                                                                                                                                                                                                                                                                                                                              |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `project`                 | W&Bプロジェクト名 (str, オプション)
| `group`                   | W&Bグループ名 (str, オプション)
| `name`                   | W&B run名。指定しない場合、State.run_nameが使用される (str, オプション)
| `entity`                   | W&Bエンティティ名（ユーザー名やW&B Team名など） (str, オプション)
| `tags`                   | W&Bタグ (List[str], オプション)
| `log_artifacts`                 | チェックポイントをwandbにログするかどうか、デフォルト: `false` (bool, オプション)|
| `rank_zero_only`         | ランクゼロプロセスのみでログを取るかどうか。Artifactsをログする場合、すべてのランクでログを取ることが強く推奨されます。ランク ≥1 のアーティファクトは保存されず、重要な情報が失われる可能性があります。例えば、Deepspeed ZeROを使用する場合、すべてのランクからのアーティファクトがないとチェックポイントからの復元が不可能になります。デフォルト: `True` (bool, オプション)
| `init_kwargs`                   | `wandb.init` に渡すパラメータ（例えば、wandbの `config` など）。[こちらを参照](https://docs.wandb.ai/ref/python/init) で `wandb.init` が受け付ける完全なリストが確認できます。

典型的な使用例は以下の通りです:

```
init_kwargs = {"notes":"この実験で高い学習率をテストしています", 
               "config":{"arch":"Llama",
                         "use_mixed_precision":True
                         }
               }

wandb_logger = WandBLogger(log_artifacts=True, init_kwargs=init_kwargs)
```

### 予測サンプルのログ

WandBLoggerを介してWeights & Biasesにログを取るタイミングを制御するために、[ComposerのCallbacks](https://docs.mosaicml.com/projects/composer/en/stable/trainer/callbacks.html) システムを使用できます。この例では、バリデーション画像と予測のサンプルがログされます：

```python
import wandb
from composer import Callback, State, Logger

class LogPredictions(Callback):
    def __init__(self, num_samples=100, seed=1234):
        super().__init__()
        self.num_samples = num_samples
        self.data = []
        
    def eval_batch_end(self, state: State, logger: Logger):
        """バッチごとに予測を計算し、self.dataに保存します"""
        
        if state.timer.epoch == state.max_duration: # 最終バルエポックのとき
            if len(self.data) < self.num_samples:
                n = self.num_samples
                x, y = state.batch_pair
                outputs = state.outputs.argmax(-1)
                data = [[wandb.Image(x_i), y_i, y_pred] for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))]
                self.data += data
            
    def eval_end(self, state: State, logger: Logger):
        "wandb.Tableを作成してログします"
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