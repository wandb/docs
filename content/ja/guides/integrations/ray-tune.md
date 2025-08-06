---
title: Ray チューニング
description: W&B を Ray Tune と統合する方法
menu:
  default:
    identifier: ray-tune
    parent: integrations
weight: 360
---

W&B は [Ray](https://github.com/ray-project/ray) と統合するために、2 つの軽量なインテグレーションを提供しています。

- `WandbLoggerCallback` 関数は、Tune にレポートされたメトリクスを自動で Wandb API に記録します。
- `setup_wandb()` 関数は、関数 API で利用でき、Tune のトレーニング情報とともに自動で Wandb API を初期化します。通常通り Wandb API を使って、例えば `run.log()` でトレーニングプロセスをログできます。

## インテグレーションの設定

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb の設定は、`tune.run()` の config パラメータに wandb キーを渡すことで行います（下記の例を参照してください）。

wandb の config エントリの内容は、キーワード引数として `wandb.init()` にそのまま渡されます。ただし、以下の設定だけは `WandbLoggerCallback` 自身の設定として使われます。

### パラメータ

`project (str)`: Wandb プロジェクト名。必須です。

`api_key_file (str)`: Wandb APIキー を含むファイルへのパス。

`api_key (str)`: Wandb APIキー。`api_key_file` の代わりに設定可能。

`excludes (list)`: ログから除外したいメトリクスのリスト。

`log_config (bool)`: 結果の辞書の config パラメータをログするかどうか。デフォルトは False です。

`upload_checkpoints (bool)`: True の場合、モデルのチェックポイントがアーティファクトとしてアップロードされます。デフォルトは False です。

### 使用例

```python
from ray import tune, train
from ray.air.integrations.wandb import WandbLoggerCallback


def train_fc(config):
    for i in range(10):
        train.report({"mean_accuracy": (i + config["alpha"]) / 10})


tuner = tune.Tuner(
    train_fc,
    param_space={
        "alpha": tune.grid_search([0.1, 0.2, 0.3]),
        "beta": tune.uniform(0.5, 1.0),
    },
    run_config=train.RunConfig(
        callbacks=[
            WandbLoggerCallback(
                project="<your-project>", api_key="<your-api-key>", log_config=True
            )
        ]
    ),
)

results = tuner.fit()
```

## setup_wandb

```python
from ray.air.integrations.wandb import setup_wandb
```

このユーティリティ関数は、Ray Tune で Wandb を使うための初期化をサポートします。基本的な使い方は、トレーニング関数内で `setup_wandb()` を呼び出してください。

```python
from ray.air.integrations.wandb import setup_wandb


def train_fn(config):
    # wandb を初期化
    wandb = setup_wandb(config)
    run = wandb.init(
        project=config["wandb"]["project"],
        api_key_file=config["wandb"]["api_key_file"],
    )

    for i in range(10):
        loss = config["a"] + config["b"]
        run.log({"loss": loss})
        tune.report(loss=loss)
    run.finish()


tuner = tune.Tuner(
    train_fn,
    param_space={
        # ここでサーチ空間を定義
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb の設定
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
```

## サンプルコード

インテグレーションの動作を確認できるサンプルをいくつか用意しています：

* [Colab](https://wandb.me/raytune-colab): インテグレーションをすぐに試せるシンプルなデモ。
* [Dashboard](https://wandb.ai/anmolmann/ray_tune): サンプルから生成されたダッシュボードの閲覧。