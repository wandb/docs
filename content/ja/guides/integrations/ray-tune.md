---
title: Ray Tune
description: W&B と Ray Tune を統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-ray-tune
    parent: integrations
weight: 360
---

W&B は、2 つの軽量 なインテグレーションを提供することで、[Ray](https://github.com/ray-project/ray) と統合します。

- `WandbLoggerCallback` 関数は、Tune に報告された メトリクス を Wandb API に自動的に ログ します。
- 関数 API で使用できる `setup_wandb()` 関数は、Tune の トレーニング 情報で Wandb API を自動的に初期化します。`wandb.log()` を使用して トレーニング プロセス を ログ するなど、通常どおり Wandb API を使用できます。

## インテグレーション の設定

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb の 設定 は、wandb キー を `tune.run()` の config パラメータ に渡すことで行われます (下記の例を参照)。

wandb config エントリ の内容は、キーワード 引数 として `wandb.init()` に渡されます。例外は、`WandbLoggerCallback` 自体を 設定 するために使用される次の 設定 です。

### パラメータ

`project (str)`: Wandb の プロジェクト 名。必須。

`api_key_file (str)`: Wandb APIキー が含まれるファイル へのパス。

`api_key (str)`: Wandb APIキー。`api_key_file` の代替。

`excludes (list)`: ログ から除外する メトリクス のリスト。

`log_config (bool)`: results 辞書 の config パラメータ を ログ するかどうか。デフォルト は False です。

`upload_checkpoints (bool)`: True の場合、モデル の チェックポイント は Artifacts としてアップロードされます。デフォルト は False です。

### 例

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

このユーティリティ 関数 は、Ray Tune で使用するために Wandb を初期化するのに役立ちます。基本的な使用法については、 トレーニング 関数 で `setup_wandb()` を呼び出します。

```python
from ray.air.integrations.wandb import setup_wandb


def train_fn(config):
    # Initialize wandb
    wandb = setup_wandb(config)

    for i in range(10):
        loss = config["a"] + config["b"]
        wandb.log({"loss": loss})
        tune.report(loss=loss)


tuner = tune.Tuner(
    train_fn,
    param_space={
        # define search space here
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb configuration
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
```

## コード 例

インテグレーション の仕組みを理解できるように、いくつかの例を作成しました。

* [Colab](http://wandb.me/raytune-colab): インテグレーション を試すための簡単な デモ 。
* [Dashboard](https://wandb.ai/anmolmann/ray_tune): 例から生成された ダッシュボード を表示します。
