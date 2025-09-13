---
title: Ray Tune
description: W&B を Ray Tune と連携する方法。
menu:
  default:
    identifier: ja-guides-integrations-ray-tune
    parent: integrations
weight: 360
---

W&B は、2 つの軽量なインテグレーションを提供することで [Ray](https://github.com/ray-project/ray) と連携します。

- `WandbLoggerCallback` 関数は、Ray Tune にレポートされたメトリクスを Wandb API に自動でログします。
- 関数 API で使える `setup_wandb()` は、Tune のトレーニング情報を用いて Wandb API を自動初期化します。初期化後は通常どおり Wandb API を利用でき、例えば `run.log()` でトレーニング プロセスをログできます。

## インテグレーションを設定する

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb の設定は、`tune.run()` の config 引数に wandb キーを渡すことで行います（下記の例を参照）。

wandb 設定エントリの内容は、キーワード引数として `wandb.init()` に渡されます。例外として、以下の設定は `WandbLoggerCallback` 自体の設定に使用されます。

### パラメータ

`project (str)`：Wandb Project 名。必須。

`api_key_file (str)`：Wandb API キーを含むファイルへのパス。

`api_key (str)`：Wandb API キー。`api_key_file` を設定する代わりに使用できます。

`excludes (list)`：ログから除外するメトリクスのリスト。

`log_config (bool)`：結果の辞書の config パラメータをログするかどうか。デフォルトは False。

`upload_checkpoints (bool)`：True の場合、モデル チェックポイントを Artifacts としてアップロードします。デフォルトは False。

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

このユーティリティ関数は、Ray Tune で Wandb を使うための初期化を支援します。基本的な使い方としては、トレーニング関数内で `setup_wandb()` を呼び出します。

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
        # ここに探索空間を定義
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb の設定
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
```

## サンプル コード

インテグレーションの動作を確認できるサンプルをいくつか用意しました。

* [Colab](https://wandb.me/raytune-colab)：インテグレーションを試せるシンプルなデモ。
* [ダッシュボード](https://wandb.ai/anmolmann/ray_tune)：サンプルから生成されたダッシュボードを閲覧。