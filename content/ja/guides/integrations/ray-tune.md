---
title: Ray Tune
description: W&B を Ray Tune と統合する方法。
menu:
  default:
    identifier: ja-guides-integrations-ray-tune
    parent: integrations
weight: 360
---

W&B は、2 つの軽量なインテグレーションを提供することで [Ray](https://github.com/ray-project/ray) と統合します。

- `WandbLoggerCallback` 関数は、Tune に報告されたメトリクスを自動的に Wandb API にログします。
-  `setup_wandb()` 関数は、関数 API と一緒に使用するもので、Tune のトレーニング情報を使用して Wandb API を自動的に初期化します。通常どおり Wandb API を使用できます。たとえば、`wandb.log()` を用いてあなたのトレーニングプロセスをログすることができます。

## インテグレーションの設定

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb の設定は、wandb キーを `tune.run()` の config パラメータに渡すことで行われます（下記の例を参照してください）。

wandb の config エントリの内容は、キーワード引数として `wandb.init()` に渡されます。例外として、`WandbLoggerCallback` 自体の設定に使用される次の設定があります:

### Parameters

`project (str)`: Wandb プロジェクトの名前。必須。

`api_key_file (str)`: Wandb API キーが含まれるファイルへのパス。

`api_key (str)`: Wandb API キー。`api_key_file` を設定する代わり。

`excludes (list)`: ログから除外するメトリクスのリスト。

`log_config (bool)`: 結果辞書の設定パラメータをログするかどうか。デフォルトは False です。

`upload_checkpoints (bool)`: True の場合、モデルチェックポイントがアーティファクトとしてアップロードされます。デフォルトは False です。

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

このユーティリティ関数は、Ray Tune で Wandb を使用するために初期化を支援します。基本的な使用法では、トレーニング関数内で `setup_wandb()` を呼び出してください:

```python
from ray.air.integrations.wandb import setup_wandb


def train_fn(config):
    # Wandb を初期化
    wandb = setup_wandb(config)

    for i in range(10):
        loss = config["a"] + config["b"]
        wandb.log({"loss": loss})
        tune.report(loss=loss)


tuner = tune.Tuner(
    train_fn,
    param_space={
        # ここに探索空間を定義
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # Wandb 設定
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
```

## 例 コード

インテグレーションの動作を確認するためのいくつかの例を作成しました:

* [Colab](http://wandb.me/raytune-colab): インテグレーションを試すためのシンプルなデモ。
* [Dashboard](https://wandb.ai/anmolmann/ray_tune): 例から生成されたダッシュボードを表示。