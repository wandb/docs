---
title: Ray チューニング
description: W&B を Ray Tune と連携する方法
menu:
  default:
    identifier: ja-guides-integrations-ray-tune
    parent: integrations
weight: 360
---

W&B は [Ray](https://github.com/ray-project/ray) と連携して、2つの軽量なインテグレーションを提供しています。

- `WandbLoggerCallback` 関数は、Tune に報告されたメトリクスを自動的に Wandb API にログします。
- 関数 API と組み合わせて利用できる `setup_wandb()` 関数は、Tune のトレーニング情報を使って自動的に Wandb API を初期化します。通常通り Wandb API を利用でき、たとえば `run.log()` でトレーニングプロセスを記録できます。

## インテグレーションの設定方法

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb の設定は、`tune.run()` の config パラメータに wandb キーを渡すことで行います（下記の例をご覧ください）。

wandb の config エントリの内容は、キーワード引数として `wandb.init()` に渡されます。ただし、次の設定項目は `WandbLoggerCallback` 自体の設定用として使われます。

### パラメータ

`project (str)`: Wandb プロジェクト名。必須。

`api_key_file (str)`: Wandb API キーが記載されたファイルのパス。

`api_key (str)`: Wandb API キー。`api_key_file` の代替設定。

`excludes (list)`: ログから除外するメトリクスのリスト。

`log_config (bool)`: results 辞書の config パラメータをログするかどうか。デフォルトは False。

`upload_checkpoints (bool)`: True の場合、モデルチェックポイントが artifact としてアップロードされます。デフォルトは False。

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

このユーティリティ関数は、Ray Tune で Wandb を使うための初期化を簡単にします。基本的な使い方は、トレーニング関数内で `setup_wandb()` を呼び出します。

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
        # ここでサーチスペースを定義
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb 設定情報
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
```

## サンプルコード

インテグレーションの動きを確認できるサンプルをいくつか用意しています。

* [Colab](https://wandb.me/raytune-colab): インテグレーションをすぐに試せるシンプルなデモ。
* [Dashboard](https://wandb.ai/anmolmann/ray_tune): サンプルから生成されたダッシュボードを閲覧できます。