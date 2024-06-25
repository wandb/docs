---
description: W&B を Ray Tune と統合する方法
slug: /guides/integrations/ray-tune
displayed_sidebar: default
---


# Ray Tune

W&Bは、軽量な2つのインテグレーションを提供することで[Ray](https://github.com/ray-project/ray)と統合します。

1つは `WandbLoggerCallback` で、これはTuneに報告されたメトリクスを自動的にWandb APIにログします。もう1つは `@wandb_mixin` デコレーターで、これは関数APIと一緒に使用できます。これにより、Tuneのトレーニング情報でWandb APIが自動的に初期化されます。通常通りWandb APIを使用することができ、例として `wandb.log()` を使ってトレーニングプロセスをログします。

## WandbLoggerCallback

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandbの設定は、`tune.run()` のconfig引数にwandbキーを渡すことで行います（以下の例を参照）。

wandbの設定エントリの内容は、キーワード引数として `wandb.init()` に渡されます。ただし、以下の設定は `WandbLoggerCallback` 自体の設定に使用されます。

### パラメータ

`api_key_file (str)` – `Wandb API KEY` を含むファイルのパス。

`api_key (str)` – Wandb APIキー。`api_key_file`を設定する代わりに使用します。

`excludes (list)` – `log`から除外されるメトリクスのリスト。

`log_config (bool)` – 結果辞書の設定引数をログするかどうかを示すブール値。これは例えば `PopulationBasedTraining` でパラメータがトレーニング中に変わる場合に意味があります。デフォルトはFalseです。

### 例

```python
from ray import tune, train
from ray.tune.logger import DEFAULT_LOGGERS
from ray.air.integrations.wandb import WandbLoggerCallback

def train_fc(config):
    for i in range(10):
        train.report({"mean_accuracy":(i + config['alpha']) / 10})

search_space = {
    'alpha': tune.grid_search([0.1, 0.2, 0.3]),
    'beta': tune.uniform(0.5, 1.0)
}

analysis = tune.run(
    train_fc,
    config=search_space,
    callbacks=[WandbLoggerCallback(
        project="<your-project>",
        api_key="<your-name>",
        log_config=True
    )]
)

best_trial = analysis.get_best_trial("mean_accuracy", "max", "last")
```

## wandb\_mixin

```python
ray.tune.integration.wandb.wandb_mixin(func)
```

このRay TuneのTrainable `mixin`は、`Trainable`クラスや関数APIで `@wandb_mixin` と使うためにWandb APIを初期化するのに役立ちます。

基本的な使い方としては、トレーニング関数の前に `@wandb_mixin` デコレータを付けるだけです：

```python
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_fn(config):
    wandb.log()
```

Wandbの設定は、 `tune.run()` の `config` 引数に `wandb key` を渡すことで行います（以下の例を参照）。

wandbの設定エントリの内容は、キーワード引数として `wandb.init()` に渡されます。ただし、以下の設定は `WandbTrainableMixin` 自体の設定に使用されます。

### パラメータ

`api_key_file (str)` – Wandb `API KEY` を含むファイルのパス。

`api_key (str)` – Wandb APIキー。`api_key_file`を設定する代わりに使用します。

Wandbの `group`、`run_id`、`run_name` はTuneによって自動的に選択されますが、各設定値を入力することで上書き可能です。

他の有効な設定については、こちらをご覧ください: [https://docs.wandb.com/library/init](https://docs.wandb.com/library/init)

### 例:

```python
from ray import tune
from ray.tune.integration.wandb import wandb_mixin


@wandb_mixin
def train_fn(config):
    for i in range(10):
        loss = self.config["a"] + self.config["b"]
        wandb.log({"loss": loss})
        tune.report(loss=loss)


tune.run(
    train_fn,
    config={
        # ここで探索空間を定義
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandbの設定
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
```

## コード例

インテグレーションの動作を確認するいくつかの例を作成しました：

* [Colab](http://wandb.me/raytune-colab): インテグレーションを試すためのシンプルなデモ。
* [Dashboard](https://wandb.ai/anmolmann/ray\_tune): 例から生成されたダッシュボードを見る。