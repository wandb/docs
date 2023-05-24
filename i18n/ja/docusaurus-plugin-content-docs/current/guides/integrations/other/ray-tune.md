---
slug: /guides/integrations/ray-tune
description: How to integrate W&B with Ray Tune.
displayed_sidebar: default
---

# Ray Tune

W&Bは、軽量な2つの統合を提供することで[Ray](https://github.com/ray-project/ray)と統合します。

1つ目は`WandbLogger`で、Tuneに報告されたメトリクスをWandb APIに自動的にログします。もう1つは`@wandb_mixin`デコレータで、関数APIと一緒に使用できます。これにより、Tuneのトレーニング情報を用いてWandb APIが自動的に初期化されます。たとえば、`wandb.log()`を使用してトレーニングプロセスを記録するなど、通常どおりWandb APIを使用できます。

## WandbLogger

```python
from ray.tune.integration.wandb import WandbLogger
```

Wandbの設定は、`tune.run()`のconfigパラメータにwandbキーを渡すことで行われます（以下の例参照）。

wandb configエントリの内容は、キーワード引数として`wandb.init()`に渡されます。ただし、以下の設定は、`WandbLogger`自体の設定に使用される例外です。

### パラメータ

`api_key_file (str)` – `Wandb APIキー`が含まれるファイルへのパス。

`api_key (str)` – Wandb APIキー。`api_key_file`を設定する代わりに使用。

`excludes (list)` – `log`から除外されるべきメトリクスのリスト。

`log_config (bool)` – 結果 dict の config パラメータがログに記録されるかどうかを示す真偽値。`PopulationBasedTraining`のように、トレーニング中にパラメータが変更される場合などに役立ちます。デフォルトはFalseです。
### 例

```python
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
tune.run(
    train_fn,
    config={
        # ここで検索空間を定義します
        "parameter_1": tune.choice([1, 2, 3]),
        "parameter_2": tune.choice([4, 5, 6]),
        # wandb設定
        "wandb": {
            "project": "Optimization_Project",
            "api_key_file": "/path/to/file",
            "log_config": True
        }
    },
    loggers=DEFAULT_LOGGERS + (WandbLogger, ))
```

## wandb\_mixin

```python
ray.tune.integration.wandb.wandb_mixin(func)
```

このRay Tune Trainable `mixin`は、`Trainable`クラスや関数APIの`@wandb_mixin`とともに、Wandb APIを初期化するのに役立ちます。

基本的な使い方は、トレーニング関数の前に`@wandb_mixin`デコレータを付けるだけです:

```python
from ray.tune.integration.wandb import wandb_mixin

@wandb_mixin
def train_fn(config):
    wandb.log()
```

Wandbの設定は、`wandb key`を`tune.run()`の`config`パラメータに渡すことで行われます（下の例を参照してください）。

wandb設定エントリの内容は、`wandb.init()`にキーワード引数として渡されます。ただし、以下の設定は`WandbTrainableMixin`自体を設定するために使用されます。

### パラメータ

`api_key_file（str）` – Wandbの`APIキー`が含まれるファイルへのパス。

`api_key（str）` – Wandb APIキー。`api_key_file`を設定する代わりに。

Wandbの`group`、`run_id`、`run_name`はTuneによって自動的に選択されますが、対応する設定値を記入することで上書きすることができます。

他の有効な設定設定については、こちらを参照してください。[https://docs.wandb.com/library/init](https://docs.wandb.com/library/init)

### 例：

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

        # ここで検索範囲を定義する

        "a": tune.choice([1, 2, 3]),

        "b": tune.choice([4, 5, 6]),

        # wandbの設定

        "wandb": {

            "project": "Optimization_Project",

            "api_key_file": "/path/to/file"

        }

    })

```



## 例示コード



以下は、統合の使い方を確認するために作成したいくつかの例です。



* [Colab](http://wandb.me/raytune-colab): 統合を試す簡単なデモ。

* [ダッシュボード](https://wandb.ai/anmolmann/ray\_tune): この例から生成されたダッシュボードを表示します。