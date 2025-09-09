---
title: Experiments の設定
description: 実験の設定を保存するには、辞書のようなオブジェクトを使用します。
menu:
  default:
    identifier: ja-guides-models-track-config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

run の `config` プロパティを使って、トレーニング設定を保存します:
- ハイパーパラメーター
- データセット名やモデルタイプなどの入力設定
- 実験 におけるその他の独立変数

`wandb.Run.config` プロパティを使うと、実験 を分析したり、将来あなたの作業を再現するのが簡単になります。W&B App で 設定値 ごとにグループ化したり、異なる W&B run の 設定を比較したり、各 トレーニング 設定が出力に与える影響を評価できます。`config` プロパティは 辞書のような オブジェクトで、複数の 辞書のような オブジェクトから構成できます。

{{% alert %}}
loss や accuracy などの出力メトリクスや従属変数を保存するには、`wandb.Run.config` ではなく `wandb.Run.log()` を使ってください。
{{% /alert %}}



## 実験設定を用意する
通常、設定はトレーニングスクリプトの冒頭で定義します。ただし、機械学習 のワークフローはさまざまなので、トレーニングスクリプトの最初で定義する必要はありません。

config の変数名では、ピリオド (`.`) の代わりにダッシュ (`-`) かアンダースコア (`_`) を使ってください。
	
スクリプトでルート直下以外の `wandb.Run.config` のキーにアクセスする場合は、属性アクセス構文 `config.key.value` ではなく、辞書アクセス構文 `["key"]["value"]` を使ってください。

以下では、実験設定を定義する一般的なシナリオを紹介します。

### 初期化時に設定を渡す
スクリプトの冒頭で `wandb.init()` API を呼び出す際に辞書を渡すと、W&B の Run としてデータを同期・ログするバックグラウンドプロセスが生成されます。

次のコードスニペットは、設定値を持つ Python の辞書を定義し、W&B Run を初期化するときにその辞書を引数として渡す方法を示します。

```python
import wandb

# config 用の辞書オブジェクトを定義
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化するときに config の辞書を渡す
with wandb.init(project="config_example", config=config) as run:
    ...
```

入れ子の辞書を `config` に渡すと、W&B はドットで名前をフラット化します。

Python の他の辞書と同様に、辞書から値を取り出せます:

```python
# キーをインデックスにして値にアクセス
hidden_layer_sizes = run.config["hidden_layer_sizes"]
kernel_sizes = run.config["kernel_sizes"]
activation = run.config["activation"]

# Python の辞書の get() メソッド
hidden_layer_sizes = run.config.get("hidden_layer_sizes")
kernel_sizes = run.config.get("kernel_sizes")
activation = run.config.get("activation")
```

{{% alert %}}
Developer Guide や例では、読みやすさのために、設定値を個別の変数にコピーしています。この手順は任意です。
{{% /alert %}}

### argparse で設定する
argparse オブジェクトで設定を指定できます。[argparse](https://docs.python.org/3/library/argparse.html) は argument parser の略で、Python 3.2 以降の標準ライブラリモジュールです。コマンドライン引数の柔軟性とパワーを活かしたスクリプトを簡単に書けます。

これは、コマンドラインから起動するスクリプトの結果をトラッキングするのに便利です。

以下の Python スクリプトは、実験の config を定義・設定するためのパーサーオブジェクトの定義方法を示します。デモのために、トレーニングループを模擬する関数 `train_one_epoch` と `evaluate_one_epoch` を用意しています:

```python
# config_experiment.py
import argparse
import random

import numpy as np
import wandb


# トレーニングと評価のデモコード
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B の Run を開始
    with wandb.init(project="config_example", config=args) as run:
        # config の辞書から値を取り出して
        # 読みやすさのために変数に格納
        lr = run.config["learning_rate"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # トレーニングを模擬し、W&B に値をログする
        for epoch in np.arange(1, epochs):
            train_acc, train_loss = train_one_epoch(epoch, lr, bs)
            val_acc, val_loss = evaluate_one_epoch(epoch)

            run.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="Learning rate"
    )

    args = parser.parse_args()
    main(args)
```
### スクリプト全体で設定する
スクリプト全体で、config オブジェクトにパラメータを追加することもできます。次のコードスニペットは、config オブジェクトに新しいキーと値のペアを追加する方法を示します:

```python
import wandb

# config 用の辞書オブジェクトを定義
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化するときに config の辞書を渡す
with wandb.init(project="config_example", config=config) as run:
    # W&B を初期化した後に config を更新
    run.config["dropout"] = 0.2
    run.config.epochs = 4
    run.config["batch_size"] = 32
```

複数の値をまとめて更新できます:

```python
run.config.update({"lr": 0.1, "channels": 16})
```

### Run が終了した後に設定を更新する
[W&B Public API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) を使って、完了した run の config を更新できます。 

API には、あなたの Entity、Project 名、そして run の ID を渡す必要があります。これらの情報は Run オブジェクト、または [W&B App]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で確認できます:

```python
with wandb.init() as run:
    ...

# Run がこのスクリプトまたはノートブックから開始されている場合は、
# 以下の値を Run オブジェクトから取得できます。そうでなければ、W&B App の UI からコピーできます。
username = run.entity
project = run.project
run_id = run.id

# api.run() は wandb.init() とは異なる型のオブジェクトを返すことに注意してください。
api = wandb.Api()
api_run = api.run(f"{username}/{project}/{run_id}")
api_run.config["bar"] = 32
api_run.update()
```




## `absl.FLAGS`


[`absl` フラグ](https://abseil.io/docs/python/guides/flags)を渡すこともできます。

```python
flags.DEFINE_string("model", None, "model to run")  # 名前、デフォルト、ヘルプ

run.config.update(flags.FLAGS)  # absl のフラグを config に追加
```

## ファイルベースの config
実行スクリプトと同じディレクトリーに `config-defaults.yaml` という名前のファイルを置くと、run はそのファイルで定義されたキーと値のペアを自動的に取り込み、`wandb.Run.config` に渡します。 

次のコードスニペットは、`config-defaults.yaml` のサンプル YAML ファイルです:

```yaml
batch_size:
  desc: Size of each mini-batch
  value: 32
```

`wandb.init` の `config` 引数で新しい値を設定することで、`config-defaults.yaml` から自動ロードされたデフォルト値を上書きできます。例:

```python
import wandb

# config-defaults.yaml をカスタム値で上書き
with wandb.init(config={"epochs": 200, "batch_size": 64}) as run:
    ...
```

`config-defaults.yaml` 以外の設定ファイルを読み込むには、`--configs command-line` 引数を使ってファイルのパスを指定します:

```bash
python train.py --configs other-config.yaml
```

### ファイルベースの config のユースケース例
たとえば、run に関するメタデータを含む YAML ファイルがあり、さらに Python スクリプト内にハイパーパラメーターの辞書があるとします。どちらもネストした `config` オブジェクトに保存できます:

```python
hyperparameter_defaults = dict(
    dropout=0.5,
    batch_size=100,
    learning_rate=0.001,
)

config_dictionary = dict(
    yaml=my_yaml_file,
    params=hyperparameter_defaults,
)

with wandb.init(config=config_dictionary) as run:
    ...
```

## TensorFlow v1 のフラグ

TensorFlow のフラグを `wandb.Run.config` オブジェクトに直接渡せます。

```python
with wandb.init() as run:
    run.config.epochs = 4

    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    run.config.update(flags.FLAGS)  # TensorFlow のフラグを config に追加
```