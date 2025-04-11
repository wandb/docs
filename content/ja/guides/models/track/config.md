---
title: 実験を設定する
description: 実験の設定を保存するために辞書のようなオブジェクトを使用します。
menu:
  default:
    identifier: ja-guides-models-track-config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

`run` の `config` プロパティを使用して、トレーニング設定を保存します:
- ハイパーパラメーター
- データセット名やモデルタイプなどの入力設定
- 実験のためのその他の独立変数

`run.config` プロパティを使用すると、実験を簡単に分析し、将来的に作業を再現できます。 W&B アプリで設定値ごとにグループ化し、さまざまな W&B Run の設定を比較し、各トレーニング設定が出力にどのように影響するかを評価できます。`config` プロパティは、複数の辞書のようなオブジェクトから構成できる辞書のようなオブジェクトです。

{{% alert %}}
出力メトリクスや損失や精度のような従属変数を保存するには、`run.config` ではなく `run.log` を使用してください。
{{% /alert %}}

## 実験の設定を行う
設定は通常、トレーニングスクリプトの最初に定義されます。ただし、機械学習ワークフローは異なる場合があるため、トレーニングスクリプトの最初に設定を定義する必要はありません。

設定変数名にはピリオド (`.`) の代わりにダッシュ (`-`) またはアンダースコア (`_`) を使用してください。

スクリプトが `run.config` キーをルート以下でアクセスする場合は、属性アクセス構文 `config.key.value` の代わりに、辞書アクセス構文 `["key"]["value"]` を使用してください。

以下のセクションでは、実験の設定を定義する一般的なシナリオをいくつか示します。

### 初期化時に設定を行う
`wandb.init()` API を呼び出して、バックグラウンドプロセスを生成し、W&B Run としてデータを同期してログに記録する際に、スクリプトの最初に辞書を渡します。

次に示すコードスニペットは、設定値を持つ Python の辞書を定義し、その辞書を引数として W&B Run を初期化する方法を示しています。

```python
import wandb

# 設定辞書オブジェクトを定義する
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化する際に設定辞書を渡す
with wandb.init(project="config_example", config=config) as run:
    ...
```

ネストされた辞書を `config` として渡す場合、W&B はドットを使用して名前をフラットにします。

Python の他の辞書にアクセスする方法と同様に、辞書から値にアクセスすることができます:

```python
# インデックス値としてキーを使用して値にアクセスする
hidden_layer_sizes = run.config["hidden_layer_sizes"]
kernel_sizes = run.config["kernel_sizes"]
activation = run.config["activation"]

# Python の辞書 get() メソッド
hidden_layer_sizes = run.config.get("hidden_layer_sizes")
kernel_sizes = run.config.get("kernel_sizes")
activation = run.config.get("activation")
```

{{% alert %}}
開発者ガイドと例全体で、設定値を別の変数にコピーします。このステップは任意です。読みやすさのために行われます。
{{% /alert %}}

### argparse を使用して設定を行う
argparse オブジェクトで設定を行うことができます。[argparse](https://docs.python.org/3/library/argparse.html) は、引数パーサの短縮形で、Python 3.2 以降の標準ライブラリモジュールであり、コマンドライン引数の柔軟性と強力さを活かしたスクリプトの記述を容易にします。

コマンドラインから起動されるスクリプトからの結果を追跡するのに便利です。

次の Python スクリプトは、実験設定を定義および設定するためのパーサーオブジェクトを定義する方法を示しています。`train_one_epoch` および `evaluate_one_epoch` 関数は、このデモの目的でトレーニングループをシミュレートするために提供されています:

```python
# config_experiment.py
import argparse
import random

import numpy as np
import wandb


# トレーニングと評価デモのコード
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B Run を開始する
    with wandb.init(project="config_example", config=args) as run:
        # 設定辞書から値をアクセスして、
        # 可読性のために変数に格納する
        lr = run.config["learning_rate"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # トレーニングをシミュレートし、値を W&B にログする
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

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="バッチサイズ")
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="トレーニングエポックの数"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="学習率"
    )

    args = parser.parse_args()
    main(args)
```
### スクリプト全体で設定を行う
スクリプト全体で、設定オブジェクトにさらにパラメータを追加できます。次に示すコードスニペットでは、設定オブジェクトに新しいキーと値のペアを追加する方法を示しています:

```python
import wandb

# 設定辞書オブジェクトを定義する
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化する際に設定辞書を渡す
with wandb.init(project="config_example", config=config) as run:
    # W&B を初期化した後に設定を更新する
    run.config["dropout"] = 0.2
    run.config.epochs = 4
    run.config["batch_size"] = 32
```

複数の値を一度に更新できます:

```python
run.config.update({"lr": 0.1, "channels": 16})
```

### Run終了後に設定を行う
[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}})を使用して、完了済みの run の設定を更新できます。

API に対して、エンティティ、プロジェクト名、および run の ID を提供する必要があります。これらの詳細は Run オブジェクトや [W&B App UI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で確認できます:

```python
with wandb.init() as run:
    ...

# Run オブジェクトから以下の値を見つけます。
# これが現在のスクリプトまたはノートブックから初期化された場合、または W&B アプリUIからそれらをコピーできます。
username = run.entity
project = run.project
run_id = run.id

# api.run() は wandb.init() と異なるタイプのオブジェクトを返すことに注意してください。
api = wandb.Api()
api_run = api.run(f"{username}/{project}/{run_id}")
api_run.config["bar"] = 32
api_run.update()
```

## `absl.FLAGS`

[`absl` flags](https://abseil.io/docs/python/guides/flags) を渡すこともできます。

```python
flags.DEFINE_string("model", None, "model to run")  # name, default, help

run.config.update(flags.FLAGS)  # absl flags を設定に追加します
```

## ファイルベースの設定
`config-defaults.yaml` という名前のファイルを run スクリプトと同じディレクトリーに配置すると、run はファイルに定義されたキーと値のペアを自動的に取得し、それを `run.config` に渡します。

以下のコードスニペットは、サンプルの `config-defaults.yaml` YAML ファイルを示しています:

```yaml
batch_size:
  desc: サイズごとのミニバッチ
  value: 32
```

`config-defaults.yaml` から自動的に読み込まれたデフォルト値を、新しい値を `wandb.init` の `config` 引数で設定して上書きできます。たとえば:

```python
import wandb

# 独自の値を渡して config-defaults.yaml を上書きします
with wandb.init(config={"epochs": 200, "batch_size": 64}) as run:
    ...
```

`config-defaults.yaml` 以外の設定ファイルを読み込むには、`--configs command-line` 引数を使用してファイルのパスを指定します:

```bash
python train.py --configs other-config.yaml
```

### ファイルベースの設定のユースケースの例
Run のメタデータを含む YAML ファイルがあり、Python スクリプトでハイパーパラメーターの辞書を持っているとします。それらの両方をネストされた `config` オブジェクトに保存できます:

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

## TensorFlow v1 フラグ

TensorFlow のフラグを `wandb.config` オブジェクトに直接渡すことができます。

```python
with wandb.init() as run:
    run.config.epochs = 4

    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data")
    flags.DEFINE_integer("batch_size", 128, "バッチサイズ")
    run.config.update(flags.FLAGS)  # TensorFlow のフラグを設定に追加します
```