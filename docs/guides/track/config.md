---
description: 辞書のようなオブジェクトを使用して、実験の設定を保存する
displayed_sidebar: default
---


# Configure Experiments

<head>
  <title>機械学習実験の設定</title>
</head>

[**Colabノートブックで試してみる**](http://wandb.me/config-colab)

`wandb.config` オブジェクトを使用して以下のようなトレーニング設定を保存します:
- ハイパーパラメーター
- データセット名やモデルタイプなどの入力設定
- その他、実験に使用する独立変数

`wandb.config` 属性を使用することで、実験を簡単に分析し、将来的に再現性を持たせることができます。W&Bアプリで設定値に基づいてグループ化したり、異なるW&B Runsの設定を比較したり、異なるトレーニング設定が出力にどのように影響するかを確認できます。Runの `config` 属性は辞書のようなオブジェクトであり、多くの辞書のようなオブジェクトから構築することができます。

:::info
従属変数（損失や精度など）や出力メトリクスは、代わりに `wandb.log` を使用して保存する必要があります。
:::

## 実験設定のセットアップ
設定は通常、トレーニングスクリプトの最初に定義されます。ただし、機械学習のワークフローは多様であるため、トレーニングスクリプトの最初に設定を定義する必要はありません。

:::caution
設定変数名にドットを使用するのは避けることをお勧めします。代わりにダッシュやアンダースコアを使用してください。スクリプトが `wandb.config` キーをルート以下でアクセスする場合は、属性アクセス構文 `config.key.foo` の代わりに、辞書アクセス構文 `["key"]["foo"]` を使用してください。
:::

以下のセクションでは、実験の設定を定義するさまざまな一般的なシナリオを説明します。

### 初期化時に設定を行う
スクリプトの最初に辞書を渡して、W&B Runとしてデータを同期およびログするバックグラウンドプロセスを生成するために `wandb.init()` API を呼び出します。

以下のコードスニペットは、設定値を含むPython辞書を定義し、その辞書を引数として渡してW&B Runを初期化する方法を示しています。

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

# W&Bを初期化するときに設定辞書を渡す
run = wandb.init(project="config_example", config=config)
```

:::info
ネストされた辞書を `wandb.config()` に渡すことができます。W&BはW&Bバックエンドで名前をドットでフラット化します。
:::

Pythonの他の辞書と同様に、辞書から値をアクセスします：

```python
# インデックス値としてキーを使用して値にアクセスする
hidden_layer_sizes = wandb.config["hidden_layer_sizes"]
kernel_sizes = wandb.config["kernel_sizes"]
activation = wandb.config["activation"]

# Pythonの辞書get()メソッド
hidden_layer_sizes = wandb.config.get("hidden_layer_sizes")
kernel_sizes = wandb.config.get("kernel_sizes")
activation = wandb.config.get("activation")
```

:::note
開発者向けガイドや例全体で、設定値を別の変数にコピーしています。このステップはオプションですが、可読性のために行われています。
:::

### argparseで設定を行う
argparseオブジェクトを使用して設定を行うことができます。[argparse](https://docs.python.org/3/library/argparse.html) は、引数パーサの略で、Python 3.2以降の標準ライブラリモジュールであり、コマンドライン引数の柔軟性とパワーを活用するスクリプトを簡単に書くことができます。

これは、コマンドラインから起動されるスクリプトの結果を追跡する際に便利です。

以下のPythonスクリプトは、パーサオブジェクトを定義し、実験設定を定義および設定する方法を示しています。このデモのために、トレーニングループをシミュレートするために `train_one_epoch` および `evaluate_one_epoch` 関数が提供されています：

```python
# config_experiment.py
import wandb
import argparse
import numpy as np
import random


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
    # W&B Runを開始する
    run = wandb.init(project="config_example", config=args)

    # 設定辞書から値をアクセスし、読みやすさのために変数に保存する
    lr = wandb.config["learning_rate"]
    bs = wandb.config["batch_size"]
    epochs = wandb.config["epochs"]

    # トレーニングをシミュレートし、値をW&Bにログする
    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        wandb.log(
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
        "-e", "--epochs", type=int, default=50, help="トレーニングエポック数"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="学習率"
    )

    args = parser.parse_args()
    main(args)
```
### スクリプト全体で設定を行う
スクリプト全体で設定オブジェクトにパラメーターを追加できます。以下のコードスニペットでは、設定オブジェクトに新しいキーと値のペアを追加する方法を示しています：

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

# W&Bを初期化するときに設定辞書を渡す
run = wandb.init(project="config_example", config=config)

# W&Bを初期化した後に設定を更新する
wandb.config["dropout"] = 0.2
wandb.config.epochs = 4
wandb.config["batch_size"] = 32
```
複数の値を一度に更新することもできます：

```python
wandb.init(config={"epochs": 4, "batch_size": 32})
# 後で
wandb.config.update({"lr": 0.1, "channels": 16})
```

### Runが終了した後に設定を行う
完了したRunから設定（またはその他の情報）を更新するために [W&B Public API](../../ref/python/public-api/README.md) を使用します。これは、Runの間に値を記録し忘れた場合に特に便利です。

Runが終了した後に設定を更新するには、`entity`、`project name`、および `Run ID` を提供します。これらの値はRunオブジェクト（`wandb.run`）または [W&BアプリのUI](../app/intro.md)から直接確認できます：

```python
api = wandb.Api()

# RunオブジェクトまたはW&Bアプリから直接属性にアクセスする
username = wandb.run.entity
project = wandb.run.project
run_id = wandb.run.id

run = api.run(f"{username}/{project}/{run_id}")
run.config["bar"] = 32
run.update()
```

## `absl.FLAGS`

[`absl` フラグ](https://abseil.io/docs/python/guides/flags) を渡すこともできます。

```python
flags.DEFINE_string("model", None, "実行するモデル")  # name, default, help

wandb.config.update(flags.FLAGS)  # abslフラグを設定に追加
```

## ファイルベースの設定
キーと値のペアは、自動的に `wandb.config` に渡されます。`config-defaults.yaml` というファイルを作成するだけです。

以下のコードスニペットは、サンプルの `config-defaults.yaml` YAMLファイルを示しています：

```yaml
# config-defaults.yaml
# サンプルの設定デフォルトファイル
epochs:
  desc: トレーニングするエポック数
  value: 100
batch_size:
  desc: 各ミニバッチのサイズ
  value: 32
```
`config-defaults.yaml` によって自動的に渡される値を上書きすることができます。そのために、`wandb.init` の `config` 引数に値を渡します。

`--configs` コマンドライン引数を使用して、異なる設定ファイルを読み込むこともできます。

### ファイルベースの設定のユースケース例
例えば、いくつかのメタデータを含むYAMLファイルと、そのPythonスクリプト内のハイパーパラメーターの辞書があるとします。両方をネストされた `config` オブジェクトに保存することができます：

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

wandb.init(config=config_dictionary)
```

## TensorFlow v1 フラグ

TensorFlowのフラグを直接 `wandb.config` オブジェクトに渡すことができます。

```python
wandb.init()
wandb.config.epochs = 4

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data")
flags.DEFINE_integer("batch_size", 128, "バッチサイズ。")
wandb.config.update(flags.FLAGS)  # TensorFlowフラグを設定に追加
```
