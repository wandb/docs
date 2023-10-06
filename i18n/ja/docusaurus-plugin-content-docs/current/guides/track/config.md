---
description: Use a dictionary-like object to save your experiment configuration
displayed_sidebar: ja
---

# 実験を設定する

<head>
  <title>機械学習実験を設定する</title>
</head>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/config-colab)

以下のようなトレーニング設定を保存するために`wandb.config`オブジェクトを使用してください:
- ハイパーパラメーター
- データセット名やモデルタイプなどの入力設定
- 実験に関連する他の独立変数

`wandb.config`属性は、実験を分析し、将来的に作業を再現するのに便利です。W&Bアプリで設定値をグループ分けしたり、異なるW&B Runsの設定を比較したり、異なるトレーニング設定が出力にどのように影響するかを見ることができます。Run の `config` 属性は辞書のようなオブジェクトであり、多くの辞書のようなオブジェクトから構築できます。

:::info
損失や精度のような従属変数や出力指標は、代わりに `wandb.log`で保存する必要があります。
:::

## 実験設定を設定する
設定は通常、トレーニングスクリプトの冒頭で定義されます。ただし、機械学習のワークフローは異なる場合があるため、トレーニングスクリプトの冒頭で設定を定義する必要はありません。

:::caution
構成変数名にドットを使用しないことをお勧めします。代わりにダッシュまたはアンダースコアを使用してください。スクリプトが `wandb.config` のルート以下のキーにアクセスする場合は、属性アクセス構文 `config.key.foo` の代わりに、辞書アクセス構文 `["key"]["foo"]` を使用してください。
:::
以下のセクションでは、実験設定を定義する方法について、さまざまな一般的なシナリオを概説しています。

### 初期化時に設定を行う
スクリプトの最初に、`wandb.init()` APIを呼び出してディクショナリを渡し、W&B Runとしてデータを同期およびログするバックグラウンドプロセスを生成します。

以下のコードスニペットでは、設定値を持つPythonディクショナリを定義し、W&B Runを初期化する際にそのディクショナリを引数として渡す方法を示しています。

```python
import wandb

# configディクショナリオブジェクトを定義する
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&Bを初期化するときにconfigディクショナリを渡す
run = wandb.init(project="config_example", config=config)
```

:::info
`wandb.config()` にはネストしたディクショナリを渡すことができます。W&Bは、W&Bバックエンドでドットを使って名前をフラット化します。
:::

ディクショナリから値を取得する方法は、Pythonの他のディクショナリから値を取得する方法と同様です。

```python
# キーをインデックス値として値にアクセスする
hidden_layer_sizes = wandb.config["hidden_layer_sizes"]
kernel_sizes = wandb.config["kernel_sizes"]
activation = wandb.config["activation"]

# Pythonの辞書型のget()メソッド
hidden_layer_sizes = wandb.config.get("hidden_layer_sizes")
kernel_sizes = wandb.config.get("kernel_sizes")
activation = wandb.config.get("activation")
```

:::note
開発者ガイドや例題では、設定値を別の変数にコピーしています。このステップはオプションです。これは可読性のために行われています。
:::

### argparseを使って設定を行う
argparseオブジェクトで設定を行うことができます。[argparse](https://docs.python.org/3/library/argparse.html)は、Python 3.2以降の標準ライブラリモジュールで、コマンドライン引数の柔軟性とパワーを活用したスクリプトを簡単に作成できます。

これは、コマンドラインから起動されたスクリプトの結果をトラッキングするのに便利です。

次のPythonスクリプトは、実験設定を定義し設定するために、パーサーオブジェクトを定義する方法を示しています。関数`train_one_epoch`と`evaluate_one_epoch`は、このデモンストレーションの目的のためにトレーニングループをシミュレートするために提供されています。

```python
# config_experiment.py
import wandb
import argparse
import numpy as np
import random


# トレーニングと評価デモコード
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B Runを開始
    run = wandb.init(project="config_example", config=args)

    # config辞書から値を取得し、読みやすさのために変数に保存する
    lr = wandb.config["learning_rate"]
    bs = wandb.config["batch_size"]
    epochs = wandb.config["epochs"]

    # トレーニングとW&Bへの値のロギングをシミュレーションする
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
```

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="バッチサイズ")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="トレーニングエポック数")
    parser.add_argument("-lr", "--learning_rate", type=int, default=0.001, help="学習率")


args = parser.parse_args()

main(args)
```
### スクリプト全体で設定を設定する
スクリプト全体でconfigオブジェクトにさらにパラメータを追加することができます。以下のコードスニペットでは、configオブジェクトに新しいキーと値のペアを追加する方法を示しています。

```python
import wandb

# config辞書オブジェクトを定義
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&Bを初期化するときにconfig辞書を渡す
run = wandb.init(project="config_example", config=config)

# W&Bを初期化した後でconfigを更新する
wandb.config["dropout"] = 0.2
wandb.config.epochs = 4
wandb.config["batch_size"] = 32
```
一度に複数の値を更新することができます:

```python
wandb.init(config={"epochs": 4, "batch_size": 32})
# その後
wandb.config.update({"lr": 0.1, "channels": 16})
```
### Runが終了した後に設定を行う
[W&B Public API](../../ref/python/public-api/README.md)を使って、Runが終了した後に設定（またはRun全体からの他の事項）を更新します。これは、Run中に値をログに記録し忘れた場合に特に便利です。

Runが終了した後に設定を更新するには、`entity`、`project name`、および`Run ID`を提供してください。これらの値は、Runオブジェクト自体`wandb.run`または[W&B App UI](../app/intro.md)から直接取得できます。

```python
api = wandb.Api()

# runオブジェクトまたはW&Bアプリから直接属性にアクセス
username = wandb.run.entity
project = wandb.run.project
run_id = wandb.run.id

run = api.run(f"{username}/{project}/{run_id}")
run.config["bar"] = 32
run.update()
```

<!-- ## `argparse.Namespace` -->


<!-- [`argparse`](https://docs.python.org/3/library/argparse.html)で返される引数を渡すことができます。これは、コマンドラインから異なるハイパーパラメータ値を素早くテストするのに便利です。 -->


<!-- ```python -->
<!-- wandb.init(config={"lr": 0.1}) -->
<!-- wandb.config.epochs = 4 -->
parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                     help='トレーニング用の入力バッチサイズ（デフォルト：8）')
args = parser.parse_args()
wandb.config.update(args) # 引数をすべて設定変数として追加
<!-- ``` -->

## `absl.FLAGS`


[`absl`フラグ](https://abseil.io/docs/python/guides/flags)も渡すことができます。

```python
flags.DEFINE_string("model", None, "実行するモデル")  # 名前、デフォルト、ヘルプ

wandb.config.update(flags.FLAGS)  # abslフラグをconfigに追加
```

## ファイルベースの設定
キーと値のペアは、`config-defaults.yaml`という名前のファイルを作成することで自動的に`wandb.config`に渡されます。

以下のコードスニペットは、サンプルの`config-defaults.yaml` YAMLファイルを示しています。

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
自動的に渡される`config-defaults.yaml`による値を上書きすることができます。そのためには、`wandb.init`の`config`引数に値を渡してください。
また、コマンドライン引数`--configs`で別の設定ファイルを読み込むこともできます。

### ファイルベースの設定の例
例えば、runのメタデータを含むYAMLファイルと、Pythonスクリプト内のハイパーパラメーターのディクショナリがあるとします。両方をネストされた`config`オブジェクトに保存できます。

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

## TensorFlow v1フラグ

`wandb.config`オブジェクトに直接TensorFlowのフラグを渡すことができます。

```python
wandb.init()
wandb.config.epochs = 4

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data")
flags.DEFINE_integer("batch_size", 128, "Batch size.")
wandb.config.update(flags.FLAGS)  # add tensorflow flags as config
```
<!-- ## データセット識別子 -->

データセットに一意の識別子（ハッシュや他の識別子）を付けることで、`wandb.config` を使って実験への入力としてトラッキングできます。

```yaml
wandb.config.update({"dataset": "ab131"})
```

<!-- ## 設定ファイルの更新 -->

パブリックAPIを使って、`config`ファイルに値を追加することができます。これは、runが終了した後でも行うことができます。

```python
import wandb

api = wandb.Api()
run = api.run("username/project/run_id")
run.config["foo"] = 32
run.update()
```

<!-- ## キー・バリューペア -->

`wandb.config` に任意のキーと値のペアをログに記録することができます。これらは、トレーニング中のモデルのタイプごとに異なることができます。例：`wandb.config.update({"my_param": 10, "learning_rate": 0.3, "model_architecture": "B"})`。