---
title: 実験を設定
description: 辞書のようなオブジェクトを使って、実験の設定を保存します
menu:
  default:
    identifier: config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

run の `config` プロパティを使って、トレーニングの設定を保存できます：
- ハイパーパラメーター
- データセット名やモデルタイプなどの入力設定
- 実験のためのその他の独立変数

`wandb.Run.config` プロパティを使うことで、実験の分析や将来の再現が簡単になります。W&B App では設定値でグループ化したり、異なる W&B run の設定を比較したり、それぞれのトレーニング設定が出力にどう影響したかを評価することができます。`config` プロパティは辞書のようなオブジェクトで、複数の辞書型オブジェクトから構成することもできます。

{{% alert %}}
loss や accuracy などの出力メトリクス（従属変数）を保存したい場合は、`wandb.Run.config` ではなく `wandb.Run.log()` を使ってください。
{{% /alert %}}



## 実験設定のセットアップ
設定は通常、トレーニングスクリプトの最初で定義します。ただし、機械学習ワークフローはさまざまなので、必ずしもスクリプトの冒頭で設定を定義する必要はありません。

config の変数名にはピリオド（`.`）の代わりにダッシュ（`-`）またはアンダースコア（`_`）を使いましょう。

また、`wandb.Run.config` のルート以下のキーにアクセスする場合は、属性アクセス構文 `config.key.value` ではなく、辞書アクセス構文 `["key"]["value"]` を使ってください。

以下のセクションでは、よく使われる実験設定の方法を紹介します。

### 初期化時に設定を渡す
スクリプトの最初で辞書型の設定を渡し、`wandb.init()` API を呼び出すと、W&B Run としてデータの同期とログを行うバックグラウンドプロセスが生成されます。

以下のコードスニペットは、設定値の Python 辞書を定義し、それを W&B Run の初期化時に引数として渡す例です。

```python
import wandb

# config 辞書オブジェクトを定義
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化時に config を渡す
with wandb.init(project="config_example", config=config) as run:
    ...
```

ネストされた辞書を `config` として渡すと、W&B は名前をドット区切りでフラット化します。

辞書の値には、Python で他の辞書にアクセスする方法と同様にアクセスできます。

```python
# キーをインデックスとして値にアクセス
hidden_layer_sizes = run.config["hidden_layer_sizes"]
kernel_sizes = run.config["kernel_sizes"]
activation = run.config["activation"]

# Python の辞書の get() メソッド
hidden_layer_sizes = run.config.get("hidden_layer_sizes")
kernel_sizes = run.config.get("kernel_sizes")
activation = run.config.get("activation")
```

{{% alert %}}
Developer Guide やサンプルコード内では、設定値をそれぞれの変数にコピーしていますが、これは可読性のためで必須ではありません。
{{% /alert %}}

### argparse で設定を渡す
argparse オブジェクトを使って experiment の設定を渡すこともできます。[argparse](https://docs.python.org/3/library/argparse.html) は Python 3.2 以降の標準ライブラリで、コマンドライン引数を柔軟かつ強力に扱えるようにします。

コマンドラインから起動されるスクリプトの結果を追跡したいときに便利です。

以下の Python スクリプトは、設定用パーサオブジェクトの定義・実験設定・W&B への保存方法の例です。`train_one_epoch`、`evaluate_one_epoch` 関数はデモ用のトレーニングループをシミュレートしています。

```python
# config_experiment.py
import argparse
import random

import numpy as np
import wandb


# トレーニングと評価のデモ関数
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B Run を開始
    with wandb.init(project="config_example", config=args) as run:
        # config 辞書から値を取得、可読性のため変数に格納
        lr = run.config["learning_rate"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # トレーニングをシミュレートし、W&B に値をログ
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
        "-e", "--epochs", type=int, default=50, help="トレーニングのエポック数"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="学習率"
    )

    args = parser.parse_args()
    main(args)
```
### スクリプト内で逐次 config をセットする
スクリプトの途中で config オブジェクトに新しいパラメータを追加できます。次のコードスニペットは、config オブジェクトへ新しいキーと値のペアを追加する方法を示しています。

```python
import wandb

# config 辞書オブジェクトを定義
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化時に config を渡す
with wandb.init(project="config_example", config=config) as run:
    # W&B 初期化後に config を更新
    run.config["dropout"] = 0.2
    run.config.epochs = 4
    run.config["batch_size"] = 32
```

複数の値をまとめて更新することもできます。

```python
run.config.update({"lr": 0.1, "channels": 16})
```

### Run 完了後に設定を変更する
[W&B Public API]({{< relref "/ref/python/public-api/index.md" >}}) を使って、完了した run の config を更新できます。

API には entity、プロジェクト名、run の ID を渡す必要があります。これらの情報は Run オブジェクトや [W&B App]({{< relref "/guides/models/track/workspaces.md" >}}) 上で確認できます。

```python
with wandb.init() as run:
    ...

# Run オブジェクト または App UI で次の値を取得
username = run.entity
project = run.project
run_id = run.id

# api.run() は wandb.init() とは異なる型のオブジェクトを返します
api = wandb.Api()
api_run = api.run(f"{username}/{project}/{run_id}")
api_run.config["bar"] = 32
api_run.update()
```




## `absl.FLAGS`


[`absl` フラグ](https://abseil.io/docs/python/guides/flags)も渡すことができます。

```python
flags.DEFINE_string("model", None, "モデルの名称")  # name, default, help

run.config.update(flags.FLAGS)  # absl フラグを config に追加
```

## ファイルベースの config
run スクリプトと同じディレクトリーに `config-defaults.yaml` という名前のファイルを置くと、そのファイルに定義されたキーと値のペアが自動的に `wandb.Run.config` に反映されます。

以下のコードスニペットは `config-defaults.yaml` の記述例です。

```yaml
batch_size:
  desc: 各ミニバッチのサイズ
  value: 32
```

`config-defaults.yaml` で自動的に読み込まれたデフォルト値は、`wandb.init` の `config` 引数で値を指定することで上書きできます。例えば：

```python
import wandb

# 独自の値を渡して config-defaults.yaml を上書き
with wandb.init(config={"epochs": 200, "batch_size": 64}) as run:
    ...
```

`config-defaults.yaml` 以外の設定ファイルを読み込む場合は、`--configs` コマンドライン引数でファイルのパスを指定してください。

```bash
python train.py --configs other-config.yaml
```

### ファイルベース config のユースケース例
YAML ファイルに run 用のメタデータ、Python スクリプト内でハイパーパラメータの辞書を持つ場合、両方をネストした `config` オブジェクトで保存できます。

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

TensorFlow のフラグもそのまま `wandb.Run.config` オブジェクトに追加できます。

```python
with wandb.init() as run:
    run.config.epochs = 4

    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data")
    flags.DEFINE_integer("batch_size", 128, "バッチサイズ。")
    run.config.update(flags.FLAGS)  # tensorflow フラグを config に追加
```