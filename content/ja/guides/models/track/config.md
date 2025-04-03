---
title: Configure experiments
description: 実験 の 設定 を保存するには、 辞書 のような オブジェクト を使用します
menu:
  default:
    identifier: ja-guides-models-track-config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

`config` プロパティを使用すると、トレーニングの設定を保存できます。
- ハイパーパラメータ
- データセット名やモデルタイプなどの入力設定
- 実験のその他の独立変数

`run.config` プロパティを使用すると、実験の分析や将来の作業の再現が容易になります。W&B App で設定値ごとにグループ化したり、異なる W&B の run の設定を比較したり、各トレーニング設定がアウトプットに与える影響を評価したりできます。`config` プロパティは、複数の辞書のようなオブジェクトから構成できる辞書のようなオブジェクトです。

{{% alert %}}
損失や精度などの出力メトリクスや従属変数を保存するには、`run.config` の代わりに `run.log` を使用してください。
{{% /alert %}}

## 実験設定の構成

通常、設定はトレーニングスクリプトの最初に定義されます。ただし、機械学習のワークフローは異なる場合があるため、トレーニングスクリプトの最初に設定を定義する必要はありません。

config 変数名では、ピリオド（`.`）の代わりにダッシュ（`-`）またはアンダースコア（`_`）を使用してください。

スクリプトがルートより下の `run.config` キーにアクセスする場合は、属性アクセス構文 `config.key.value` の代わりに辞書アクセス構文 `["key"]["value"]` を使用してください。

以下のセクションでは、実験設定を定義するさまざまな一般的なシナリオについて概説します。

### 初期化時に設定を構成する
W&B Run としてデータを同期およびログ記録するためのバックグラウンド プロセスを生成するために、`wandb.init()` API を呼び出すときに、スクリプトの先頭で辞書を渡します。

次のコードスニペットは、設定値を持つ Python 辞書を定義する方法と、W&B Run を初期化するときにその辞書を引数として渡す方法を示しています。

```python
import wandb

# config 辞書オブジェクトを定義します
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化するときに config 辞書を渡します
with wandb.init(project="config_example", config=config) as run:
    ...
```

`config` としてネストされた辞書を渡すと、W&B はドットを使用して名前をフラット化します。

Python の他の辞書にアクセスするのと同じように、辞書から値にアクセスします。

```python
# インデックス値としてキーを使用して値にアクセスします
hidden_layer_sizes = run.config["hidden_layer_sizes"]
kernel_sizes = run.config["kernel_sizes"]
activation = run.config["activation"]

# Python 辞書 get() メソッド
hidden_layer_sizes = run.config.get("hidden_layer_sizes")
kernel_sizes = run.config.get("kernel_sizes")
activation = run.config.get("activation")
```

{{% alert %}}
開発者ガイドと例全体を通して、設定値を個別の変数にコピーしています。このステップはオプションです。これは、読みやすさのために行われています。
{{% /alert %}}

### argparse で設定を構成する
argparse オブジェクトで設定を構成できます。[argparse](https://docs.python.org/3/library/argparse.html)（引数パーサーの略）は、Python 3.2 以降の標準ライブラリモジュールであり、コマンドライン引数のすべての柔軟性と機能を活用するスクリプトを簡単に作成できます。

これは、コマンドラインから起動されたスクリプトの結果を追跡するのに役立ちます。

次の Python スクリプトは、パーサーオブジェクトを定義して実験設定を定義および設定する方法を示しています。関数 `train_one_epoch` および `evaluate_one_epoch` は、このデモンストレーションの目的でトレーニングループをシミュレートするために提供されています。

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
    # W&B Run を開始します
    with wandb.init(project="config_example", config=args) as run:
        # config 辞書から値にアクセスし、読みやすくするために変数に格納します
        lr = run.config["learning_rate"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # トレーニングをシミュレートし、値を W&B に記録します
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
        "-e", "--epochs", type=int, default=50, help="トレーニングエポック数"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="学習率"
    )

    args = parser.parse_args()
    main(args)
```
### スクリプト全体で設定を構成する
スクリプト全体で config オブジェクトにパラメーターを追加できます。次のコードスニペットは、新しいキーと値のペアを config オブジェクトに追加する方法を示しています。

```python
import wandb

# config 辞書オブジェクトを定義します
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化するときに config 辞書を渡します
with wandb.init(project="config_example", config=config) as run:
    # W&B を初期化した後に config を更新します
    run.config["dropout"] = 0.2
    run.config.epochs = 4
    run.config["batch_size"] = 32
```

一度に複数の値を更新できます。

```python
run.config.update({"lr": 0.1, "channels": 16})
```

### Run の完了後に設定を構成する
[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使用して、完了した Run の config を更新します。

API には、エンティティ、プロジェクト名、および Run の ID を指定する必要があります。これらの詳細は、Run オブジェクトまたは [W&B App UI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で確認できます。

```python
with wandb.init() as run:
    ...

# 現在のスクリプトまたはノートブックから開始された場合は、Run オブジェクトから次の値を見つけるか、W&B App UI からコピーできます。
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
flags.DEFINE_string("model", None, "実行するモデル")  # name, default, help

run.config.update(flags.FLAGS)  # absl フラグを config に追加します
```

## ファイルベースの Configs
`config-defaults.yaml` という名前のファイルを Run スクリプトと同じディレクトリーに配置すると、Run はファイルで定義されたキーと値のペアを自動的に取得し、`run.config` に渡します。

次のコードスニペットは、サンプルの `config-defaults.yaml` YAML ファイルを示しています。

```yaml
batch_size:
  desc: 各ミニバッチのサイズ
  value: 32
```

`config-defaults.yaml` から自動的にロードされたデフォルト値をオーバーライドするには、`wandb.init` の `config` 引数で更新された値を設定します。次に例を示します。

```python
import wandb

# カスタム値を渡して config-defaults.yaml をオーバーライドします
with wandb.init(config={"epochs": 200, "batch_size": 64}) as run:
    ...
```

`config-defaults.yaml` 以外の構成ファイルをロードするには、`--configs コマンドライン` 引数を使用し、ファイルへのパスを指定します。

```bash
python train.py --configs other-config.yaml
```

### ファイルベースの Configs のユースケース例
Run のメタデータを含む YAML ファイルと、Python スクリプトにハイパーパラメーターの辞書があるとします。ネストされた `config` オブジェクトに両方を保存できます。

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

TensorFlow フラグを `wandb.config` オブジェクトに直接渡すことができます。

```python
with wandb.init() as run:
    run.config.epochs = 4

    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data")
    flags.DEFINE_integer("batch_size", 128, "バッチサイズ")
    run.config.update(flags.FLAGS)  # tensorflow フラグを config として追加します
```