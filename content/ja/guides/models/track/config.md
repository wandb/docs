---
title: Configure experiments
description: 辞書 のようなオブジェクトを使って、 実験 の 設定 を保存します
menu:
  default:
    identifier: ja-guides-models-track-config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

`wandb.config` オブジェクトを使用すると、次のようなトレーニング設定を保存できます。
- ハイパーパラメーター
- データセット名やモデルタイプなどの入力設定
- 実験 のその他の独立変数

`wandb.config` 属性を使用すると、 実験 を分析し、将来の作業を再現することが容易になります。W&B App で 設定 値でグループ化したり、異なる W&B の Run の 設定 を比較したり、異なるトレーニング設定が 出力 にどのように影響するかを確認したりできます。Run の `config` 属性は 辞書 のような オブジェクト で、多くの 辞書 のような オブジェクト から構築できます。

{{% alert %}}
従属変数 (損失 や 精度 など) または 出力 メトリクス は、代わりに `wandb.log` で保存する必要があります。
{{% /alert %}}

## 実験 設定 のセットアップ

設定 は通常、トレーニングスクリプトの最初に定義されます。ただし、 機械学習 の ワークフロー は異なる場合があるため、トレーニングスクリプトの最初に 設定 を定義する必要はありません。

{{% alert color="secondary" %}}
config 変数名にドットを使用しないことをお勧めします。代わりに、ダッシュまたはアンダースコアを使用してください。スクリプトがルート以下の `wandb.config` キーに アクセス する場合は、属性 アクセス 構文 `config.key.foo` の代わりに 辞書 アクセス 構文 `["key"]["foo"]` を使用してください。
{{% /alert %}}

次のセクションでは、 実験 設定 を定義するさまざまな一般的なシナリオについて説明します。

### 初期化時に 設定 を設定する

スクリプトの先頭で `wandb.init()` API を呼び出すときに 辞書 を渡して、W&B の Run としてデータを同期および ログ に記録するバックグラウンド プロセス を生成します。

次の コードスニペット は、 設定 値を持つ Python 辞書 を定義する方法と、W&B の Run を初期化するときにその 辞書 を 引数 として渡す方法を示しています。

```python
import wandb

# config 辞書 オブジェクト を定義します
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化するときに config 辞書 を渡します
run = wandb.init(project="config_example", config=config)
```

{{% alert %}}
ネストされた 辞書 を `wandb.config()` に渡すことができます。W&B は、W&B バックエンドでドットを使用して名前をフラット化します。
{{% /alert %}}

Python の他の 辞書 に アクセス するのと同様に、 辞書 から 値 に アクセス します。

```python
# キーを インデックス 値として使用して 値 に アクセス します
hidden_layer_sizes = wandb.config["hidden_layer_sizes"]
kernel_sizes = wandb.config["kernel_sizes"]
activation = wandb.config["activation"]

# Python 辞書 get() メソッド
hidden_layer_sizes = wandb.config.get("hidden_layer_sizes")
kernel_sizes = wandb.config.get("kernel_sizes")
activation = wandb.config.get("activation")
```

{{% alert %}}
開発者 ガイド と 例 全体で、 設定 値 を個別の変数にコピーします。このステップはオプションです。読みやすくするために行われます。
{{% /alert %}}

### argparse で 設定 を設定する

argparse オブジェクト を使用して 設定 を設定できます。[argparse](https://docs.python.org/3/library/argparse.html) (argument parser の略) は、Python 3.2 以降の標準 ライブラリ モジュールであり、 コマンドライン 引数 のすべての柔軟性とパワーを利用するスクリプトを簡単に記述できます。

これは、 コマンドライン から 起動 されたスクリプトの 結果 を追跡するのに役立ちます。

次の Python スクリプト は、parser オブジェクト を定義して 実験 config を定義および設定する方法を示しています。関数 `train_one_epoch` と `evaluate_one_epoch` は、この デモ の目的でトレーニング ループ をシミュレートするために提供されています。

```python
# config_experiment.py
import wandb
import argparse
import numpy as np
import random


# トレーニング と 評価 の デモ コード
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B の Run を開始します
    run = wandb.init(project="config_example", config=args)

    # config 辞書 から 値 に アクセス し、読みやすくするために変数に格納します
    lr = wandb.config["learning_rate"]
    bs = wandb.config["batch_size"]
    epochs = wandb.config["epochs"]

    # トレーニング をシミュレートし、W&B に 値 を ログ に記録します
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

    parser.add_argument("-b", "--batch_size", type=int, default=32, help=" バッチサイズ ")
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help=" トレーニング エポック の数"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help=" 学習率 "
    )

    args = parser.parse_args()
    main(args)
```
### スクリプト全体で 設定 を設定する

スクリプト全体で config オブジェクト に パラメータ を追加できます。次の コードスニペット は、新しいキーと 値 のペアを config オブジェクト に追加する方法を示しています。

```python
import wandb

# config 辞書 オブジェクト を定義します
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# W&B を初期化するときに config 辞書 を渡します
run = wandb.init(project="config_example", config=config)

# W&B を初期化した後で config を更新します
wandb.config["dropout"] = 0.2
wandb.config.epochs = 4
wandb.config["batch_size"] = 32
```
一度に複数の 値 を更新できます。

```python
wandb.init(config={"epochs": 4, "batch_size": 32})
# 後で
wandb.config.update({"lr": 0.1, "channels": 16})
```

### Run の完了後に 設定 を設定する

[W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を使用して、Run の完了後に config (または完全な Run のその他すべて) を更新します。これは、Run 中に 値 を ログ に記録するのを忘れた場合に特に役立ちます。

Run の完了後に 設定 を更新するには、`entity`、`project name`、および `Run ID` を指定します。これらの 値 は、Run オブジェクト 自体 `wandb.run` から、または [W&B App UI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) から直接見つけることができます。

```python
api = wandb.Api()

# Run オブジェクト から、または W&B App から属性に直接 アクセス します
username = wandb.run.entity
project = wandb.run.project
run_id = wandb.run.id

run = api.run(f"{username}/{project}/{run_id}")
run.config["bar"] = 32
run.update()
```

## `absl.FLAGS`

[`absl` flags](https://abseil.io/docs/python/guides/flags) を渡すこともできます。

```python
flags.DEFINE_string("model", None, " 実行 する モデル ")  # name, default, help

wandb.config.update(flags.FLAGS)  # absl flags を config に追加します
```

## ファイルベースの Configs
`config-defaults.yaml` という名前のファイルを run スクリプト と同じ ディレクトリー に配置すると、run はファイルで定義されたキーと 値 のペアを自動的に取得し、`wandb.config` に渡します。

次の コードスニペット は、サンプル `config-defaults.yaml` YAML ファイルを示しています。

```yaml
batch_size:
  desc: 各ミニ バッチ のサイズ
  value: 32
```

`wandb.init` の `config` 引数で更新された 値 を設定することにより、`config-defaults.yaml` から自動的に ロード された デフォルト 値 をオーバーライドできます。次に 例 を示します。

```python
import wandb
# カスタム 値 を渡して config-defaults.yaml をオーバーライドします
wandb.init(config={"epochs": 200, "batch_size": 64})
```

`config-defaults.yaml` 以外の 設定 ファイルを ロード するには、`--configs コマンドライン` 引数 を使用して、ファイルへのパスを指定します。

```bash
python train.py --configs other-config.yaml
```

### ファイルベースの configs の ユースケース の 例
run の メタデータ を含む YAML ファイルと、Python スクリプト に ハイパーパラメーター の 辞書 があるとします。ネストされた `config` オブジェクト に両方を保存できます。

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

## TensorFlow v1 flags

TensorFlow flags を `wandb.config` オブジェクト に直接渡すことができます。

```python
wandb.init()
wandb.config.epochs = 4

flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/data")
flags.DEFINE_integer("batch_size", 128, " バッチサイズ 。")
wandb.config.update(flags.FLAGS)  # tensorflow flags を config として追加します
```