---
title: 実験を設定する
description: 辞書のようなオブジェクトを使って、実験の設定を保存します
menu:
  default:
    identifier: ja-guides-models-track-config
    parent: experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

run の `config` プロパティを使って、トレーニング設定を保存しましょう。  
- ハイパーパラメーター
- データセット名やモデルタイプなどの入力設定
- その他、実験用の独立変数全般

`wandb.Run.config` プロパティを使うことで、実験の解析や将来の再現がとても簡単になります。W&B App 上で設定値ごとにグループ化したり、異なる W&B Run の設定を比較したり、各トレーニング設定が結果に与える効果を評価できます。`config` プロパティは、複数の辞書（dictionary）型から構成できる、辞書のようなオブジェクトです。

{{% alert %}}
出力メトリクスや loss, accuracy のような従属変数は、`wandb.Run.config` ではなく `wandb.Run.log()` で保存してください。
{{% /alert %}}



## 実験設定のセットアップ
設定は通常、トレーニングスクリプトの冒頭で定義します。ただし、機械学習ワークフローによっては、最初に設定を定義する必要はありません。

config 変数名にはピリオド（`.`）ではなく、ダッシュ（`-`）またはアンダースコア（`_`）を使いましょう。
	
`wandb.Run.config` のルート以下のキーにアクセスする場合、属性アクセスの `config.key.value` ではなく、辞書アクセスの `["key"]["value"]` を使ってください。

次のセクションでは、いくつかの典型的な実験設定方法について説明します。

### 初期化時に設定を渡す
スクリプトの冒頭で `wandb.init()` API を呼ぶ際、辞書型の設定を渡すことで、W&B Run としてデータの同期・ログを行うバックグラウンドプロセスが生成されます。

以下のコードスニペットは、設定値を Python の辞書として定義し、それを W&B Run 初期化時に引数で渡す方法を示しています。

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

# W&B を初期化する時に config 辞書を渡す
with wandb.init(project="config_example", config=config) as run:
    ...
```

入れ子の辞書を `config` として渡すと、W&B は名前をドット区切りでフラット化します。

値へのアクセスは Python の他の辞書オブジェクトと同じように行えます。

```python
# キーを添え字として値を取得
hidden_layer_sizes = run.config["hidden_layer_sizes"]
kernel_sizes = run.config["kernel_sizes"]
activation = run.config["activation"]

# Python の get() メソッド
hidden_layer_sizes = run.config.get("hidden_layer_sizes")
kernel_sizes = run.config.get("kernel_sizes")
activation = run.config.get("activation")
```

{{% alert %}}
この Developer Guide や例では、設定値を個別の変数にコピーしていますが、これは読みやすさのためで必須ではありません。
{{% /alert %}}

### argparse で設定を渡す
argparse オブジェクトを使って設定を渡すことも可能です。[argparse](https://docs.python.org/3/library/argparse.html) は、コマンドライン引数を柔軟かつ簡単に扱える、Python 3.2 以降標準搭載のライブラリです。

コマンドラインから実行するスクリプトの結果をトラッキングしたい場合などに便利です。

以下の Python スクリプトは、parser オブジェクトを定義して実験設定をセットする例です。`train_one_epoch` と `evaluate_one_epoch` 関数は、このデモのためのトレーニングループをシミュレートしています。

```python
# config_experiment.py
import argparse
import random

import numpy as np
import wandb


# トレーニング・評価のデモ用コード
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # W&B Run の開始
    with wandb.init(project="config_example", config=args) as run:
        # config 辞書から値を変数へ保存（読みやすさのため）
        lr = run.config["learning_rate"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # トレーニングをシミュレートして W&B へ値を記録
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
### スクリプト全体で適宜 config を追加する
スクリプト内のさまざまなタイミングで、config にパラメータを追加することもできます。下記のコードスニペットは、config オブジェクトに新たなキーと値のペアを追加する方法です。

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

# W&B 初期化時に config を渡す
with wandb.init(project="config_example", config=config) as run:
    # W&B 初期化後に config を更新
    run.config["dropout"] = 0.2
    run.config.epochs = 4
    run.config["batch_size"] = 32
```

複数の値をまとめてアップデートすることも可能です。

```python
run.config.update({"lr": 0.1, "channels": 16})
```

### Run 終了後に設定を編集する
完了した run の config を編集するには、[W&B Public API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) を使います。

API には entity・project 名・run の ID を指定する必要があります。これらは Run オブジェクトや [W&B App]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) から取得できます。

```python
with wandb.init() as run:
    ...

# Run オブジェクトや W&B App UI から以下を取得
username = run.entity
project = run.project
run_id = run.id

# api.run() は wandb.init() と異なる型のオブジェクトを返します
api = wandb.Api()
api_run = api.run(f"{username}/{project}/{run_id}")
api_run.config["bar"] = 32
api_run.update()
```




## `absl.FLAGS`


[`absl` flags](https://abseil.io/docs/python/guides/flags) も config に渡すことができます。

```python
flags.DEFINE_string("model", None, "モデル名")  # name, default, help

run.config.update(flags.FLAGS)  # absl の flag を config に追加
```

## ファイルベースの Config
run スクリプトと同じディレクトリーに `config-defaults.yaml` というファイルを設置すると、そのファイルで定義されたキーと値が自動的に検出され、`wandb.Run.config` に渡されます。

以下はサンプルの `config-defaults.yaml` ファイルです。

```yaml
batch_size:
  desc: 各ミニバッチのサイズ
  value: 32
```

`config-defaults.yaml` で自動ロードされたデフォルト値は、`wandb.init` の `config` 引数で任意の値を設定することで上書きできます。

```python
import wandb

# config-defaults.yaml の設定をカスタム値で上書き
with wandb.init(config={"epochs": 200, "batch_size": 64}) as run:
    ...
```

`config-defaults.yaml` 以外の設定ファイルを使いたい場合は、`--configs` コマンドライン引数でファイルパスを指定してください。

```bash
python train.py --configs other-config.yaml
```

### ファイルベース config のユースケース例
YAML ファイルで run 用メタデータを持ち、Python スクリプト内でハイパーパラメーターの辞書を用意している場合、両方を入れ子の `config` オブジェクトとして保存できます。

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

## TensorFlow v1 flags

TensorFlow の flags も、そのまま `wandb.Run.config` オブジェクトに渡せます。

```python
with wandb.init() as run:
    run.config.epochs = 4

    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data")
    flags.DEFINE_integer("batch_size", 128, "バッチサイズ。")
    run.config.update(flags.FLAGS)  # tensorflow の flag を config として追加
```