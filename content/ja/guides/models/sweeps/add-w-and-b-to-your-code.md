---
title: コードに W&B (wandb) を追加する
description: あなたの Python コードスクリプトや Jupyter ノートブックに W&B を追加しましょう。
menu:
  default:
    identifier: ja-guides-models-sweeps-add-w-and-b-to-your-code
    parent: sweeps
weight: 2
---

W&B Python SDK をスクリプトやノートブックに追加する方法はいくつもあります。このセクションでは、「ベストプラクティス」とされる W&B Python SDK の組み込み例をご紹介します。自分のコードへ自然に統合できるようにしましょう。

### 元のトレーニングスクリプト

例えば、以下のような Python スクリプトがあるとします。ここでは `main` という名前の関数を定義して、一般的なトレーニングループを模倣しています。各エポックごとに、トレーニングデータと検証データにおける精度と損失が計算されますが、この例ではランダムな値を利用しています。

ハイパーパラメータの値は `config` という辞書に格納しています。セルの最後で `main` 関数を呼び出し、模擬トレーニングのコードを実行します。

```python
import random
import numpy as np

def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss

def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss

# ハイパーパラメーターの値を格納した config 変数
config = {"lr": 0.0001, "bs": 16, "epochs": 5}

def main():
    # `wandb.Run.config` から値を取得していることに注意
    # 定数ではなく config から取得します
    lr = config["lr"]
    bs = config["bs"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "training loss:", val_loss)        
```

### W&B Python SDK を使ったトレーニングスクリプト

以下のコード例では、どのようにして W&B Python SDK を自分のコードへ組み込むかを示しています。CLI から W&B Sweep ジョブを開始したい場合は CLI タブを、Jupyter ノートブックや Python スクリプトから Sweep ジョブを開始したい場合は Python SDK タブをご確認ください。

{{< tabpane text=true >}} {{% tab header="Python スクリプトまたはノートブック" %}}  
W&B Sweep を作成するには、次のような手順をコードに追加します：

1. W&B Python SDK をインポートします。
2. 辞書オブジェクトを作成し、キーと値のペアで sweep 設定を定義します。下記例では、バッチサイズ（`batch_size`）、エポック数（`epochs`）、学習率（`lr`）のハイパーパラメーターを sweep の中で変更しています。詳細は [スイープ設定を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) をご覧ください。
3. sweep 設定の辞書を [`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ja" >}}) に渡して sweep を初期化します。これにより sweep ID（`sweep_id`）が返却されます。詳細は [スイープの初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) をご確認ください。
4. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) API を利用して、バックグラウンドプロセスによって [W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) としてデータを同期・ログできるようにします。
5. （オプション）固定値を直接記述するのではなく、`wandb.config` から値を取得して利用しましょう。
6. [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}) で最適化したいメトリクスをログします。設定で定義したメトリクスを必ずログしてください。この例の設定辞書（`sweep_configuration`）内では、`val_acc` を最大化するよう sweep を定義しています。
7. sweep の開始には [`wandb.agent`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ja" >}}) API を呼び出します。sweep ID と、sweep が実行する関数名 (`function=main`)、最大で試行する run 数 (`count=4`) を指定します。詳細は [スイープエージェントの開始]({{< relref path="./start-sweep-agents.md" lang="ja" >}}) をご覧ください。

```python
import wandb
import numpy as np
import random


# `wandb.Run.config` からハイパーパラメーターを取得し、
# それらを使ってモデルを学習・メトリクスを返す関数
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


# スイープ設定を定義した辞書
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

# （オプション）Project 名を指定
project = "my-first-sweep"

def main():
    # `with` 文を使うことで run を自動的に終了できます。
    # これは、各 run の最後で `run.finish()` を呼ぶのと等価です
    with wandb.init(project=project) as run:

        # ハイパーパラメーターを `wandb.Run.config` から取得
        lr = run.config["lr"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # トレーニングループを実行し、パフォーマンス値を W&B に記録
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
    # config 辞書を渡してスイープを初期化
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    # スイープジョブを開始
    wandb.agent(sweep_id, function=main, count=4)

```

{{% alert %}} このコードスニペットは、`with` 文を使って [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) API を初期化し、バックグラウンドプロセスで [W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) としてデータを同期・ログする方法を示しています。これにより、記録した値をアップロードし終えたら run が正しく終了されます。別の方法として、`wandb.init()` をトレーニングスクリプトの最初、`wandb.Run.finish()` を最後に呼ぶ方法もあります。
{{% /alert %}}

{{% /tab %}} {{% tab header="CLI" %}}

W&B Sweep を作成するには、まず YAML の設定ファイルを用意します。このファイルに、Sweep で探索したいハイパーパラメーターを記載します。以下の例では、バッチサイズ（`batch_size`）、エポック数（`epochs`）、学習率（`lr`）のハイパーパラメーターを Sweep を通じて変更します。

```yaml
# config.yaml
program: train.py
method: random
name: sweep
metric:
  goal: maximize
  name: val_acc
parameters:
  batch_size:
    values: [16, 32, 64]
  lr:
    min: 0.0001
    max: 0.1
  epochs:
    values: [5, 10, 15]
```

W&B Sweep 設定の詳細については [スイープ設定を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

YAML ファイル内の `program` キーには、Python スクリプトの名前を指定する必要があります。

次に、以下の操作をコード例に追加します：

1. W&B Python SDK（`wandb`）と PyYAML (`yaml`) をインポートします。PyYAML は YAML 設定ファイルを読み込むために使います。
2. 設定ファイルを読み込みます。
3. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) API を使い、[W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) としてバックグラウンドでデータを同期・ログするプロセスを作成します。config オブジェクトを config 引数として渡します。
4. ハイパーパラメータは固定値ではなく、`wandb.Run.config` から取得・定義します。
5. [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}) で最適化したいメトリクスをログします。設定に定義されているメトリクスを必ず記録してください。この例では `val_acc` を最大化するよう sweep を設定しています。

```python
import wandb
import yaml
import random
import numpy as np


def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    # デフォルトのハイパーパラメータを設定
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    with wandb.init(config=config) as run:
        for epoch in np.arange(1, run.config['epochs']):
            train_acc, train_loss = train_one_epoch(epoch, run.config['lr'], run.config['batch_size'])
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

# main 関数を呼び出す
main()
```

CLI で sweep エージェントに試行させる run の最大回数を指定できます（省略可）。この例では最大回数を 5 に設定します。

```bash
NUM=5
```

続いて [`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドで sweep を初期化します。YAML ファイル名を指定し、オプションで Project 名を `--project` フラグで与えます：

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

このコマンドで sweep ID が返されます。Sweep の初期化手順について詳しくは [スイープの初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

返却された sweep ID をコピーし、以下の例の `sweepID` を適宜置き換えて、[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドで sweep ジョブを開始します：

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

詳細については [スイープジョブの開始]({{< relref path="./start-sweep-agents.md" lang="ja" >}}) をご確認ください。

{{% /tab %}} {{< /tabpane >}}

## メトリクスをログする際の注意

Sweep のメトリクスは必ず明示的に W&B へログする必要があります。サブディレクトリー内など、推奨されない場所にメトリクスを記録しないよう注意してください。

例えば、以下の擬似コードを考えてみましょう。ユーザーが検証損失（`"val_loss": loss`）をログしたいとします。まず値を辞書に入れていますが、`wandb.Run.log()` に渡す際に辞書のキー・値ペアに明示的にアクセスしていません：

```python
# W&B Python ライブラリのインポートと W&B へのログイン
import wandb
import random

def train():
    # トレーニング・検証のメトリクスをシミュレート
    offset = random.random() / 5
    epoch = 5  # エポック値をシミュレート
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    return loss, acc


def main():
    with wandb.init(entity="<entity>", project="my-first-sweep") as run:
        val_loss, val_acc = train()
        # 正しくない例。辞書内のキー・値ペアに
        # 明示的にアクセスする必要があります
        # 正しいやり方は次のコードブロックを参照
        run.log({"val_loss": val_loss, "val_acc": val_acc})

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 設定辞書でスイープを初期化
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

# スイープジョブを開始
wandb.agent(sweep_id, function=main, count=10)
```

正しい方法は、Python の辞書からキー・値ペアを明確に指定して `wandb.Run.log()` メソッドに渡します。例えば次の例では、辞書で指定したキー・値ペアを使ってメトリクスをログしています：

```python title="train.py"
# W&B Python ライブラリのインポートと W&B へのログイン
import wandb
import random


def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    return loss, acc


def main():
    with wandb.init(entity="<entity>", project="my-first-sweep") as run:
        # 正しい例。メトリクスをログする時に辞書の
        # キー・値ペアを明確に指定しています
        val_loss, val_acc = train()
        run.log({"val_loss": val_loss, "val_acc": val_acc})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```