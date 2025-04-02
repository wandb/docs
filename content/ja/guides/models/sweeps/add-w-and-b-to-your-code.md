---
title: Add W&B (wandb) to your code
description: W&B を Python コード スクリプトまたは Jupyter Notebook に追加します。
menu:
  default:
    identifier: ja-guides-models-sweeps-add-w-and-b-to-your-code
    parent: sweeps
weight: 2
---

W&B Python SDK をスクリプトまたは Jupyter Notebook に追加する方法は多数あります。以下に、W&B Python SDK を独自のコードに統合する方法の「ベストプラクティス」の例を示します。

### 元のトレーニングスクリプト

Python スクリプトに次のコードがあるとします。ここでは、典型的なトレーニングループを模倣する `main` という関数を定義します。エポックごとに、トレーニングデータセットと検証データセットで精度と損失が計算されます。これらの値は、この例の目的のためにランダムに生成されます。

ここでは、ハイパーパラメータの値を格納する `config` という 辞書 を定義しました。セルの最後に、`main` 関数を呼び出して、モックトレーニングコードを実行します。

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

# config variable with hyperparameter values
config = {"lr": 0.0001, "bs": 16, "epochs": 5}

def main():
    # Note that we define values from `wandb.config`
    # instead of defining hard values
    # ハードコードされた値を定義する代わりに、`wandb.config` から値を定義することに注意してください
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

### W&B Python SDK を使用したトレーニングスクリプト

次のコード例は、W&B Python SDK をコードに追加する方法を示しています。CLI で W&B Sweep ジョブを開始する場合は、[CLI] タブを確認してください。Jupyter Notebook または Python スクリプト内で W&B Sweep ジョブを開始する場合は、[Python SDK] タブを確認してください。

<Tabs>
  <TabItem value="python script or notebook" label="Python スクリプトまたはノートブック">
  W&B Sweep を作成するために、次のコード例を追加しました。

1. Weights & Biases Python SDK をインポートします。
2. キーと値のペアで sweep configuration を定義する 辞書 オブジェクトを作成します。以下の例では、バッチサイズ (`batch_size`)、エポック数 (`epochs`)、および学習率 (`lr`) の ハイパーパラメータ は、各 sweep 中に変化します。sweep configuration の作成方法の詳細については、[sweep configuration の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。
3. sweep configuration 辞書 を [`wandb.sweep`]({{< relref path="/ref/python/sweep.md" lang="ja" >}}) に渡します。これにより、sweep が初期化されます。これにより、sweep ID (`sweep_id`) が返されます。sweep の初期化方法の詳細については、[sweep の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})を参照してください。
4. [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) API を使用して、[W&B Run]({{< relref path="/ref/python/run.md" lang="ja" >}}) としてデータを同期およびログ記録するバックグラウンド プロセス を生成します。
5. (オプション) ハードコードされた値を定義する代わりに、`wandb.config` から値を定義します。
6. 最適化する メトリクス を [`wandb.log`]({{< relref path="/ref/python/log.md" lang="ja" >}}) でログに記録します。configuration で定義された メトリクス をログに記録する必要があります。configuration 辞書 (この例では `sweep_configuration`) 内で、`val_acc` 値を最大化するように sweep を定義しました。
7. [`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}}) API 呼び出しで sweep を開始します。sweep ID、sweep が実行する関数の名前 (`function=main`) を指定し、試行する run の最大数を 4 (`count=4`) に設定します。W&B Sweep の開始方法の詳細については、[sweep agent の開始]({{< relref path="./start-sweep-agents.md" lang="ja" >}})を参照してください。

```python
import wandb
import numpy as np
import random

# Define sweep config
# sweep configuration を定義する
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

# Initialize sweep by passing in config.
# (Optional) Provide a name of the project.
# configuration を渡して sweep を初期化します。
# （オプション） Project の名前を指定します。
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")


# Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a
# model and return metric
# `wandb.config` から ハイパーパラメータ 値を取得し、それらを使用して
# モデルをトレーニングし、メトリクス を返すトレーニング関数を定義する
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    run = wandb.init()

    # note that we define values from `wandb.config`
    # instead of defining hard values
    # ハードコードされた値を定義する代わりに、`wandb.config` から値を定義することに注意してください
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

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


# Start sweep job.
# sweep ジョブを開始します。
wandb.agent(sweep_id, function=main, count=4)
```

  </TabItem>
  <TabItem value="cli" label="CLI">

W&B Sweep を作成するには、まず YAML configuration ファイルを作成します。configuration ファイルには、sweep で探索する ハイパーパラメータ が含まれています。以下の例では、バッチサイズ (`batch_size`)、エポック数 (`epochs`)、および学習率 (`lr`) の ハイパーパラメータ は、各 sweep 中に変化します。

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
    values: [16,32,64]
  lr:
    min: 0.0001
    max: 0.1
  epochs:
    values: [5, 10, 15]
```

W&B Sweep configuration の作成方法の詳細については、[sweep configuration の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。

YAML ファイルの `program` キーに Python スクリプトの名前を指定する必要があることに注意してください。

次に、次のコード例を追加します。

1. Wieghts & Biases Python SDK (`wandb`) と PyYAML (`yaml`) をインポートします。PyYAML は、YAML configuration ファイルを読み込むために使用されます。
2. configuration ファイルを読み込みます。
3. [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) API を使用して、[W&B Run]({{< relref path="/ref/python/run.md" lang="ja" >}}) としてデータを同期およびログ記録するバックグラウンド プロセス を生成します。config オブジェクトを config パラメータに渡します。
4. ハードコードされた値を使用する代わりに、`wandb.config` から ハイパーパラメータ 値を定義します。
5. 最適化する メトリクス を [`wandb.log`]({{< relref path="/ref/python/log.md" lang="ja" >}}) でログに記録します。configuration で定義された メトリクス をログに記録する必要があります。configuration 辞書 (この例では `sweep_configuration`) 内で、`val_acc` 値を最大化するように sweep を定義しました。

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
    # Set up your default hyperparameters
    # デフォルトのハイパーパラメータを設定する
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # Note that we define values from `wandb.config`
    # instead of  defining hard values
    # ハードコードされた値を定義する代わりに、`wandb.config` から値を定義することに注意してください
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

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


# Call the main function.
# main 関数を呼び出します。
main()
```

CLI に移動します。CLI 内で、sweep agent が試行する run の最大数を設定します。この手順はオプションです。次の例では、最大数を 5 に設定します。

```bash
NUM=5
```

次に、[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドで sweep を初期化します。YAML ファイルの名前を指定します。オプションで、project フラグ (`--project`) の Project の名前を指定します。

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

これにより、sweep ID が返されます。sweep の初期化方法の詳細については、[sweep の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})を参照してください。

sweep ID をコピーし、次の コードスニペット の `sweepID` を置き換えて、[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドで sweep ジョブを開始します。

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

sweep ジョブの開始方法の詳細については、[sweep ジョブの開始]({{< relref path="./start-sweep-agents.md" lang="ja" >}})を参照してください。

  </TabItem>
</Tabs>

## メトリクス をログに記録する際の考慮事項

sweep configuration で指定した メトリクス を明示的に W&B にログに記録してください。サブディレクトリー内で sweep の メトリクス をログに記録しないでください。

たとえば、次の疑似コードを考えてみましょう。ユーザーは 検証 損失 (`"val_loss": loss`) をログに記録したいと考えています。最初に、値を 辞書 に渡します。ただし、`wandb.log` に渡される 辞書 は、辞書 内のキーと値のペアに明示的にアクセスしません。

```python
# Import the W&B Python Library and log into W&B
# W&B Python ライブラリをインポートし、W&B にログインします
import wandb
import random

def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    val_metrics = {"val_loss": loss, "val_acc": acc}
    return val_metrics


def main():
    wandb.init(entity="<entity>", project="my-first-sweep")
    val_metrics = train()
    # Incorrect. You must explicitly access the
    # key-value pair in the dictionary
    # 間違いです。 辞書 のキーと値のペアに明示的にアクセスする必要があります
    # 正しいメトリクス の記録方法については、次のコードブロックを参照してください
    wandb.log({"val_loss": val_metrics})


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

代わりに、Python 辞書 内のキーと値のペアに明示的にアクセスします。たとえば、次のコードは、辞書 を `wandb.log` メソッドに渡すときに、キーと値のペアを指定します。

```python title="train.py"
# Import the W&B Python Library and log into W&B
# W&B Python ライブラリをインポートし、W&B にログインします
import wandb
import random


def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    val_metrics = {"val_loss": loss, "val_acc": acc}
    return val_metrics


def main():
    wandb.init(entity="<entity>", project="my-first-sweep")
    val_metrics = train()
    wandb.log({"val_loss", val_metrics["val_loss"]})


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