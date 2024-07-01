---
description: あなたの Python コードスクリプトまたは Jupyter ノートブックに W&B を追加します。
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Add W&B to your code

<head>
  <title>Add W&B to your Python code</title>
</head>

W&BのPython SDKをスクリプトやJupyter Notebookに追加する方法はいくつもあります。以下に、W&B Python SDKを自分のコードに統合するための「ベストプラクティス」の例を示します。

### オリジナルのトレーニングスクリプト

次のコードがJupyter NotebookのセルやPythonスクリプトにあると仮定します。典型的なトレーニングループを模倣する`main`という関数を定義します。各エポックのトレーニングと検証データセットに対して、精度と損失を計算します。この例の目的のために、値はランダムに生成されます。

ハイパーパラメーターの値を格納するために`config`という辞書が定義されています（15行目）。セルの最後では、`main`関数を呼び出して、模擬トレーニングコードを実行します。

```python showLineNumbers
# train.py
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


config = {"lr": 0.0001, "bs": 16, "epochs": 5}


def main():
    # Note that we define values from `wandb.config`
    # instead of defining hard values
    lr = config["lr"]
    bs = config["bs"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "training loss:", val_loss)


# Call the main function.
main()
```

### W&B Python SDKを使用したトレーニングスクリプト

次のコード例は、W&B Python SDKをコードに追加する方法を示しています。W&B SweepジョブをCLIで開始する場合は、CLIタブを参照してください。Jupyter NotebookやPythonスクリプト内でW&B Sweepジョブを開始する場合は、Python SDKタブを参照してください。

<Tabs defaultValue="script" values={[
    {label: 'Python script or Jupyter Notebook', value: 'script'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="script">
  W&B Sweepを作成するには、以下をコード例に追加しました：

1. 行1：Weights & Biases Python SDKをインポートします。
2. 行6：キーと値のペアでsweep設定を定義する辞書オブジェクトを作成します。以下の例では、バッチサイズ (`batch_size`)、エポック (`epochs`)、学習率 (`lr`)のハイパーパラメーターが各sweepごとに変化します。sweep設定の作成方法については、[Define sweep configuration](./define-sweep-configuration.md)をご参照ください。
3. 行19：sweep設定辞書を[`wandb.sweep`](../../ref/python/sweep)に渡します。これによりsweepが初期化されます。これによりsweep ID (`sweep_id`)が返されます。sweepの初期化方法については、[Initialize sweeps](./initialize-sweeps.md)をご参照ください。
4. 行33：[`wandb.init()`](../../ref/python/init.md) APIを使用して背景プロセスを生成し、データを同期して[W&B Run](../../ref/python/run.md)としてログを取ります。
5. 行37-39：（オプション）ハードコーディングされた値の代わりに`wandb.config`から値を定義します。
6. 行45：[`wandb.log`](../../ref/python/log.md)を使用して最適化したいメトリクスを記録します。設定で定義したメトリクスを記録する必要があります。この例の設定辞書 (`sweep_configuration`)では、`val_acc`値を最大化するようにsweepを定義しました。
7. 行54：[`wandb.agent`](../../ref/python/agent.md) API呼び出しでsweepを開始します。sweep ID（19行目）、実行する関数の名前（`function=main`）、試行回数の最大数を4（`count=4`）に設定します。W&B Sweepの開始方法については、[Start sweep agents](./start-sweep-agents.md)をご参照ください。

```python showLineNumbers
import wandb
import numpy as np
import random

# Define sweep config
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
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")


# Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a
# model and return metric
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
wandb.agent(sweep_id, function=main, count=4)
```
  </TabItem>
  <TabItem value="cli">

W&B Sweepを作成するには、まずYAML設定ファイルを作成します。設定ファイルには、sweepで探索したいハイパーパラメーターが含まれています。以下の例では、バッチサイズ (`batch_size`)、エポック (`epochs`)、学習率 (`lr`)のハイパーパラメーターが各sweepごとに変化します。

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

W&B Sweep設定の作成方法については、[Define sweep configuration](./define-sweep-configuration.md)をご参照ください。

YAMLファイルの`program`キーにはPythonスクリプトの名前を必ず提供してください。

次に、以下をコード例に追加します：

1. 行1-2：Weights & Biases Python SDK (`wandb`)およびPyYAML (`yaml`)をインポートします。PyYAMLはYAML設定ファイルを読み込むために使用されます。
2. 行18：設定ファイルを読み込みます。
3. 行21：[`wandb.init()`](../../ref/python/init.md) APIを使用して背景プロセスを生成し、データを同期して[W&B Run](../../ref/python/run.md)としてログを取ります。configオブジェクトをconfigパラメーターに渡します。
4. 行25 - 27：ハードコーディングされた値を使用する代わりに、`wandb.config`からハイパーパラメーターの値を定義します。
5. 行33-39：[`wandb.log`](../../ref/python/log.md)を使用して最適化したいメトリクスを記録します。設定で定義したメトリクスを記録する必要があります。この例の設定辞書 (`sweep_configuration`)では、`val_acc`値を最大化するようにsweepを定義しました。

```python showLineNumbers
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
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # Note that we define values from `wandb.config`
    # instead of defining hard values
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
main()
```

CLIに移動します。CLI内で、sweep agentが試行する最大の実行数を設定します。このステップはオプションです。以下の例では、最大数を5に設定しています。

```bash
NUM=5
```

次に、[`wandb sweep`](../../ref/cli/wandb-sweep.md)コマンドでsweepを初期化します。YAMLファイルの名前を提供します。オプションとして、プロジェクトの名前をプロジェクトフラグ (`--project`) で提供します：

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

これによりsweep IDが返されます。sweepの初期化方法については、[Initialize sweeps](./initialize-sweeps.md)をご参照ください。

sweep IDをコピーし、以下のコードスニペットにある`sweepID`を置き換えて、[`wandb agent`](../../ref/cli/wandb-agent.md)コマンドでsweepジョブを開始します：

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

sweepジョブの開始方法については、[Start sweep jobs](./start-sweep-agents.md)をご参照ください。
  </TabItem>
</Tabs>

## メトリクスをログする際の注意点

sweep設定で指定したメトリクスを明示的にW&Bへログするようにしてください。サブディレクトリ内でsweepのメトリクスをログしないでください。

例えば、以下の擬似コードを考えてみてください。ユーザーは検証損失 (`"val_loss": loss`)をログしたいとします。まず、辞書に値を渡します（16行目）。しかし、`wandb.log`に渡す辞書は、辞書内のキーと値のペアを明示的にアクセスしていません：

```python title="train.py" showLineNumbers
# Import the W&B Python Library and log into W&B
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
    # highlight-next-line
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

代わりに、Python辞書内のキーと値のペアを明示的にアクセスします。例えば、以下のコードでは、辞書を作成した後、`wandb.log`メソッドに辞書を渡す際にキーと値のペアを明示的に指定します：

```python title="train.py" showLineNumbers
# Import the W&B Python Library and log into W&B
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
    # highlight-next-line
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

