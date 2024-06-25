---
description: PythonコードスクリプトやJupyter NotebookにW&Bを追加します。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


#  W&B をコードに追加する

<head>
  <title>W&B を Python コードに追加する</title>
</head>

W&B Python SDK をスクリプトや Jupyter Notebook に追加する方法は多数あります。以下に、W&B Python SDK を自身のコードに統合する「ベストプラクティス」の例を示します。

### 元のトレーニングスクリプト

次のコードが Jupyter Notebook のセルや Python スクリプトにあるとします。`main` という関数を定義し、これは一般的なトレーニングループを模倣しています。それぞれのエポックで、訓練および検証データセットに対する精度と損失が計算されます。この例のために、値はランダムに生成されています。

`config` という辞書を定義し、そこにハイパーパラメータの値を保存します（15行目）。セルの最後で、`main` 関数を呼び出してモックトレーニングコードを実行します。

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
    # 注: 固定値を直接定義する代わりに
    # `wandb.config` の値を使用しています
    lr = config["lr"]
    bs = config["bs"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "training loss:", val_loss)


# main 関数を呼び出します。
main()
```

### W&B Python SDK を使ったトレーニングスクリプト

以下のコード例は、W&B Python SDK をコードに追加する方法を示しています。CLIでW&B Sweepジョブを開始する場合はCLIタブを調べてください。Jupyter ノートブックや Python スクリプト内で W&B Sweep ジョブを開始する場合は、Python SDK タブを探索してください。

<Tabs
  defaultValue="script"
  values={[
    {label: 'Python script or Jupyter Notebook', value: 'script'},
    {label: 'CLI', value: 'cli'},
  ]}>
  <TabItem value="script">
  W&B Sweep を作成するために、以下のコードを追加しました:

1. 1行目: Weights & Biases Python SDK をインポート。
2. 6行目: キーと値のペアがsweep configurationを定義する辞書オブジェクトを作成。次の例では、`batch_size`、`epochs`、および `lr` ハイパーパラメーターが各sweep中に変更されます。sweep configurationの作成方法については、[Define sweep configuration](./define-sweep-configuration.md) を参照してください。
3. 19行目: sweep configuration 辞書を [`wandb.sweep`](../../ref/python/sweep) に渡します。これにより、sweepが初期化されます。これにより、sweep ID (`sweep_id`) が返されます。sweepの初期化方法について詳しくは、[Initialize sweeps](./initialize-sweeps.md) を参照してください。
4. 33行目:`wandb.init()` APIを使用して、バックグラウンドプロセスを生成し、[W&B Run](../../ref/python/run.md) としてデータを同期およびログする。
5. 37-39行目: （オプション）固定値を直接定義する代わりに、`wandb.config` の値を使用します。
6. 45行目: [`wandb.log`](../../ref/python/log.md) を使用して、最適化したいメトリクスをログに記録します。設定で定義したメトリクスをログに記録する必要があります。この例では、設定辞書 (`sweep_configuration`) 内で `val_acc` の値を最大化するように定義しています。
7. 54行目: [`wandb.agent`](../../ref/python/agent.md) API 呼び出しで sweep を開始します。sweep ID（19行目）、sweep が実行する関数の名前（`function=main`）、および最大試行数を4に設定（`count=4`）。W&B Sweepの開始方法について詳しくは、[Start sweep agents](./start-sweep-agents.md) を参照してください。

```python showLineNumbers
import wandb
import numpy as np
import random

# sweep 設定を定義
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

# 設定を渡してsweepを初期化。（オプション）プロジェクト名を指定します。
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")


# `wandb.config` からハイパーパラメータを取得し、モデルをトレーニングしてメトリクスを返す関数
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

    # 注: 固定値を直接定義する代わりに
    # `wandb.config` の値を使用しています
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


# sweepジョブを開始します。
wandb.agent(sweep_id, function=main, count=4)
```
  </TabItem>
  <TabItem value="cli">

  W&B Sweep を作成するには、最初に YAML 設定ファイルを作成します。設定ファイルには、sweepで探索したいハイパーパラメーターが含まれています。次の例では、`batch_size`、`epochs`、および `lr` ハイパーパラメータが各sweep中に変更されます。
  
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

W&B Sweep 設定の作成方法について詳しくは、[Define sweep configuration](./define-sweep-configuration.md)をご覧ください。

YAMLファイルの `program` キーには、Python スクリプトの名前を指定する必要があることに注意してください。

次に、別のコード例を追加します:

1. 1-2行目: Weights & Biases Python SDK (`wandb`) と PyYAML (`yaml`) をインポート。PyYAMLはYAML設定ファイルを読み込むために使用されます。
2. 18行目: 設定ファイルを読み込みます。
3. 21行目: [`wandb.init()`](../../ref/python/init.md) API を使用して、バックグラウンドプロセスを生成し、[W&B Run](../../ref/python/run.md) としてデータを同期およびログする。config オブジェクトを config パラメータに渡します。
4. 25-27行目: 固定値を直接使用する代わりに、`wandb.config` からハイパーパラメータ値を定義します。
5. 33-39行目: [`wandb.log`](../../ref/python/log.md) を使用して、最適化したいメトリクスをログに記録します。設定で定義したメトリクスをログに記録する必要があります。この例では、設定辞書 (`sweep_configuration`) 内で `val_acc` の値を最大化するように定義しています。

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
    # デフォルトのハイパーパラメータを設定
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # 注: 固定値を直接定義する代わりに
    # `wandb.config` の値を使用しています
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


# main 関数を呼び出します。
main()
```

CLI に移動します。CLI 内で、sweep agent が試行する最大 run 数を設定します。これは任意の手順です。以下の例では、最大数を5に設定します。

```bash
NUM=5
```

次に、[`wandb sweep`](../../ref/cli/wandb-sweep.md) コマンドを使用して sweep を初期化します。YAML ファイルの名前を指定します。オプションで、project フラグ (`--project`) にプロジェクト名を指定できます。

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

これにより sweep ID が返されます。sweepの初期化方法について詳しくは、[Initialize sweeps](./initialize-sweeps.md) を参照してください。

sweep ID をコピーし、次のコードスニペットの `sweepID` を置き換えて、[`wandb agent`](../../ref/cli/wandb-agent.md) コマンドを使用して sweep ジョブを開始します。

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

sweepジョブの開始方法について詳しくは、[Start sweep jobs](./start-sweep-agents.md) を参照してください。
  </TabItem>
</Tabs>

## メトリクスをログする際の考慮事項

確実に、設定した sweep configuration で指定したメトリクスを W&B に明示的にログしてください。サブディレクトリ内で sweep のメトリクスをログしないでください。

例えば、次の pseudocode を考えてみましょう。ユーザーは検証損失 (`"val_loss": loss`) をログしたいと考えています。最初に値を辞書に渡します（16行目）。しかし、`wandb.log` に渡される辞書は辞書内のキーと値のペアを明示的にアクセスしていません。

```python title="train.py" showLineNumbers
# W&B Python ライブラリをインポートして W&B にログイン
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

代わりに、Python 辞書内のキーと値のペアを明示的に参照してください。例えば、次のコード（辞書を作成した後の行）で、`wandb.log` メソッドに辞書を渡す際にキーと値のペアを指定します。

```python title="train.py" showLineNumbers
# W&B Python ライブラリをインポートして W&B にログイン
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

wandb.agent(sweep_id, function=main, count=10)
```