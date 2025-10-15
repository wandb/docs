---
title: コードに W&B (wandb) を追加する
description: Python コード スクリプトまたは Jupyter Notebook に W&B を追加します。
menu:
  default:
    identifier: ja-guides-models-sweeps-add-w-and-b-to-your-code
    parent: sweeps
weight: 2
---

W&B Python SDKをスクリプトやJupyterノートブックに追加する方法は数多くあります。以下は、W&B Python SDKを独自のコードに統合するための「ベストプラクティス」の例です。

### オリジナルトレーニングスクリプト

次のPythonスクリプトのコードを持っているとしましょう。`main` という関数を定義し、典型的なトレーニングループを模倣します。各エポックごとに、トレーニングおよび検証データセットに対して精度と損失が計算されます。この例の目的のために値はランダムに生成されます。

ハイパーパラメーター値を格納するための辞書 `config` を定義しました。セルの最後に、モックトレーニングコードを実行するために `main` 関数を呼び出します。

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

# ハイパーパラメーター値を含む設定変数
config = {"lr": 0.0001, "bs": 16, "epochs": 5}

def main():
    # 固定値を定義する代わりに `wandb.config` から値を定義していることに注意
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

### W&B Python SDKを用いたトレーニングスクリプト

以下のコード例は、W&B Python SDKをコードに追加する方法を示しています。CLIでW&B Sweepジョブを開始する場合、CLIタブを探索したいでしょう。JupyterノートブックやPythonスクリプト内でW&B Sweepジョブを開始する場合、Python SDKタブを探索してください。

{{< tabpane text=true >}}
    {{% tab header="Python スクリプトまたはノートブック" %}}
 W&B Sweepを作成するために、コード例に以下を追加しました:

1. Weights & Biases Python SDKをインポートします。
2. キーと値のペアがスイープ設定を定義する辞書オブジェクトを作成します。次の例では、バッチサイズ (`batch_size`), エポック (`epochs`), および学習率 (`lr`) のハイパーパラメーターが各スイープで変化します。スイープ設定の作成方法についての詳細は、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。
3. スイープ設定辞書を[`wandb.sweep`]({{< relref path="/ref/python/sweep.md" lang="ja" >}})に渡します。これによりスイープが初期化され、スイープID (`sweep_id`) が返されます。スイープの初期化方法についての詳細は、[Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}})を参照してください。
4. [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) APIを使用して、データを同期およびログ化しながら、バックグラウンドプロセスを生成して [W&B Run]({{< relref path="/ref/python/run.md" lang="ja" >}}) として実行します。
5. (オプション) 固定値を定義する代わりに `wandb.config` から値を定義します。
6. `wandb.log` を使用して最適化したいメトリクスをログします。設定で定義されたメトリクスを必ずログしてください。この例では、設定辞書 (`sweep_configuration`) で `val_acc` を最大化するスイープを定義しました。
7. [`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}}) API呼び出しを使用してスイープを開始します。スイープID、スイープが実行する関数の名前 (`function=main`)、および試行する最大run数を4に設定します (`count=4`)。W&B Sweepの開始方法についての詳細は、[Start sweep agents]({{< relref path="./start-sweep-agents.md" lang="ja" >}})を参照してください。

```python
import wandb
import numpy as np
import random

# スイープ設定を定義
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

# 設定を渡してスイープを初期化します。
# (オプション) プロジェクト名を指定
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")


# `wandb.config` からハイパーパラメーターを受け取り、
# モデルをトレーニングしてメトリクスを返すトレーニング関数を定義
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

    # 固定値を定義する代わりに `wandb.config`
    # から値を定義していることに注意
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


# スイープジョブを開始
wandb.agent(sweep_id, function=main, count=4)
```

{{% /tab %}}
{{% tab header="CLI" %}}

W&B Sweepを作成するために、最初にYAML設定ファイルを作成します。設定ファイルにはスイープが探索するハイパーパラメーターを含んでいます。次の例では、バッチサイズ (`batch_size`), エポック (`epochs`), および学習率 (`lr`) のハイパーパラメーターが各スイープで変化します。

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

W&B Sweep設定の作成方法についての詳細は、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。

YAMLファイルで `program` のキーにPythonスクリプトの名前を必ず指定してください。

次に、コード例に以下を追加します:

1. Weights & Biases Python SDK (`wandb`) と PyYAML (`yaml`) をインポートします。PyYAMLはYAML設定ファイルを読み込むために使用します。
2. 設定ファイルを読み込みます。
3. [`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) APIを使用して、データを同期およびログ化しながら、バックグラウンドプロセスを生成して [W&B Run]({{< relref path="/ref/python/run.md" lang="ja" >}}) として実行します。configパラメーターに設定オブジェクトを渡します。
4. 固定値を使用する代わりに `wandb.config` からハイパーパラメーター値を定義します。
5. `wandb.log` を使用して最適化したいメトリクスをログします。設定で定義されたメトリクスを必ずログしてください。この例では、設定辞書 (`sweep_configuration`) で `val_acc` を最大化するスイープを定義しました。

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
    # 標準のハイパーパラメーターをセットアップします
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    run = wandb.init(config=config)

    # 固定値を定義する代わりに `wandb.config`
    # から値を定義していることに注意
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


# メイン関数を呼び出します。
main()
```

CLIに移動します。CLI内で、スイープエージェントが試行する最大run数を設定します。これは任意のステップです。次の例では最大回数を5に設定しています。

```bash
NUM=5
```

次に、[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドを使用してスイープを初期化します。YAMLファイルの名前を指定します。プロジェクトフラグ（`--project`）のためにプロジェクト名を指定することもできます:

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

これによりスイープIDが返されます。スイープの初期化方法についての詳細は、[Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}})を参照してください。

スイープIDをコピーし、次のコードスニペットの `sweepID` を置き換えて、[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドでスイープジョブを開始します:

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

スイープジョブの開始方法についての詳細は、[Start sweep jobs]({{< relref path="./start-sweep-agents.md" lang="ja" >}})を参照してください。

{{% /tab %}}
{{< /tabpane >}}

## メトリクスをログする際の考慮事項

スイープ設定で指定したメトリクスを明示的にW&Bにログすることを確認してください。スイープのメトリクスをサブディレクトリ内でログしないでください。

例えば、以下の擬似コードを考えてみてください。ユーザーが検証損失 (`"val_loss": loss`) をログしたいとします。まず、ユーザーは辞書に値を渡しますが、`wandb.log` に渡される辞書は辞書内のキーと値のペアに明示的にアクセスしていません:

```python
# W&B Pythonライブラリをインポートし、W&Bにログイン
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
    # 不正確。辞書内のキーと値のペアに明示的にアクセスする必要があります。
    # 次のコードブロックでメトリクスを正しくログする方法を参照してください。
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

代わりに、Python辞書内でキーと値のペアに明示的にアクセスしてください。例えば、次のコードは `wandb.log` メソッドに辞書を渡す際にキーと値のペアを指定しています:

```python title="train.py"
# W&B Pythonライブラリをインポートし、W&Bにログイン
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