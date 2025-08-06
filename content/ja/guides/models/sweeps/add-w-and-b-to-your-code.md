---
title: あなたのコードに W&B (wandb) を追加する
description: Python のコードスクリプトや Jupyter Notebook に W&B を追加しましょう。
menu:
  default:
    identifier: add-w-and-b-to-your-code
    parent: sweeps
weight: 2
---

W&B Python SDK をスクリプトやノートブックに追加する方法はさまざまです。このセクションでは、W&B Python SDK を自身のコードに組み込む「おすすめのベストプラクティス」を紹介します。

### 元のトレーニングスクリプト

以下のような Python スクリプトがあるとします。ここでは、典型的なトレーニングループを模した `main` という関数を定義しています。各エポックで、トレーニングおよび検証のデータセットに対して精度と損失を計算します。この例では値はランダムに生成されています。

`config` という辞書を用意し、ハイパーパラメーターの値を保存しています。セルの最後で `main` 関数を呼び出し、モックのトレーニング処理を実行しています。

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

# ハイパーパラメータ値を持つ config 変数
config = {"lr": 0.0001, "bs": 16, "epochs": 5}

def main():
    # ハードコーディングせず、`wandb.Run.config` から値を取得
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

ここからは、W&B Python SDK をコードに追加する方法を紹介します。CLI から W&B Sweep ジョブを始める場合は CLI タブを、Jupyter ノートブックや Python スクリプト内で Sweep ジョブを始める場合は Python SDK タブを参照してください。

{{< tabpane text=true >}} {{% tab header="Python script or notebook" %}}
W&B Sweep を作成するため、コード例に以下を追加しました：

1. W&B Python SDK をインポートします。
2. 辞書オブジェクトを作成し、スイープ設定のキーと値を定義します。下記の例では、バッチサイズ（`batch_size`）、エポック数（`epochs`）、学習率（`lr`）のハイパーパラメータを各 Sweep で変更します。詳しくは[スイープ設定の定義方法]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}})をご覧ください。
3. スイープ設定の辞書を[`wandb.sweep()`]({{< relref "/ref/python/sdk/functions/sweep.md" >}})に渡します。これでスイープが初期化され、スイープID（`sweep_id`）が返されます。詳細は[スイープの初期化]({{< relref "./initialize-sweeps.md" >}})参照。
4. [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) API でバックグラウンドプロセスを生成し、[W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}})としてデータの同期・ログを実施。
5. （オプション）ハードコーディングせず、`wandb.config` から値を取得する形で値を定義。
6. 最適化したいメトリクスを[`wandb.Run.log()`]({{< relref "/ref/python/sdk/classes/run.md/#method-runlog" >}})でログします。設定で定義したメトリクスを必ずログしてください。本例の設定辞書（`sweep_configuration`）では `val_acc` の値を最大化するよう指定しています。
7. [`wandb.agent`]({{< relref "/ref/python/sdk/functions/agent.md" >}}) API で sweep を開始します。スイープID、スイープで実行する関数名（`function=main`）、試行する run の最大数（`count=4`）を指定します。[sweep agent の開始方法]({{< relref "./start-sweep-agents.md" >}})も参照ください。

```python
import wandb
import numpy as np
import random

# ハイパーパラメータ値を`wandb.Run.config`から受け取って
# モデルをトレーニングし、メトリクスを返すトレーニング関数
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss

def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss

# スイープ設定の辞書を定義
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

# （オプション）Project名を指定
project = "my-first-sweep"

def main():
    # `with` 文でコンテキストマネージャーを利用して run を自動終了
    # これは各 run の最後に `run.finish()` を使うのと同等です
    with wandb.init(project=project) as run:

        # ハイパーパラメータ値を明示的に設定せず、
        # `wandb.Run.config` から取得
        lr = run.config["lr"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # トレーニングループを実施し、パフォーマンスを W&B へログ
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
    # 辞書の設定を渡してスイープを初期化
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    # スイープジョブを開始
    wandb.agent(sweep_id, function=main, count=4)
```

{{% alert %}}
上記コードスニペットは、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) API を `with` 文のコンテキストマネージャーで呼び出し、[W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}})としてバックグラウンドで同期・ログ処理を実行する方法を示しています。これにより、ログした値をアップロードした後に run を適切に終了できます。もう一つの方法は、トレーニングスクリプトの最初と最後で `wandb.init()` および `wandb.Run.finish()` を個別に呼び出す方法です。
{{% /alert %}}

{{% /tab %}} {{% tab header="CLI" %}}

W&B Sweep を作成するには、まず YAML 設定ファイルを用意します。このファイルには sweep で探索したいハイパーパラメータを記述します。以下の例ではバッチサイズ（`batch_size`）、エポック数（`epochs`）、学習率（`lr`）を各 Sweep で変化させています。

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

W&B Sweep 設定ファイル作成の詳細は、[スイープ設定の定義方法]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}})もご参照ください。

YAML ファイルの `program` キーには、ご自身の Python スクリプト名を指定してください。

続いて、以下の内容をコード例に追加します：

1. W&B Python SDK (`wandb`) と PyYAML (`yaml`) をインポートします。PyYAML は YAML 設定ファイルを読み込むために使用します。
2. 設定ファイルを読み込みます。
3. [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) API でバックグラウンドプロセスを生成し、[W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}}) としてデータの同期・ログを実施します。ここでは config パラメータに config オブジェクトを渡します。
4. ハードコーディングせず、`wandb.Run.config` からハイパーパラメータ値を取得します。
5. 最適化したいメトリクスを[`wandb.Run.log()`]({{< relref "/ref/python/sdk/classes/run.md/#method-runlog" >}})でログします。設定で定義したメトリクス（本例では `val_acc`）を必ずログしてください。

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
    # デフォルトのハイパーパラメータをセットアップ
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

# main 関数を呼び出します
main()
```

CLI では、sweep agent が試行する最大 run 数を指定できます。これはオプションですが、本例では最大 5 回に設定しています。

```bash
NUM=5
```

次に、[`wandb sweep`]({{< relref "/ref/cli/wandb-sweep.md" >}}) コマンドでスイープを初期化します。YAML ファイル名を指定し、（オプションで）プロジェクト名も --project フラグで指定できます。

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

このコマンドで sweep ID が返されます。sweep の初期化方法の詳細は
[スイープの初期化]({{< relref "./initialize-sweeps.md" >}})もご参照ください。

sweep ID をコピーし、以下のスニペットで `sweepID` を置き換えて、[`wandb agent`]({{< relref "/ref/cli/wandb-agent.md" >}}) コマンドで sweep ジョブを開始します。

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

詳細は [sweep ジョブの開始方法]({{< relref "./start-sweep-agents.md" >}}) もご参照ください。

{{% /tab %}} {{< /tabpane >}}

## メトリクスをログする際の注意点

スイープメトリクスを W&B へ明示的にログすることを忘れないでください。サブディレクトリー内で sweep のメトリクスをログしないようにしましょう。

例えば、以下のような疑似コードを考えてみてください。ユーザーが検証損失（`"val_loss": loss`）をログしたい場合、まず辞書に値を格納します。しかし、辞書を `wandb.Run.log()` に渡す際、キーと値のペアを明示的に指定していません。

```python
# W&B Python ライブラリをインポートし、W&B にログイン
import wandb
import random

def train():
    # トレーニングと検証のメトリクスをシミュレーション
    offset = random.random() / 5
    epoch = 5  # エポック値をシミュレーション
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    return loss, acc

def main():
    with wandb.init(entity="<entity>", project="my-first-sweep") as run:
        val_loss, val_acc = train()
        # 正しくありません。辞書内で
        # キーバリューペアを明示的に参照する必要があります
        # 正しいやり方は次のコードブロック参照
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

# sweep ジョブを開始
wandb.agent(sweep_id, function=main, count=10)
```

代わりに、Python 辞書のキーバリューペアを明示的に参照してください。以下の例では、`wandb.Run.log()` メソッドに渡す際、キーバリューペアを正しく指定しています。

```python title="train.py"
# W&B Python ライブラリをインポートし、W&B にログイン
import wandb
import random

def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    return loss, acc

def main():
    with wandb.init(entity="<entity>", project="my-first-sweep") as run:
        # 正しい例。辞書内のキーバリューペアを
        # メトリクスをログする際に明示的に参照
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