---
title: コードに W&B （wandb）を追加する
description: W&B を Python のコード（スクリプト）または Jupyter ノートブックに追加します。
menu:
  default:
    identifier: ja-guides-models-sweeps-add-w-and-b-to-your-code
    parent: sweeps
weight: 2
---

このガイドでは、Python の トレーニングスクリプト または ノートブック に W&B を統合する方法の推奨事項を紹介します。

## 元の トレーニングスクリプト

以下のように モデル を学習する Python スクリプトがあるとします。目標は、検証精度（`val_acc`）を最大化する ハイパーパラメーター を見つけることです。

Python スクリプトでは、`train_one_epoch` と `evaluate_one_epoch` の 2 つの関数を定義します。`train_one_epoch` は 1 エポック分の トレーニング をシミュレートし、学習時の精度と損失を返します。`evaluate_one_epoch` は 検証 データセット 上で モデル を評価することをシミュレートし、検証時の精度と損失を返します。

学習率（`lr`）、バッチサイズ（`batch_size`）、エポック数（`epochs`）といった ハイパーパラメーター の 値 を含む 設定 用の 辞書（`config`）を定義します。設定 辞書の 値 が トレーニング プロセス を制御します。

続いて、典型的な学習ループを模した `main` 関数を定義します。各 エポック ごとに、学習および検証 データセット で精度と損失を計算します。

{{< alert >}}
この コード はモックの トレーニングスクリプト です。実際に モデル を学習するのではなく、精度と損失の ランダム な 値 を生成して学習プロセスをシミュレートします。この コード の目的は、W&B を トレーニングスクリプト に統合する方法を示すことです。
{{< /alert >}}

```python
import random
import numpy as np

def train_one_epoch(epoch, lr, batch_size):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss

def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss

# config variable with hyperparameter values
config = {"lr": 0.0001, "batch_size": 16, "epochs": 5}

def main():
    lr = config["lr"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, batch_size)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "validation loss:", val_loss)

if __name__ == "__main__":
    main()
```

次のセクションでは、トレーニング 中に ハイパーパラメーター と メトリクス を追跡できるよう、Python スクリプト に W&B を追加します。W&B を使って、検証精度（`val_acc`）を最大化する最適な ハイパーパラメーター を見つけます。

## W&B Python SDK を用いた トレーニングスクリプト

Python スクリプト や ノートブック への W&B の統合方法は、Sweeps の管理方法によって異なります。Python の ノートブック や スクリプト から、あるいは コマンドライン から sweep ジョブを開始できます。

{{< tabpane text=true >}}
{{% tab header="Python スクリプト または ノートブック" %}} 

Python スクリプト に以下を追加します。

1. キーと 値 の組で [sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を定義する 辞書 オブジェクトを作成します。sweep configuration には、W&B に探索してほしい ハイパーパラメーター と、最適化したい メトリクス を指定します。前節の例を踏まえると、バッチサイズ（`batch_size`）、エポック（`epochs`）、学習率（`lr`）を各 sweep で変化させる ハイパーパラメーター とします。検証スコアの精度を最大化したいので、`"goal": "maximize"` を設定し、最適化対象の 変数 名（この例では `val_acc`、すなわち `"name": "val_acc"`）を指定します。
2. sweep configuration の 辞書 を [`wandb.sweep()`]({{< relref path="/ref/python/sdk/functions/sweep.md" lang="ja" >}}) に渡します。これにより sweep が初期化され、sweep ID（`sweep_id`）が返されます。詳細は [Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。
3. スクリプト の先頭で W&B Python SDK（`wandb`）をインポートします。
4. `main` 関数内で、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) API を使って、[W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) として データ を同期・ログするバックグラウンド プロセスを生成します。`wandb.init()` メソッドには パラメータ として Project 名を渡してください。Project 名を渡さない場合、W&B はデフォルトの Project 名を使用します。
5. `wandb.Run.config` オブジェクトから ハイパーパラメーター の 値 を取得します。これにより、ハードコードした 値 の代わりに sweep configuration の 辞書 で定義した 値 を使用できます。
6. 最適化対象の メトリクス を [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}) を使って W&B に ログ します。設定 で定義した メトリクス を必ず ログ してください。たとえば最適化する メトリクス を `val_acc` と定義した場合、`val_acc` を ログ する必要があります。メトリクス を ログ しないと、W&B は何を最適化すべきかを認識できません。設定 辞書（この例では `sweep_configuration`）内で、`val_acc` の 値 を最大化するように sweep を定義します。
7. [`wandb.agent()`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ja" >}}) で sweep を開始します。sweep ID、sweep が実行する関数名（`function=main`）、試行する run の最大数（`count=4`）を指定します。

これらをすべて組み合わせると、スクリプト は次のようになります。

```python
import wandb # W&B Python SDK をインポート
import numpy as np
import random
import argparse

def train_one_epoch(epoch, lr, batch_size):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss

def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss

def main(args=None):
    # sweep agent から呼ばれるときは args は None
    # そのため Project は sweep の config から取得する
    project = args.project if args else None
    
    with wandb.init(project=project) as run:
        # `wandb.Run.config` オブジェクトから ハイパーパラメーター を取得
        lr = run.config["lr"]
        batch_size = run.config["batch_size"]
        epochs = run.config["epochs"]

        # 学習ループを実行し、性能値を W&B に ログ する
        for epoch in np.arange(1, epochs):
            train_acc, train_loss = train_one_epoch(epoch, lr, batch_size)
            val_acc, val_loss = evaluate_one_epoch(epoch)
            run.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc, # 最適化するメトリクス
                    "val_loss": val_loss,
                }
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="sweep-example", help="W&B project name")
    args = parser.parse_args()

    # sweep config の 辞書 を定義
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        # 最適化したいメトリクス
        # 例えば、検証精度を最大化したいなら
        # "goal": "maximize" と、最適化対象の 変数 名（この場合は "val_acc"）を指定
        "metric": {
            "goal": "maximize",
            "name": "val_acc"
            },
        "parameters": {
            "batch_size": {"values": [16, 32, 64]},
            "epochs": {"values": [5, 10, 15]},
            "lr": {"max": 0.1, "min": 0.0001},
        },
    }

    # 設定 辞書 を渡して sweep を初期化
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)

    # sweep ジョブを開始
    wandb.agent(sweep_id, function=main, count=4)
```

{{% /tab %}} {{% tab header="CLI" %}}

sweep configuration を記述した YAML 設定ファイルを作成します。設定ファイルには、sweep が探索する ハイパーパラメーター を含めます。次の例では、各 sweep で バッチサイズ（`batch_size`）、エポック（`epochs`）、学習率（`lr`）の ハイパーパラメーター を変化させます。

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

W&B の Sweep 設定の作成方法については、[Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

YAML ファイルの `program` キーには、Python スクリプト の名前を指定する必要があります。

次に、以下を コード 例に追加します。

1. W&B Python SDK（`wandb`）と PyYAML（`yaml`）をインポートします。PyYAML は YAML 設定ファイルの読み込みに使用します。
2. 設定ファイルを読み込みます。
3. [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) API を使って、[W&B Run]({{< relref path="/ref/python/sdk/classes/run.md" lang="ja" >}}) として データ を同期・ログするバックグラウンド プロセスを生成します。config パラメータに config オブジェクトを渡します。
4. ハードコードした 値 の代わりに、`wandb.Run.config` から ハイパーパラメーター を取得して使用します。
5. [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}) で、最適化したい メトリクス を ログ します。設定 で定義した メトリクス は必ず ログ してください。設定 辞書（この例では `sweep_configuration`）では、`val_acc` の 値 を最大化するように sweep を定義しています。

```python
import wandb
import yaml
import random
import numpy as np


def train_one_epoch(epoch, lr, batch_size):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    # 既定の ハイパーパラメーター を設定
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

CLI で、sweep agent が試行する run の最大数を設定します。これは任意です。この例では最大数を 5 に設定します。

```bash
NUM=5
```

次に、[`wandb sweep`]({{< relref path="/ref/cli/wandb-sweep.md" lang="ja" >}}) コマンドで sweep を初期化します。YAML ファイル名を指定し、任意で Project 名を `--project` フラグで指定します。

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

これにより sweep ID が返されます。sweep の初期化方法については、[Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

返ってきた sweep ID をコピーし、次の コードスニペット の `sweepID` を置き換えて、[`wandb agent`]({{< relref path="/ref/cli/wandb-agent.md" lang="ja" >}}) コマンドで sweep ジョブを開始します。

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

詳細は [Start sweep jobs]({{< relref path="./start-sweep-agents.md" lang="ja" >}}) を参照してください。

{{% /tab %}} {{< /tabpane >}}


{{% alert title="Sweep で W&B にメトリクスを ログ する際の注意" %}}
sweep configuration と `wandb.Run.log()` の両方で、定義して最適化対象としている メトリクス を必ず ログ してください。例えば sweep configuration 内で最適化する メトリクス を `val_acc` と定義した場合、W&B にも `val_acc` を ログ する必要があります。メトリクス を ログ しないと、W&B は何を最適化すべきかを認識できません。

```python
with wandb.init() as run:
    val_loss, val_acc = train()
    run.log(
        {
            "val_loss": val_loss,
            "val_acc": val_acc
            }
        )
```

次は W&B へのメトリクスの ログ の誤った例です。sweep configuration で最適化対象は `val_acc` ですが、この コード は `val_acc` を `validation` という キー の下の入れ子の 辞書 に ログ しています。メトリクス は入れ子ではなく、直接 ログ する必要があります。

```python
with wandb.init() as run:
    val_loss, val_acc = train()
    run.log(
        {
            "validation": {
                "val_loss": val_loss, 
                "val_acc": val_acc
                }
            }
        )
```

{{% /alert %}}