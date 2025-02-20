---
title: 'Tutorial: Define, initialize, and run a sweep'
description: スイープ クイックスタートでは、スイープを定義、初期化、および run する方法を示します。主な手順は4つあります。
menu:
  default:
    identifier: ja-guides-models-sweeps-walkthrough
    parent: sweeps
weight: 1
---

このページでは、sweep を定義、初期化、実行する方法を示しています。主なステップは4つあります。

1. [トレーニングコードを設定する]({{< relref path="#set-up-your-training-code" lang="ja" >}})
2. [スイープ設定で検索空間を定義する]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ja" >}})
3. [スイープを初期化する]({{< relref path="#initialize-the-sweep" lang="ja" >}})
4. [スイープエージェントを開始する]({{< relref path="#start-the-sweep" lang="ja" >}})

以下のコードを Jupyter ノートブックまたは Python スクリプトにコピーして貼り付けてください:

```python
# W&B Python ライブラリをインポートし、W&B にログインします
import wandb

wandb.login()

# 1: 目的/トレーニング関数を定義します
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: 検索空間を定義します
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: スイープを開始します
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

以下のセクションでは、コードサンプルの各ステップを分解して説明します。

## トレーニングコードを設定する

`wandb.config` からハイパーパラメーターの 値を受け取り、それを使用してモデルをトレーニングし、メトリクスを返すトレーニング関数を定義します。

オプションとして、W&B Run の出力を保存したいプロジェクトの名前を指定することができます（[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) のプロジェクトパラメータ）。プロジェクトが指定されていない場合、run は「Uncategorized」プロジェクトに入れられます。

{{% alert %}}
sweep と run は同じプロジェクト内にある必要があります。そのため、W&B を初期化するときに提供する名前は、sweep を初期化するときに提供するプロジェクトの名前と一致する必要があります。
{{% /alert %}}

```python
# 1: 目的/トレーニング関数を定義します
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
```

## スイープ設定で検索空間を定義する

辞書内で、スイープしたいハイパーパラメーターを指定します。設定オプションの詳細については、[スイープ設定を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}) を参照してください。

以下の例では、ランダム検索（'method':'random'）を使用するスイープ設定を示しています。スイープは設定で指定されたバッチサイズ、エポック、学習率の値をランダムに選択します。

スイープ全体を通して、W&B はメトリクス キー（`metric`）で指定されたメトリクスを最大化します。以下の例では、W&B は検証精度（`'val_acc'`）を最大化します（`'goal':'maximize'`）。

```python
# 2: 検索空間を定義します
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
```

## スイープを初期化する

W&B は _Sweep Controller_ を使用して、クラウド（標準）、ローカル（local）で、または複数のマシンにわたってスイープを管理します。Sweep Controllers の詳細については、[Search and stop algorithms locally]({{< relref path="./local-controller.md" lang="ja" >}}) を参照してください。

スイープを初期化すると、スイープ識別番号が返されます。

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

スイープの初期化に関する詳細については、[スイープの初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}}) を参照してください。

## スイープを開始する

【`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}}) API を呼び出してスイープを開始します。

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 結果を可視化する（オプション）

プロジェクトを開いて、W&B アプリのダッシュボードでライブ結果を確認できます。わずか数回のクリックで、[パラレル座標図]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[パラメータの重要性]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) などの、豊かでインタラクティブなグラフを構築します。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps ダッシュボードの例" >}}

結果の可視化方法についての詳細は、[スイープ結果の可視化]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}) を参照してください。ダッシュボードの例については、このサンプル [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3) を参照してください。

## エージェントを停止する（オプション）

ターミナルから `Ctrl+c` を押して、Sweep エージェントが現在実行している run を停止します。エージェントを終了するには、run が停止した後に再度 `Ctrl+c` を押します。