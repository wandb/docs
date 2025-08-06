---
title: チュートリアル：Sweep の定義、初期化、実行
description: Sweeps クイックスタートでは、スイープを定義し、初期化し、実行する方法を紹介します。主に 4 つのステップがあります。
menu:
  default:
    identifier: walkthrough_sweeps
    parent: sweeps
weight: 1
---

このページでは、スイープを定義し、初期化し、実行する方法を説明します。主なステップは4つあります。

1. [トレーニングコードをセットアップする]({{< relref "#set-up-your-training-code" >}})
2. [スイープ設定で探索空間を定義する]({{< relref "#define-the-search-space-with-a-sweep-configuration" >}})
3. [スイープを初期化する]({{< relref "#initialize-the-sweep" >}})
4. [スイープエージェントを開始する]({{< relref "#start-the-sweep" >}})

以下のコードを Jupyter Notebook や Python スクリプトにコピー＆ペーストしてください。

```python 
# W&B Python ライブラリをインポートし、W&B にログイン
import wandb

# 1: 目的関数／トレーニング関数を定義
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})

# 2: 探索空間を定義
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: スイープを開始
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

以下のセクションでは、上記コードの各ステップについて詳しく説明します。

## トレーニングコードをセットアップする

`wandb.Run.config` からハイパーパラメーターの値を受け取り、それらを使ってモデルをトレーニングし、メトリクスを返すトレーニング関数を定義します。

オプションで、W&B Run の出力を保存したい Project 名（[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) の `project` パラメータ）を指定できます。Project を指定しない場合、run は "Uncategorized" Project に配置されます。

{{% alert %}}
スイープと run の両方が同じ Project 内にある必要があります。そのため、W&B を初期化する際に指定する Project 名と、スイープを初期化する際に指定する Project 名を一致させてください。
{{% /alert %}}

```python
# 1: 目的関数／トレーニング関数を定義
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})
```

## スイープ設定で探索空間を定義する

スイープ対象のハイパーパラメーターを辞書（dictionary）形式で指定します。設定オプションの詳細は[スイープ設定の定義]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}})をご覧ください。

次の例は、ランダムサーチ（`'method':'random'`）を使ったスイープ設定例です。このスイープでは、設定内で指定したバッチサイズ、エポック、学習率の値からランダムに組み合わせを選択します。

`metric` キーで `"goal": "minimize"` を指定すると、W&B はそのメトリクスを最小化するように最適化します。この例では、`"name": "score"` なので、`score` を最小化することを目指します。

```python
# 2: 探索空間を定義
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

W&B は _Sweep Controller_ を使って、クラウド（standard）やローカル（local）、1台または複数マシンでスイープを管理します。Sweep Controller の詳細は[ローカルでの探索・停止アルゴリズム]({{< relref "./local-controller.md" >}})を参照してください。

スイープを初期化すると、スイープID（識別番号）が返されます。

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

スイープの初期化の詳細は[スイープの初期化]({{< relref "./initialize-sweeps.md" >}})をご覧ください。

## スイープエージェントを開始する

スイープを開始するには、[`wandb.agent`]({{< relref "/ref/python/sdk/functions/agent.md" >}}) API を呼び出します。

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 結果を可視化する（任意）

Project を開いて、W&B App ダッシュボード上でリアルタイムの結果を確認できます。数クリックで、[パラレル座標プロット]({{< relref "/guides/models/app/features/panels/parallel-coordinates.md" >}})、[パラメータの重要度分析]({{< relref "/guides/models/app/features/panels/parameter-importance.md" >}})、[その他のグラフ]({{< relref "/guides/models/app/features/panels/" >}})など、インタラクティブなグラフを作成できます。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard example" >}}

結果の可視化方法の詳細は[スイープ結果の可視化]({{< relref "./visualize-sweep-results.md" >}})を、ダッシュボードのサンプルは[こちらの Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3) をご覧ください。

## エージェントを停止する（任意）

ターミナルで `Ctrl+C` を押すと、現在の run を停止できます。もう一度押すとエージェント自体が終了します。

```