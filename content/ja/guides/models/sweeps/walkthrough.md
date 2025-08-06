---
title: チュートリアル：sweep の定義、初期化、および実行
description: スイープクイックスタートでは、スイープの定義、初期化、実行方法を紹介します。主なステップは4つあります。
menu:
  default:
    identifier: ja-guides-models-sweeps-walkthrough
    parent: sweeps
weight: 1
---

このページでは、スイープの定義、初期化、実行方法を説明します。主なステップは4つです。

1. [トレーニングコードのセットアップ]({{< relref path="#set-up-your-training-code" lang="ja" >}})
2. [スイープ設定で探索空間を定義]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ja" >}})
3. [スイープを初期化]({{< relref path="#initialize-the-sweep" lang="ja" >}})
4. [スイープエージェントを開始]({{< relref path="#start-the-sweep" lang="ja" >}})


以下のコードを Jupyter Notebook または Python スクリプトにコピー＆ペーストしてください。

```python 
# W&B Python ライブラリをインポート & W&B にログイン
import wandb

# 1: 目的関数 / トレーニング関数の定義
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})

# 2: 探索空間の定義
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: スイープの開始
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

続くセクションでは、このコードサンプルの各ステップについて詳しく解説します。


## トレーニングコードのセットアップ
`wandb.Run.config` からハイパーパラメーターの値を受け取り、それを使ってモデルのトレーニングやメトリクスの返却を行うトレーニング関数を定義します。

W&B Run の出力先となるプロジェクト名（[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}) の project パラメータ）をオプションで指定できます。プロジェクトを指定しなかった場合、その run は「Uncategorized」プロジェクトに格納されます。

{{% alert %}}
スイープと run は同じプロジェクト内にある必要があります。したがって、W&B を初期化する際に指定するプロジェクト名は、スイープを初期化する際のプロジェクト名と一致させてください。
{{% /alert %}}

```python
# 1: 目的関数 / トレーニング関数の定義
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})
```

## スイープ設定で探索空間を定義

探索するハイパーパラメーターを辞書形式で指定します。設定オプションの詳細は[スイープ設定の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。

下記の例はランダム検索 (`'method':'random'`) を用いたスイープ設定です。ここでは、バッチサイズ、エポック数、学習率について設定で指定した値からランダムに選びます。

`metric` キーに `"goal": "minimize"` を設定した場合、W&B は指定したメトリクス（この例では `score`）の最小化を目指して最適化を行います（`"name": "score"`）。


```python
# 2: 探索空間の定義
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
```

## スイープを初期化

W&B は _Sweep Controller_ を使い、クラウド（standard）、ローカル（local）、複数マシンを跨いで Sweeps を管理します。Sweep Controller の詳細は[ローカルでの探索や停止アルゴリズムについて]({{< relref path="./local-controller.md" lang="ja" >}})をご確認ください。

スイープを初期化すると、スイープ ID（識別番号）が返却されます：

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

スイープの初期化について詳しくは[スイープの初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})を参照してください。

## スイープエージェントを開始

スイープを実行するには、[`wandb.agent`]({{< relref path="/ref/python/sdk/functions/agent.md" lang="ja" >}}) API を使います。

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 結果の可視化（オプション）

プロジェクトを開くと、W&B アプリのダッシュボードでリアルタイムの結果を確認できます。クリック操作だけで、[パラレル座標プロット]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[パラメータの重要度分析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}})、[その他のチャート各種]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})など、豊富でインタラクティブなグラフが作成できます。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard example" >}}

結果の可視化方法の詳細は[スイープ結果の可視化]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})をご覧ください。ダッシュボードのサンプルはこの [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3) から確認できます。

## エージェントの停止（オプション）

ターミナルで `Ctrl+C` を押すと現在の run を停止できます。もう一度押すとエージェントが終了します。

```
```