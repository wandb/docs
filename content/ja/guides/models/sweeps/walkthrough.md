---
title: 'Tutorial: Define, initialize, and run a sweep'
description: Sweeps クイックスタート では、 sweep を定義、初期化、および実行する方法を紹介します。主な手順は 4 つあります。
menu:
  default:
    identifier: ja-guides-models-sweeps-walkthrough
    parent: sweeps
weight: 1
---

このページでは、sweepを定義、初期化、および実行する方法について説明します。主な手順は4つあります。

1. [トレーニング コードをセットアップする]({{< relref path="#set-up-your-training-code" lang="ja" >}})
2. [sweep configuration で探索空間を定義する]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ja" >}})
3. [sweepを初期化する]({{< relref path="#initialize-the-sweep" lang="ja" >}})
4. [sweep エージェントを開始する]({{< relref path="#start-the-sweep" lang="ja" >}})

次のコードをコピーして、Jupyter Notebook または Python スクリプトに貼り付けます。

```python
# W&B Python Library をインポートして W&B にログインします
import wandb

wandb.login()

# 1: 目的関数/トレーニング関数を定義します
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

# 2: 探索空間を定義します
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: sweepを開始します
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

以下のセクションでは、コード サンプルの各ステップを分解して説明します。

## トレーニング コードをセットアップする
`wandb.config` からハイパー パラメーター値を入力し、それらを使用してモデルをトレーニングし、メトリクスを返すトレーニング関数を定義します。

必要に応じて、W&B Run の出力を保存する project の名前を指定します (project パラメータは [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) にあります)。project が指定されていない場合、run は「未分類」project に配置されます。

{{% alert %}}
sweep と run は両方とも同じ project に存在する必要があります。したがって、W&B を初期化するときに指定する名前は、sweep を初期化するときに指定する project の名前と一致する必要があります。
{{% /alert %}}

```python
# 1: 目的関数/トレーニング関数を定義します
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
```

## sweep configuration で探索空間を定義する
辞書内で、sweep するハイパー パラメーターを指定します。configuration オプションの詳細については、[sweep configuration の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。

次の例は、ランダム検索 (`'method':'random'`) を使用する sweep configuration を示しています。sweep は、バッチサイズ、エポック、および学習率の configuration にリストされている値のランダムなセットをランダムに選択します。

sweep 全体を通して、W&B は metric キー (`metric`) で指定されたメトリクスを最大化します。次の例では、W&B は検証精度 (`'val_acc'`) を最大化します (`'goal':'maximize'`)。

```python
# 2: 探索空間を定義します
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
```

## Sweep を初期化する

W&B は _Sweep Controller_ を使用して、クラウド (標準)、ローカル (ローカル) で 1 台以上のマシンにまたがる sweeps を管理します。Sweep Controller の詳細については、[ローカルで検索および停止アルゴリズムを実行する]({{< relref path="./local-controller.md" lang="ja" >}})を参照してください。

sweep 識別番号は、sweep を初期化すると返されます。

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

sweep の初期化の詳細については、[sweeps の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})を参照してください。

## Sweep を開始する

[`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}}) API 呼び出しを使用して、sweep を開始します。

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 結果を視覚化する (オプション)

project を開いて、W&B App ダッシュボードでライブ結果を確認します。数回クリックするだけで、[パラレル座標図]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[パラメーターの重要性分析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}})、[その他]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}) などの豊富なインタラクティブなグラフを作成できます。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard example" >}}

結果の視覚化方法の詳細については、[sweep 結果の視覚化]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})を参照してください。ダッシュボードの例については、このサンプルの [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)を参照してください。

## エージェントを停止する (オプション)

ターミナルから、`Ctrl+c` を押して、Sweep agent が現在実行している run を停止します。エージェントを強制終了するには、run が停止した後にもう一度 `Ctrl+c` を押します。
