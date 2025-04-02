---
title: 'Tutorial: Define, initialize, and run a sweep'
description: Sweeps クイックスタート では、sweep の定義、初期化、および実行方法について説明します。主な手順は4つあります。
menu:
  default:
    identifier: ja-guides-models-sweeps-walkthrough
    parent: sweeps
weight: 1
---

このページでは、sweep の定義、初期化、および実行方法について説明します。主な手順は4つあります。

1. [トレーニング コードのセットアップ]({{< relref path="#set-up-your-training-code" lang="ja" >}})
2. [sweep configuration での探索空間の定義]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ja" >}})
3. [sweep の初期化]({{< relref path="#initialize-the-sweep" lang="ja" >}})
4. [sweep agent の起動]({{< relref path="#start-the-sweep" lang="ja" >}})

次のコードをコピーして Jupyter Notebook または Python スクリプトに貼り付けます。

```python
# W&B Python ライブラリをインポートして W&B にログインします
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

# 3: sweep を開始します
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

以下のセクションでは、コード サンプルの各ステップを分解して説明します。

## トレーニング コードのセットアップ
`wandb.config` から ハイパーパラメーター の値を受け取り、それらを使用して model をトレーニングし、メトリクスを返すトレーニング関数を定義します。

必要に応じて、W&B Run の出力を保存する project の名前を指定します ([`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}}) の project パラメータ)。project が指定されていない場合、run は「未分類」の project に配置されます。

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

## sweep configuration での探索空間の定義

辞書で sweep する ハイパーパラメーター を指定します。configuration オプションについては、[sweep configuration の定義]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。

上記の例は、ランダム検索 (`'method':'random'`) を使用する sweep configuration を示しています。sweep は、バッチサイズ、エポック、および学習率について、configuration にリストされている値のランダムなセットをランダムに選択します。

W&B は、`"goal": "minimize"` が関連付けられている場合、`metric` キーで指定された メトリクス を最小化します。この場合、W&B は メトリクス `score` (`"name": "score"`) の最小化のために最適化します。

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

## Sweep の初期化

W&B は _Sweep Controller_ を使用して、クラウド (標準)、ローカル (ローカル) で1つまたは複数のマシンにわたる Sweeps を管理します。Sweep Controller の詳細については、[ローカルでの検索と停止アルゴリズム]({{< relref path="./local-controller.md" lang="ja" >}})を参照してください。

sweep 識別番号は、sweep を初期化するときに返されます。

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

sweep の初期化の詳細については、[sweep の初期化]({{< relref path="./initialize-sweeps.md" lang="ja" >}})を参照してください。

## Sweep の開始

[`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}}) API 呼び出しを使用して sweep を開始します。

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 結果の可視化 (オプション)

project を開いて、W&B App ダッシュボードでライブ結果を確認します。数回クリックするだけで、[パラレル座標図]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[パラメーター の重要性分析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}) などの豊富なインタラクティブなグラフを構築できます。[詳細]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard example" >}}

結果の可視化方法の詳細については、[sweep 結果の可視化]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})を参照してください。ダッシュボードの例については、このサンプル[Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)を参照してください。

## エージェント の停止 (オプション)

ターミナル で、`Ctrl+C` を押して現在の run を停止します。もう一度押すと、agent が終了します。
