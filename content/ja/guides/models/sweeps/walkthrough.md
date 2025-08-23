---
title: 'チュートリアル: Sweep を定義、初期化、実行する'
description: スイープ クイックスタートでは、スイープを定義、初期化、実行する方法を示します。主な手順は4つあります。
menu:
  default:
    identifier: ja-guides-models-sweeps-walkthrough
    parent: sweeps
weight: 1
---

このページでは、スイープを定義、初期化、および実行する方法を示します。主に4つのステップがあります。

1. [トレーニングコードをセットアップする]({{< relref path="#set-up-your-training-code" lang="ja" >}})
2. [スイープ設定で探索空間を定義する]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ja" >}})
3. [スイープを初期化する]({{< relref path="#initialize-the-sweep" lang="ja" >}})
4. [スイープエージェントを開始する]({{< relref path="#start-the-sweep" lang="ja" >}})

以下のコードを Jupyter ノートブックまたは Python スクリプトにコピーして貼り付けてください。

```python
# W&B Python ライブラリをインポートして W&B にログインする
import wandb

wandb.login()

# 1: 目的/トレーニング関数を定義する
def objective(config):
    score = config.x**3 + config.y
    return score

def train():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

if __name__ == '__main__':
    # 2: 探索空間を定義する
    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "minimize", "name": "score"},
        "parameters": {
            "x": {"max": 0.1, "min": 0.01},
            "y": {"values": [1, 3, 7]},
        },
    }

    # 3: スイープを開始する
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

    wandb.agent(sweep_id, function=train, count=10)
```

以下のセクションでは、そのコードサンプルの各ステップを分解し、説明します。

## トレーニングコードをセットアップする

`wandb.config` からハイパーパラメーターの値を取り込み、それを使用してモデルをトレーニングし、メトリクスを返すトレーニング関数を定義します。

オプションとして、W&B Run の出力を保存したいプロジェクトの名前（[`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}})内のproject パラメータ）を指定します。プロジェクトが指定されていない場合、Run は「Uncategorized」プロジェクトに入ります。

{{% alert %}}
スイープとrunは同じプロジェクト内にある必要があります。したがって、W&Bを初期化するときに指定する名前は、スイープを初期化するときに指定するプロジェクトの名前と一致する必要があります。
{{% /alert %}}

```python
# 1: 目的/トレーニング関数を定義する
def objective(config):
    score = config.x**3 + config.y
    return score


def train():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
```

## スイープ設定で探索空間を定義する

探索するハイパーパラメーターを辞書で指定します。設定オプションについては、[スイープ設定を定義する]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}})を参照してください。

次の例では、ランダム検索（`'method':'random'`）を使用するスイープ設定を示しています。スイープは、バッチサイズ、エポック、および学習率の設定にリストされているランダムな値を無作為に選択します。

W&Bは、`"goal": "minimize"`が関連付けられているときに `metric` キーで指定されたメトリクスを最小化します。この場合、W&Bはメトリクス `score`（`"name": "score"`）を最小化するように最適化します。

```python
if __name__ == '__main__':
    # 2: 探索空間を定義する
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

W&Bは、クラウド（標準）またはローカル（ローカル）で複数のマシンを横断してスイープを管理するために、_Sweep Controller_ を使用します。Sweep Controller についての詳細は、[ローカルで探索と停止のアルゴリズムを確認する]({{< relref path="./local-controller.md" lang="ja" >}})を参照してください。

スイープを初期化すると、スイープ識別番号が返されます。

```python
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

スイープの初期化に関する詳細は、[スイープを初期化する]({{< relref path="./initialize-sweeps.md" lang="ja" >}})を参照してください。

## スイープを開始する

スイープを開始するには、[`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}}) APIコールを使用します。

```python
    wandb.agent(sweep_id, function=train, count=10)
```

## 結果を視覚化する（オプション）

プロジェクトを開くと、W&Bアプリダッシュボードでライブ結果を確認できます。数回のクリックで豊富なインタラクティブグラフを構築します。例えば、[並列座標プロット]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}})、[パラメータの重要度解析]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}})、および[その他]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}})です。

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps ダッシュボード例" >}}

結果の視覚化方法に関する詳細は、[スイープ結果を視覚化する]({{< relref path="./visualize-sweep-results.md" lang="ja" >}})を参照してください。サンプルのダッシュボードについては、この[スイーププロジェクト](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)を参照してください。

## エージェントを停止する（オプション）

ターミナルで `Ctrl+C` を押して、現在のランを停止します。もう一度押すと、エージェントが終了します。
