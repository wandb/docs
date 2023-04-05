---
description: Sweeps quickstart shows how to define, initialize, and run a sweep. There are four main steps
---

# クイックスタート

<head>
  <title>Sweeps Quickstart</title>
</head>

前述のクイックスタートは、スウィープの定義、初期化、および実行方法を示しています。4つの主要ステップがあります：

1. [トレーニングコードのセットアップ](#set-up-your-training-code)
2. [スウィープ構成で検索空間を定義する](#define-the-search-space-with-a-sweep-configuration)
3. [スウィープを初期化する](#initialize-the-sweep)
4. [スウィープエージェントを開始する](#start-the-sweep)

以下のスウィープクイックスタートコードをコピーして、JupyterノートブックまたはPythonスクリプトに貼り付けます：

```python
# Import the W&B Python Library and log into W&B
import wandb
wandb.login()

# 1: Define objective/training function
def objective(config):
    score = config.x ** 3 + config.y
    return score

def main():
    wandb.init(project='my-first-sweep')
    score = objective(wandb.config)
    wandb.log({'score': score})

# 2: Define the search space
sweep_configuration = {
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'score'},
    'parameters': 
    {
        'x': {'max': 0.1, 'min': 0.01},
        'y': {'values': [1, 3, 7]},
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')
wandb.agent(sweep_id, function=main, count=10)
```

以下のセクションでは、クイックスタートコードサンプル内の各ステップを詳細に説明します


## トレーニングコードのセットアップ​
「wandb.config」からハイパーパラメーター値を取り込み、この値を使ってモデルをトレーニングし、メトリクスを返すトレーニング関数を定義します。

オプションで、W&B Runの出力を保存したいプロジェクトに名前を付けます（[`wandb.init`](../../ref/python/init.md))内のプロジェクトパラメーター）。プロジェクトを指定しない場合、runは「未分類」プロジェクトに保存されます。

:::caution
W&BスウィープとW&B Runはともに、同じプロジェクト内にあることが必須です。このため、W&Bを初期化する時に付けた名前は、W&Bスウィープを初期化した時に付けたプロジェクト名と一致している必要があります。
:::

```python
# 1: Define objective/training function
def objective(config):
    score = config.x ** 3 + config.y
    return score

def main():
    wandb.init(project='my-first-sweep')
    score = objective(wandb.config)
    wandb.log({'score': score})
```

## スウィープ構成で検索空間を定義する​
辞書内で、スウィープしたいハイパーパラメーターを指定します。設定オプションの詳細情報は、[スウィープ構成を定義する](./define-sweep-configuration.md)を参照してください。

前述の例は、ランダム検索を使用するスウィープ構成を示しています（'method':'random'）。スウィープは、設定にリスト表示された、バッチサイズ、エポックおよび学習速度の値セットをランダムに選択します。

スウィープを通して、W&Bは、メトリックキー（metric）で指定されたメトリックを最大化します。以下の例では、W&Bは検証精度（'val_acc'）を最大化（'goal':'maximize'）します。


```python
# 2: Define the search space
sweep_configuration = {
    'method': 'random',
    'metric': {'goal': 'minimize', 'name': 'score'},
    'parameters': 
    {
        'x': {'max': 0.1, 'min': 0.01},
        'y': {'values': [1, 3, 7]},
     }
}
```

## スウィープの初期化

W&Bはスウィープコントローラを使って、クラウド上で（標準）、ローカルで（ローカル）、1台または複数台のマシン上でスウィープを管理します。スウィープコントローラの詳細情報は、[アルゴリズムをローカルで検索および停止する](./local-controller.md)を参照してください。

スウィープを初期化すると、スウィープの識別番号が返されます：

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project='my-first-sweep')
```

スウィープの初期化に関する詳細情報は、[Initialize sweeps](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。

## スウィープを開始する​

wandbエージェント[`wandb.agent`](../../ref/python/agent.md) APコールを使ってW&Bスウィープを開始します。

```python
wandb.agent(sweep_id, function=main, count=10)
```

## 可視化の結果（オプション）

プロジェクトを開き、W&Bスウィープダッシュボードでライブ結果を確認します。わずか数回のクリックで、[パラレル座標図](../app/features/panels/parallel-coordinates.md),[ パラメーター重要性分析](../app/features/panels/parameter-importance.md)、および[その他のグラフ](../app/features/panels/intro.md)など、充実したインタラクティブなグラフを作成できます。

![Quickstart Sweeps Dashboard example](/images/sweeps/quickstart_dashboard_example.png)

結果を可視化する方法に関する詳細情報は、[スウィープ結果を可視化する](https://docs.wandb.ai/guides/sweeps/visualize-sweep-results).を参照してください。サンプルダッシュボードについては、このサンプル[スウィーププロジェクト](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)をご覧ください。

## エージェントを停止する（オプション）

端末から、Ctrl+cを押して、スウィープエージェントが現在実行しているrunを停止します。エージェントを停止するには、runが停止した後にCtrl+cを再度押します。


