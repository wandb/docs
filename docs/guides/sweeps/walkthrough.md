---
description: Sweeps クイックスタートでは、sweep の定義、初期化、および実行方法を示します。主な手順は4つあります。
displayed_sidebar: default
---


# Walkthrough

<head>
  <title>Sweeps Walkthrough</title>
</head>

このページでは、sweep を定義し、初期化し、実行する方法を示します。主に4つのステップがあります：

1. [トレーニングコードを設定する](#set-up-your-training-code)
2. [sweep の設定で探索空間を定義する](#define-the-search-space-with-a-sweep-configuration)
3. [sweep を初期化する](#initialize-the-sweep)
4. [sweep agent を開始する](#start-the-sweep)

以下のコードを Jupyter Notebook か Python スクリプトにコピーして貼り付けてください：

```python 
# W&B Python ライブラリをインポートし、W&B にログインします
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

以下のセクションでは、コードサンプル内の各ステップについて説明しています。

## Set up your training code
`wandb.config` からハイパーパラメーターの値を受け取り、それらを使ってモデルをトレーニングし、メトリクスを返すトレーニング関数を定義します。

オプションとして、W&B Run の出力を保存するプロジェクトの名前を指定することもできます（[`wandb.init`](../../ref/python/init.md)の project パラメータ）。プロジェクトが指定されていない場合、run は "Uncategorized" プロジェクトに保存されます。

:::tip
sweep と run の両方が同じプロジェクトに属している必要があります。したがって、W&B を初期化するときに指定する名前は、sweep を初期化するときに指定するプロジェクトの名前と一致する必要があります。
:::

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

## Define the search space with a sweep configuration
辞書内で、どのハイパーパラメーターを対象にするかを指定します。設定オプションの詳細については、[Define sweep configuration](./define-sweep-configuration.md)を参照してください。

以下の例では、ランダム検索方法（`'method':'random'`）を使用した sweep 設定を示しています。sweep は、バッチサイズ、エポック、および学習率の設定に記載されたランダムな値セットをランダムに選択します。

sweeps を通じて、W&B はメトリクスキーで指定されたメトリクスを最大化します。以下の例では、W&B は検証精度（`'val_acc'`）を最大化（`'goal':'maximize'`）するように設定されています。

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

## Initialize the Sweep

W&B は _Sweep Controller_ を使用して、クラウド（標準）、ローカル（local）、または複数のマシンで sweeps を管理します。Sweep Controller についての詳細は、[Search and stop algorithms locally](./local-controller.md)を参照してください。

sweep を初期化すると、sweep 識別番号が返されます：

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

sweep の初期化についての詳細は、[Initialize sweeps](./initialize-sweeps.md)を参照してください。

## Start the Sweep

`sweep agent` を開始するには、[`wandb.agent`](../../ref/python/agent.md) API 呼び出しを使用します。

```python
wandb.agent(sweep_id, function=main, count=10)
```

## Visualize results (optional)

プロジェクトを開いて、W&B アプリのダッシュボードでライブ結果を確認します。わずか数クリックで、[パラレル座標図](../app/features/panels/parallel-coordinates.md)、[パラメータの重要性分析](../app/features/panels/parameter-importance.md)、[その他](../app/features/panels/intro.md)のようなリッチでインタラクティブなグラフを作成できます。

![Sweeps Dashboard example](/images/sweeps/quickstart_dashboard_example.png)

結果の可視化方法についての詳細は、[Visualize sweep results](./visualize-sweep-results.md)を参照してください。ダッシュボードの例については、このサンプル [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3) を参照してください。

## Stop the agent (optional)

ターミナルから `Ctrl+c` を押して、現在の sweep agent の run を停止します。agent を終了するには、run を停止した後に再度 `Ctrl+c` を押します。