---
description: >-
  Sweeps quickstart shows how to define, initialize, and run a sweep. There are
  four main steps
displayed_sidebar: ja
---

# クイックスタート

<head>
  <title>スイープ クイックスタート</title>
</head>

以下のクイックスタートでは、スイープの定義、初期化、実行の方法を示します。主に4つのステップがあります。

1. [トレーニングコードの設定](#set-up-your-training-code)
2. [スイープ構成で検索空間を定義する](#define-the-search-space-with-a-sweep-configuration)
3. [スイープを初期化する](#initialize-the-sweep)
4. [スイープエージェントを起動する](#start-the-sweep)

次のスイープ クイックスタートのコードをJupyterノートブックまたはPythonスクリプトにコピーして貼り付けてください。

```python
# W&B Pythonライブラリをインポートし、W&Bにログインする
import wandb
wandb.login()

# 1: 目的関数/トレーニング関数を定義する
def objective(config):
    score = config.x ** 3 + config.y
    return score

def main():
    wandb.init(project='my-first-sweep')
    score = objective(wandb.config)
    wandb.log({'score': score})

テキスト:

# 2: 探索空間の定義
sweep_configuration = {
    'method': 'random',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'score'
        },
    'parameters': 
    {
        'x': {'max': 0.1, 'min': 0.01},
        'y': {'values': [1, 3, 7]},
     }
}

# 3: スイープの開始
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='my-first-sweep'
    )

wandb.agent(sweep_id, function=main, count=10)
```

以下のセクションでは、クイックスタートのコードサンプルの各ステップが分解され、説明されています。
## トレーニングコードを設定する
`wandb.config`からハイパーパラメーターの値を取り入れて、モデルをトレーニングし、メトリクスを返すトレーニング関数を定義します。

オプションで、W&B Runの出力が格納されるプロジェクトの名前を指定できます（プロジェクトパラメータは[`wandb.init`](../../ref/python/init.md)にあります）。プロジェクトが指定されていない場合、ランは"Uncategorized"プロジェクトに入ります。

:::caution
W&BスイープとW&Bランは同じプロジェクトに配置する必要があります。そのため、W&Bを初期化する際に指定する名前は、W&Bスイープを初期化する際に指定するプロジェクト名と一致する必要があります。
:::

```python
# 1: 目的/トレーニング関数を定義する
def objective(config):
    score = config.x ** 3 + config.y
    return score

def main():
    wandb.init(project='my-first-sweep')
    score = objective(wandb.config)
    wandb.log({'score': score})
```

## スイープ構成で検索空間を定義する
辞書内で、スイープしたいハイパーパラメーターを指定します。設定オプションの詳細については、[Define sweep configuration](./define-sweep-configuration.md) を参照してください。

以下の例は、ランダムサーチ（`'method':'random'`）を使用したスイープ構成を示しています。スイープは、設定にリストされているバッチサイズ、エポック、学習率の値の組をランダムに選択します。

スイープ全体で、W&Bはメトリックキー（`metric`）で指定されたメトリックを最大化します。以下の例では、W&Bは検証精度（`'val_acc'`）を最大化（`'goal':'maximize'`）します。

```python
# 2: 探索空間を定義する
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

## スイープの初期化

W&Bでは、_スイープコントローラー_を使用して、クラウド（標準）、ローカル（ローカル）で1台以上のマシン間でスイープを管理します。スイープコントローラーについての詳細は、[ローカルでの検索と停止アルゴリズム](./local-controller.md)を参照してください。

スイープを初期化すると、スイープ識別番号が返されます。

```python
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='my-first-sweep'
    )
```

スイープの初期化に関する詳細は、[スイープの初期化](https://docs.wandb.ai/guides/sweeps/initialize-sweeps)を参照してください。

## スイープの開始

`wandb.agent` APIコールを使用して、W&Bスイープを開始します。
```python
wandb.agent(sweep_id, function=main, count=10)
```

## 結果の可視化（オプション）

W&B Sweepダッシュボードで、プロジェクトを開き、ライブ結果を確認します。数回クリックするだけで、[平行座標プロット](../app/features/panels/parallel-coordinates.md)や[パラメータ重要度分析](../app/features/panels/parameter-importance.md)など、豊富なインタラクティブチャートを構築できます。詳細は[こちら](../app/features/panels/intro.md)をご覧ください。

![クイックスタート・スイープダッシュボードの例](/images/sweeps/quickstart_dashboard_example.png)

結果の可視化方法についての詳細は、[スイープ結果の可視化](https://docs.wandb.ai/guides/sweeps/visualize-sweep-results)を参照してください。ダッシュボードの例として、このサンプルの[スイーププロジェクト](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3)をご覧ください。

## エージェントの停止（オプション）

端末から`Ctrl+c`を押して、現在実行中のスイープエージェントのrunを停止します。エージェントを停止するには、runが停止した後に再度`Ctrl+c`を押します。