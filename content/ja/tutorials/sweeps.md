---
title: ハイパーパラメーターをスイープでチューニングする
menu:
  tutorials:
    identifier: ja-tutorials-sweeps
    parent: null
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" >}}

望みのメトリクス（例えばモデルの精度）を満たす機械学習モデルを見つけることは、通常、複数回のイテレーションを必要とする冗長な作業です。さらに悪いことに、特定のトレーニング run にどのハイパーパラメータの組み合わせを使用すべきか不明瞭な場合があります。

W&B Sweeps を使用して、学習率、バッチサイズ、隠れ層の数、オプティマイザーの種類などのハイパーパラメータ値の組み合わせを自動的に検索し、望みのメトリクスに基づいてモデルを最適化するための組織化された効率的な方法を作成します。

このチュートリアルでは、W&B PyTorch インテグレーションを使用してハイパーパラメータ探索を作成します。[ビデオチュートリアル](http://wandb.me/sweeps-video)と一緒に進めてください。

{{< img src="/images/tutorials/sweeps-1.png" alt="" >}}

## Sweeps: 概要

Weights & Biases を使ったハイパーパラメータ探索 run はとても簡単です。以下のシンプルな3ステップで行います：

1. **スイープの定義:** 検索するパラメータ、検索戦略、最適化メトリクスなどを指定する辞書または[YAMLファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})を作成します。

2. **スイープの初期化:** 1行のコードでスイープを初期化し、sweep 設定の辞書を渡します：
`sweep_id = wandb.sweep(sweep_config)`

3. **スイープエージェントの実行:** こちらも1行のコードで完了し、`wandb.agent()`を使って`sweep_id`を渡し、モデルアーキテクチャーを定義してトレーニングする関数と共に実行します：
`wandb.agent(sweep_id, function=train)`

## 始める前に

W&B をインストールし、W&B Python SDK をノートブックにインポートします：

1. `!pip install`を使用してインストールします：

```
!pip install wandb -Uq
```

2. W&B をインポートします：

```
import wandb
```

3. W&B にログインし、プロンプトが表示されたら APIキー を提供します：

```
wandb.login()
```

## ステップ 1️: スイープを定義する

W&B スイープは、ハイパーパラメータ値を試行するための戦略と、それを評価するコードを組み合わせたものです。
スイープを開始する前に、スイープ戦略を_スイープ設定_で定義する必要があります。

{{% alert %}}
Jupyter ノートブックでスイープを開始する場合、スイープ設定はネストされた辞書でなければなりません。

コマンドラインでスイープを行う場合は、[YAMLファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})でスイープ設定を指定する必要があります。
{{% /alert %}}

### 検索メソッドの選択

最初に、設定辞書内でハイパーパラメータ検索メソッドを指定します。[選べるハイパーパラメータ検索戦略は、グリッド、ランダム、ベイズ探索です]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/#method" lang="ja" >}})。

このチュートリアルでは、ランダム検索を使用します。ノートブック内で辞書を作成し、`method` キーに対して `random` を指定します。

```
sweep_config = {
    'method': 'random'
    }
```

最適化したいメトリクスを指定します。ランダム検索メソッドを使用するスイープでは、必ずしもメトリクスとゴールを指定する必要はありません。しかし、スイープの目標を把握しておくことで、後で参照することができるため、良い習慣です。

```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### 検索するハイパーパラメータの指定

スイープ設定で検索メソッドが指定されたので、検索したいハイパーパラメータを指定します。

これを行うには、`parameter` キーに1つ以上のハイパーパラメータ名を指定し、`value` キーに1つ以上のハイパーパラメータ値を指定します。

特定のハイパーパラメータに対して検索する値は、調査しているハイパーパラメータの種類に依存します。

例えば、機械学習オプティマイザーを選択した場合、Adam オプティマイザーや確率的勾配降下法など、1つ以上の有限のオプティマイザー名を指定する必要があります。

```
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'fc_layer_size': {
        'values': [128, 256, 512]
        },
    'dropout': {
          'values': [0.3, 0.4, 0.5]
        },
    }

sweep_config['parameters'] = parameters_dict
```

時にはハイパーパラメータを追跡したいが、その値を変えたくない場合があります。この場合、スイープ設定にハイパーパラメータを追加し、使用したい正確な値を指定します。以下のコードセルでは、`epochs` が1に設定されています。

```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`random` 検索の場合、
特定の run で選ばれるパラメータの`values`は均等に確率が分布しています。

あるいは、
特定の`distribution`を指定し、
例えば`normal` 分布の平均`mu`と標準偏差`sigma`をパラメータとして指定することもできます。

```
parameters_dict.update({
    'learning_rate': {
        # 0から0.1のフラットな分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32から256の整数
        # 対数が均等分布 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

最後に完成した`sweep_config`は、試してみたい`parameters`を具体的に示したネストされた辞書で、
それらを試すための`method`を指定します。

次に、スイープ設定がどのように見えるかを確認しましょう：

```
import pprint
pprint.pprint(sweep_config)
```

設定オプションの全リストについては、[Sweep 設定オプション]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" lang="ja" >}})をご覧ください。

{{% alert %}}
可能性のある無限の選択肢を持つハイパーパラメータの場合
選択した少数の`values`を試してみるのが通常意味があります。例えば、先のスイープ設定では、`layer_size`および`dropout`パラメータキーに対して限定された値のリストが指定されています。
{{% /alert %}}

## ステップ 2️: スイープの初期化

検索戦略を定義したら、それを実装するための準備をします。

W&Bは、クラウドまたはローカルで複数のマシンにわたってスイープを管理する Sweep Controller を使用します。このチュートリアルでは、W&Bにより管理されるスイープコントローラを使用します。

スイープコントローラーはスイープを管理しますが、実際にスイープを実行するコンポーネントは、_sweep agent_として知られています。

{{% alert %}}
デフォルトでは、スイープコントローラのコンポーネントはW&Bのサーバー上で開始され、スイープエージェントはローカルマシン上で作成されます。
{{% /alert %}}

ノートブック内で、`wandb.sweep`メソッドを使用してスイープコントローラをアクティブにできます。先ほど定義したスイープ設定辞書を`sweep_config`フィールドに渡します：

```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

`wandb.sweep`関数は、後のステップでスイープをアクティブ化するために使用する`sweep_id`を返します。

{{% alert %}}
コマンドラインでは、この関数は次のように置き換えられます
```python
wandb sweep config.yaml
```
{{% /alert %}}

ターミナルで W&B Sweeps を作成する方法についての詳細は、[W&B Sweep ウォークスルー]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}})を参照してください。

## ステップ 3: 機械学習コードを定義する

スイープを実行する前に、
試してみたいハイパーパラメータ値を使用するトレーニング手順を定義します。W&B Sweepsをトレーニングコードに統合するための鍵は、各トレーニング実験のために、あなたのトレーニングロジックがスイープ設定で定義されたハイパーパラメータ値にアクセスできるようにすることです。

次のコード例では、ヘルパー関数`build_dataset`、`build_network`、`build_optimizer`、および`train_epoch`が、スイープハイパーパラメータ設定辞書にアクセスします。

次の機械学習トレーニングコードをノートブック内で実行します。これらの関数は、PyTorchでの基本的な全結合ニューラルネットワークを定義しています。

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 新しい wandb run を初期化する
    with wandb.init(config=config):
        # wandb.agent によって呼ばれた場合、以下のようにスイープコントローラによってこの構成が設定されます
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

`train`関数内では、次のW&B Python SDKメソッドが見られます:
* [`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}): 新しい W&B run を初期化します。各 run は、トレーニング関数の1回の実行です。
* [`wandb.config`]({{< relref path="/guides/models/track/config" lang="ja" >}}): 試してみたいハイパーパラメータを含むスイープ設定を渡します。
* [`wandb.log()`]({{< relref path="/ref/python/log" lang="ja" >}}): 各エポックのトレーニングlossをログに記録します。

次のセルでは、4つの関数を定義します:
`build_dataset`、`build_network`、`build_optimizer`、および`train_epoch`。 
これらの関数は基本的なPyTorchパイプラインの標準的な部分であり、
W&Bの使用によってその実装に影響しません。

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # MNISTトレーニングデータセットをダウンロード
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # 完全連結のシングル隠れ層
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))

    return network.to(device)
        

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

PyTorch を使用した W&B のインストゥルメントについての詳細は、[この Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)を参照してください。

## ステップ 4: スイープエージェントのアクティベート

スイープ設定が定義され、ハイパーパラメータを対話的に利用できるトレーニングスクリプトが準備できたので、スイープエージェントをアクティベートする準備ができました。スイープエージェントは、スイープ設定で定義されたハイパーパラメータ値のセットを使用して実験を実行する責任を負っています。

`sweep_id`を持つスイープの一部として、スイープが実行する関数（この例では`train`関数を使用）、（オプションで）スイープコントローラーに何個の設定を要求するかを指定してください。

{{% alert %}}
同じ`sweep_id`を持つスイープエージェントを異なる計算リソース上で起動することができます。スイープコントローラーがあなたの定義したスイープ設定に基づいて協働することを保証します。
{{% /alert %}}

次のセルは、トレーニング関数（`train`）を5回実行するスイープエージェントをアクティベートします：

```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
スイープ設定で`random` 検索メソッドが指定されていたため、スイープコントローラーはランダムに生成されたハイパーパラメータ値を提供します。
{{% /alert %}}

ターミナルで W&B Sweeps を作成する方法についての詳細は、[W&B Sweep ウォークスルー]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}})を参照してください。

## スイープ結果の可視化

### Parallel Coordinates プロット
このプロットは、ハイパーパラメータ値をモデルメトリクスにマッピングします。最も良いモデルパフォーマンスをもたらしたハイパーパラメータの組み合わせを特定するのに役立ちます。

{{< img src="/images/tutorials/sweeps-2.png" alt="" >}}

### ハイパーパラメータの重要度プロット
ハイパーパラメータの重要度プロットは、あなたのメトリクスの最良の予測因子だったハイパーパラメータを表面化します。
特徴重要度（ランダムフォレストモデルによる）と相関関係（暗黙の線形モデル）を報告します。

{{< img src="/images/tutorials/sweeps-3.png" alt="" >}}

これらの可視化は、最も重要であり、それによって更に探求する価値があるパラメータ（および値範囲）に絞り込むことによって、高価なハイパーパラメータ最適化にかかる時間とリソースを節約するのに役立ちます。

## W&B Sweeps についてもっと学ぶ

私たちは、あなたが試してみるためのシンプルなトレーニングスクリプトと[いくつかのスイープ設定のバリエーション](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)を用意しました。これらを試すことを強くお勧めします。

そのリポジトリには、より高度なスイープ機能を試すための例もあります。例えば、[Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla)と[Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla)が含まれています。