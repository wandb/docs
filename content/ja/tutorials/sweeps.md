---
title: Tune hyperparameters with sweeps
menu:
  tutorials:
    identifier: ja-tutorials-sweeps
    parent: null
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" >}}

機械学習モデルを探し、望むメトリック（モデルの精度など）を達成することは通常、多くの繰り返しを伴う冗長な作業です。さらに悪いことに、特定のトレーニング run にどのハイパーパラメーターの組み合わせを使用するべきか不明であることがあります。

W&B Sweeps を使用すると、学習率、バッチサイズ、隠れ層の数、オプティマイザーのタイプなどのハイパーパラメーターの値の組み合わせを自動的に検索し、望むメトリックに基づいてモデルを最適化する値を見つける効率的で整理された方法を作成できます。

このチュートリアルでは、W&B PyTorch インテグレーションを使用してハイパーパラメーター探索を作成します。 [ビデオチュートリアル](http://wandb.me/sweeps-video) に沿って進めてください。

{{< img src="/images/tutorials/sweeps-1.png" alt="" >}}

## Sweeps: 概要

Weights & Biases を使用したハイパーパラメーター探索を実行することは非常に簡単です。3つのシンプルなステップだけです。

1. **sweepを定義する：** 辞書または [YAMLファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) を作成し、検索するパラメータ、検索戦略、最適化メトリックなどを指定して行います。

2. **sweepを初期化する：** 一行のコードでsweepを初期化し、sweep設定の辞書を渡します: `sweep_id = wandb.sweep(sweep_config)`

3. **sweep agent を実行する：** これも一行のコードで達成され、`wandb.agent()` を呼び出し、sweep_idとモデルアーキテクチャーを定義し、トレーニングを行う関数を渡します: `wandb.agent(sweep_id, function=train)`

## 始める前に

W&B をインストールし、W&B Python SDK をノートブックにインポートします：

1. `!pip install` でインストール:


```
!pip install wandb -Uq
```

2. W&B をインポートします:


```
import wandb
```

3. W&B にログインし、プロンプトが表示されたら APIキー を提供します:


```
wandb.login()
```

## ステップ 1️: sweep を定義する

W&B Sweep は、試行する多数のハイパーパラメーター値の戦略と、それらを評価するコードを組み合わせたものです。
sweep を開始する前に、 _sweep 設定_ で sweep 戦略を定義する必要があります。

{{% alert %}}
Jupyter Notebook で sweep を開始する場合、作成した sweep 設定はネストされた辞書でなければなりません。

コマンドラインで sweep を実行する場合、[YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) で sweep 設定を指定する必要があります。
{{% /alert %}}

### 検索メソッドを選択する

まず、設定辞書内でハイパーパラメーター検索メソッドを指定します。 [選択できるハイパーパラメーター検索戦略は3つあります: グリッド、ランダム、ベイズ探索]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/#method" lang="ja" >}})。

このチュートリアルでは、ランダム検索を使用します。ノートブック内で辞書を作成し、`method` キーに `random` を指定します。


```
sweep_config = {
    'method': 'random'
    }
```

最適化したいメトリックを指定します。ランダム検索メソッドを使用するsweepについては、メトリックと目標を指定する必要はありませんが、後から参照できるようにsweep目標を追跡することは良い習慣です。


```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### 探索するハイパーパラメーターを指定する

sweep 設定で検索メソッドが指定されたので、次に探索したいハイパーパラメーターを指定します。

これを行うには、`parameter` キーに1つ以上のハイパーパラメーター名を指定し、`value` キーに1つ以上のハイパーパラメーター値を指定します。

特定のハイパーパラメーターの値は、調査しているハイパーパラメーターのタイプに依存します。

例えば、機械学習オプティマイザーを選択する場合、Adamオプティマイザーやスタカスティック勾配降下などの有限オプティマイザ名を指定する必要があります。


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

ハイパーパラメーターを追跡したいがその値を変えたくない場合があります。 この場合、使用したい正確な値を指定してsweep 設定にハイパーパラメーターを追加します。 以下のコードセルでは、`epochs` が 1 にセットされています。


```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`random` 検索では、指定された run の候補が等しく選ばれる可能性があります。

または、
名前付き `distribution` や、`normal` 分布の平均 `mu`
および標準偏差 `sigma` のようなパラメータを指定できます。


```
parameters_dict.update({
    'learning_rate': {
        # 0から0.1の間の一様分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32から256の整数（対数が均等に分布）
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

完了すると、`sweep_config` はネストされた辞書で、
試したい `parameters` と、それを試す `method` を正確に指定します。

sweep 設定がどう見えるか見てみましょう。


```
import pprint
pprint.pprint(sweep_config)
```

設定の全オプションを閲覧するには、[sweep 設定オプション]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" lang="ja" >}}) を参照してください。

{{% alert %}}
潜在的に無限のオプションを持つハイパーパラメーターの場合、
いくつかの選択した`values`を試すのが通常は意味があります。例えば、前述の sweep 設定では、`layer_size` および `dropout` パラメータキーに指定された有限の値のリストを持っています。
{{% /alert %}}

## ステップ 2️: Sweepを初期化する

検索戦略を定義したら、それを実現するための設定を行う時が来ました。

W&B では、クラウドまたは複数のマシンでローカルに sweeps を管理するための Sweep Controller を使用します。このチュートリアルでは、W&B によって管理されるスイープコントローラーを使用します。

sweep controllers が sweeps を管理する間、実際にsweepを実行するコンポーネントは _sweep agent_ として知られています。

{{% alert %}}
デフォルトで、sweep controllers コンポーネントは W&B のサーバー上で開始され、sweep agent、つまり sweepを作成するコンポーネントはローカルマシン上でアクティブ化されます。
{{% /alert %}}

ノートブック内で、`wandb.sweep` メソッドを使用して sweep controller をアクティベートできます。前に定義した sweep 設定の辞書を `sweep_config` フィールドに渡してください:


```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

`wandb.sweep` 関数は、後のステップで sweep をアクティベートするために使用する `sweep_id` を返します。

{{% alert %}}
コマンドライン上では、この関数は次のようになります
```python
wandb sweep config.yaml
```
{{% /alert %}}

ターミナルでの W&B sweeps の作成方法に関する詳細は、[W&B Sweep ウォークスルー]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}}) を参照してください。


## ステップ 3: 機械学習コードを定義する

sweep を実行する前に、
試したいハイパーパラメーター値を使用するトレーニングプロセスを定義します。 W&B Sweeps をトレーニングコードに統合するための鍵は、各トレーニング実験ごとに、トレーニングロジックがあなたが sweep 設定で定義したハイパーパラメーター値にアクセスできるようにすることです。

以下のコード例では、`build_dataset`、`build_network`、`build_optimizer`、および `train_epoch` ヘルパー関数が sweep ハイパーパラメーター設定辞書にアクセスします。

ノートブック内で以下の機械学習トレーニングコードを実行します。これらの関数は、PyTorch で基本的な全結合ニューラルネットワークを定義します。


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
        # wandb.agent から呼び出された場合、以下のように、
        # この設定はスイープコントローラーによって設定されます
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

`train` 関数内では、次の W&B Python SDK メソッドが確認されます:
* [`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}): 新しい W&B run を初期化する。各 run はトレーニング関数の単一の実行です。
* [`wandb.config`]({{< relref path="/guides/models/track/config" lang="ja" >}}): ハイパーパラメーターで試す設定を渡す。
* [`wandb.log()`]({{< relref path="/ref/python/log" lang="ja" >}}): エポックごとのトレーニング損失をログに記録します。

以下のセルは四つの関数を定義します：
`build_dataset`、`build_network`、`build_optimizer`、および `train_epoch`。 
これらの関数は基本的な PyTorch パイプラインの一部であり、W&B の使用によってその実装が影響を受けることはありません。


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
    network = nn.Sequential(  # 全結合、単一の隠れ層
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

PyTorch を使用して W&B を導入する詳細については、[この Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb) を参照してください。

## ステップ 4: sweep agent をアクティベートする
sweep 設定を定義し、これらのハイパーパラメーターをインタラクティブに利用できるトレーニングスクリプトを用意したので、sweep agent をアクティベートする準備が整いました。sweep agent は、sweep 設定で定義した一連のハイパーパラメーター値を使用して実験を実行する役割を担っています。

`sweep_id` を使用して `wandb.agent` メソッドで sweep agent を作成します。以下を提供してください：
1. エージェントが属する sweep (`sweep_id`)
2. sweep が実行することになっている関数。この例では、sweep は `train` 関数を使用します。
3. （オプション）sweep コントローラーから要求する設定の数 (`count`)

{{% alert %}}
同じ `sweep_id` を使って
異なるコンピュータリソース上で複数の sweep agent を開始できます。sweep コントローラーが、それらが指定したsweep設定に従って連携することを保証します。
{{% /alert %}}

次のセルは、トレーニング関数 (`train`) を5回実行するsweep agent をアクティベートします：


```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
sweep 設定で `random` 検索メソッドが指定されていたため、sweep コントローラーはランダムに生成されたハイパーパラメーターの値を提供します。
{{% /alert %}}

ターミナルでの W&B sweeps の作成方法に関する詳細は、[W&B Sweep ウォークスルー]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}}) を参照してください。

## Sweep 結果を可視化する

### Parallel Coordinates Plot
このプロットは、ハイパーパラメーターの値をモデルメトリクスにマッピングします。最良のモデルパフォーマンスをもたらすハイパーパラメーターの組み合わせに集中するのに役立ちます。

{{< img src="/images/tutorials/sweeps-2.png" alt="" >}}

### ハイパーパラメーターインポータンスプロット
ハイパーパラメータの重要性プロットは、どのハイパーパラメータがメトリクスの最高の予測因子であったかを浮き彫りにします。
特徴量の重要性（ランダムフォレストモデルから）および相関関係（暗黙的に線形モデル）を報告します。

{{< img src="/images/tutorials/sweeps-3.png" alt="" >}}

これらの可視化は、最も重要と考えられるパラメータ (および値の範囲) に集中的に取り組むことで、高価なハイパーパラメーター最適化の時間とリソースを節約するのに役立ちます。

## W&B Sweeps についてもっと学ぶ

私たちは、シンプルなトレーニングスクリプトと [いくつかのスイープ設定のバリエーション](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion) を提供しています。ぜひ試してみてください。

そのリポジトリには、[Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla) や [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla) など、より高度なスイープ機能を試すための例も含まれています。