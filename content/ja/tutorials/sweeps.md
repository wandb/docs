---
title: Tune hyperparameters with sweeps
menu:
  tutorials:
    identifier: ja-tutorials-sweeps
    parent: null
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" >}}

望ましいメトリクス (モデルの精度など) を満たす 機械学習 モデルを見つけることは、通常、複数のイテレーションを要する冗長な作業です。さらに悪いことに、特定のトレーニング run にどのハイパーパラメーターの組み合わせを使用すればよいか不明確な場合があります。

W&B Sweeps を使用して、学習率、 バッチサイズ 、隠れ層の数、 オプティマイザー の種類など、ハイパーパラメーター 値の組み合わせを自動的に検索し、目的のメトリクスに基づいてモデルを最適化する値を 見つけるための、組織化された効率的な方法を作成します。

このチュートリアルでは、W&B PyTorch インテグレーションを使用してハイパーパラメーター探索を作成します。[ビデオチュートリアル](http://wandb.me/sweeps-video)をご覧ください。

{{< img src="/images/tutorials/sweeps-1.png" alt="" >}}

## Sweeps: 概要

Weights & Biases でハイパーパラメーター sweep を実行するのは非常に簡単です。簡単な3つのステップがあります。

1. **sweep を定義する:** 検索する パラメータ 、検索戦略、最適化 メトリクス などを指定する 辞書 または [YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) を作成してこれを行います。

2. **sweep を初期化する:** 1行の コード で sweep を初期化し、sweep 設定の 辞書 を渡します。
`sweep_id = wandb.sweep(sweep_config)`

3. **sweep agent を実行する:** これも1行の コード で実現します。`wandb.agent()` を呼び出し、実行する `sweep_id` と、モデル アーキテクチャー を定義してトレーニングする関数を渡します。
`wandb.agent(sweep_id, function=train)`

## 開始する前に

W&B をインストールし、W&B Python SDK を ノートブック にインポートします。

1. `!pip install` でインストールします。

```
!pip install wandb -Uq
```

2. W&B をインポートします。

```
import wandb
```

3. W&B にログインし、プロンプトが表示されたら APIキー を入力します。

```
wandb.login()
```

## ステップ 1️: sweep を定義する

W&B Sweep は、多数のハイパーパラメーター値を試すための戦略と、それらを評価する コード を組み合わせたものです。
sweep を開始する前に、 _sweep 設定_ で sweep 戦略を定義する必要があります。

{{% alert %}}
Jupyter Notebook で sweep を開始する場合、sweep 用に作成する sweep 設定はネストされた 辞書 にする必要があります。

コマンドライン で sweep を実行する場合は、[YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}})で sweep 設定を指定する必要があります。
{{% /alert %}}

### 検索方法を選択する

まず、設定 辞書 内でハイパーパラメーター検索方法を指定します。[グリッド、 ランダム 、 ベイズ探索 の3つのハイパーパラメーター検索戦略から選択できます]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/#method" lang="ja" >}})。

このチュートリアルでは、 ランダム 検索を使用します。 ノートブック 内で、 辞書 を作成し、`method` キーに `random` を指定します。

```
sweep_config = {
    'method': 'random'
    }
```

最適化する メトリクス を指定します。 ランダム 検索方法を使用する sweep の メトリクス と目標を指定する必要はありません。ただし、後で参照できるため、sweep の目標を追跡することをお勧めします。

```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### 検索するハイパーパラメーターを指定する

これで、sweep 設定で検索方法が指定されたので、検索するハイパーパラメーターを指定します。

これを行うには、1つ以上のハイパーパラメーター名を `parameter` キーに指定し、`value` キーに1つ以上のハイパーパラメーター値を指定します。

特定のハイパーパラメーターで検索する値は、調査するハイパーパラメーターの種類によって異なります。

たとえば、 機械学習 オプティマイザー を選択する場合は、Adam オプティマイザー や確率的勾配降下など、1つ以上の有限 オプティマイザー 名を指定する必要があります。

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

ハイパーパラメーターを追跡したいが、その値を変更したくない場合があります。この場合、ハイパーパラメーターを sweep 設定に追加し、使用する正確な値を指定します。たとえば、次の コード セルでは、`epochs` が 1 に設定されています。

```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`random` 検索の場合、
パラメーターのすべての `values` は、特定の run で選択される可能性が等しくなります。

あるいは、
名前付きの `distribution` と、その パラメータ (平均 `mu` など)
および `normal` 分布の標準偏差 `sigma` を指定できます。

```
parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        # 0 ～ 0.1 のフラットな分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # integers between 32 and 256
        # 32 ～ 256 の整数
        # with evenly-distributed logarithms
        # 対数が均等に分布している
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

完了すると、`sweep_config` はネストされた 辞書 になり、
試してみたい `parameters` と、それらを試すために使用する `method` を正確に指定します。

sweep 設定がどのように見えるか見てみましょう。

```
import pprint
pprint.pprint(sweep_config)
```

設定オプションの完全なリストについては、[Sweep 設定オプション]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" lang="ja" >}})を参照してください。

{{% alert %}}
潜在的に無限のオプションがあるハイパーパラメーターの場合、
通常、いくつかの選択した `values` を試してみるのが理にかなっています。たとえば、前の sweep 設定には、`layer_size` および `dropout` パラメータ キーに指定された有限値のリストがあります。
{{% /alert %}}

## ステップ 2️: Sweep を初期化する

検索戦略を定義したら、それを実装するものを設定する時が来ました。

W&B は、 Sweep コントローラ ーを使用して、 クラウド 上またはローカルで1つまたは複数のマシンにわたって sweep を管理します。このチュートリアルでは、W&B によって管理される sweep コントローラ ーを使用します。

sweep コントローラ ーは sweep を管理しますが、実際に sweep を実行するコンポーネントは _sweep agent_ と呼ばれます。

{{% alert %}}
デフォルトでは、sweep コントローラ ーコンポーネントは W&B の サーバー で開始され、 sweep を作成するコンポーネントである sweep agent はローカルマシンでアクティブ化されます。
{{% /alert %}}

ノートブック 内から、`wandb.sweep` メソッドを使用して sweep コントローラ ーをアクティブ化できます。以前に定義した sweep 設定 辞書 を `sweep_config` フィールドに渡します。

```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

`wandb.sweep` 関数は、 sweep をアクティブ化するために後のステップで使用する `sweep_id` を返します。

{{% alert %}}
コマンドライン では、この関数は次のように置き換えられます。
```python
wandb sweep config.yaml
```
{{% /alert %}}

ターミナル で W&B Sweeps を作成する方法の詳細については、[W&B Sweep チュートリアル]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}})を参照してください。

## ステップ 3: 機械学習 コード を定義する

sweep を実行する前に、
試したいハイパーパラメーター値を使用するトレーニング手順を定義します。W&B Sweeps をトレーニング コード に統合するための鍵は、各トレーニング 実験 で、トレーニング ロジックが sweep 設定で定義したハイパーパラメーター値にアクセスできることを確認することです。

次の コード 例では、ヘルパー関数 `build_dataset`、`build_network`、`build_optimizer`、および `train_epoch` が sweep ハイパーパラメーター設定 辞書 にアクセスします。

次の 機械学習 トレーニング コード を ノートブック で実行します。この関数は、PyTorch で基本的な完全接続 ニューラルネットワーク を定義します。

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # Initialize a new wandb run
    # 新しい wandb run を初期化します
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        # wandb.agent によって呼び出された場合、以下に示すように、
	# この設定は Sweep コントローラ ーによって設定されます
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

`train` 関数内では、次の W&B Python SDK メソッドに気付くでしょう。
* [`wandb.init()`]({{< relref path="/ref/python/init" lang="ja" >}}): 新しい W&B run を初期化します。各 run は、トレーニング関数の単一の実行です。
* [`wandb.config`]({{< relref path="/guides/models/track/config" lang="ja" >}}): 実験するハイパーパラメーターを使用して sweep 設定を渡します。
* [`wandb.log()`]({{< relref path="/ref/python/log" lang="ja" >}}): 各 エポック のトレーニング損失を ログ に記録します。

次のセルは、4つの関数を定義します。
`build_dataset`、`build_network`、`build_optimizer`、および `train_epoch`。
これらの関数は、基本的な PyTorch パイプライン の標準的な部分であり、
その実装は W&B の使用によって影響を受けません。

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    # MNIST トレーニング データセット をダウンロードする
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # fully connected, single hidden layer
        # 全結合、単一隠れ層
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
        # ➡ 順伝播
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ Backward pass + weight update
        # ⬅ 逆伝播 + 重み更新
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

PyTorch との W&B の連携の詳細については、[この Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb) を参照してください。

## ステップ 4: sweep agent をアクティブ化する
sweep 設定が定義され、これらのハイパーパラメーターをインタラクティブな方法で利用できるトレーニング スクリプト があるので、 sweep agent をアクティブ化する準備ができました。 Sweep agent は、 sweep 設定で定義した一連のハイパーパラメーター値を使用して 実験 を実行する役割を担います。

`wandb.agent` メソッドを使用して sweep agent を作成します。以下を提供します。
1. agent が所属する sweep (`sweep_id`)
2. sweep が実行することになっている関数。この例では、 sweep は `train` 関数を使用します。
3. (オプション) sweep コントローラ ーに要求する設定の数 (`count`)

{{% alert %}}
同じ `sweep_id` で複数の sweep agent を開始できます
異なるコンピューティング リソース 上で。 sweep コントローラ ーは、定義した sweep 設定に従ってそれらが連携するようにします。
{{% /alert %}}

次のセルは、トレーニング関数 (`train`) を5回実行する sweep agent をアクティブ化します。

```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
`random` 検索方法が sweep 設定で指定されているため、 sweep コントローラ ーは ランダム に生成されたハイパーパラメーター値を提供します。
{{% /alert %}}

ターミナル で W&B Sweeps を作成する方法の詳細については、[W&B Sweep チュートリアル]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}})を参照してください。

## Sweep の結果を 可視化 する

### 平行座標プロット
このプロットは、ハイパーパラメーター値をモデル メトリクス にマッピングします。最高のモデル パフォーマンス につながったハイパーパラメーターの組み合わせを絞り込むのに役立ちます。

{{< img src="/images/tutorials/sweeps-2.png" alt="" >}}

### ハイパーパラメーター インポータンスプロット
ハイパーパラメーター インポータンスプロット は、どのハイパーパラメーターが メトリクス の最適な予測因子であるかを表面化します。
特徴量の重要度 ( ランダム フォレスト モデル から) と相関関係 (暗黙的に 線形モデル ) を報告します。

{{< img src="/images/tutorials/sweeps-3.png" alt="" >}}

これらの 可視化 は、最も重要な パラメータ (および値の範囲) を絞り込むことで、時間とリソースを節約し、さらに調査する価値のある高価なハイパーパラメーター最適化を実行するのに役立ちます。

## W&B Sweeps の詳細

簡単なトレーニング スクリプト と、試してみるための [いくつかの種類の sweep 設定](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion) を作成しました。これらを試してみることを強くお勧めします。

その リポジトリ には、[ ベイズ ハイパーバンド](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla) や [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla) などの、より高度な sweep 機能を試すのに役立つ例もあります。
