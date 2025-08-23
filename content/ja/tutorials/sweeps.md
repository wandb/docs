---
title: スイープでハイパーパラメーターをチューニングする
menu:
  tutorials:
    identifier: ja-tutorials-sweeps
    parent: null
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" >}}

目的のメトリクス（例えばモデルの精度など）を満たす機械学習モデルを見つけるには、何度も反復しながら試行錯誤する必要があり、手間が掛かります。さらに、どのハイパーパラメーターの組み合わせを使えばよいか分かりにくいことも多いです。

W&B Sweeps を使えば、学習率、バッチサイズ、隠れ層数、オプティマイザーの種類など、様々なハイパーパラメーターの組み合わせを自動的に探索し、希望のメトリクスを最適化する値を効率的に見つけることができます。

このチュートリアルでは、W&B と PyTorch のインテグレーションを使ってハイパーパラメーター探索を構築します。[ビデオチュートリアル](https://wandb.me/sweeps-video)もご覧ください。

{{< img src="/images/tutorials/sweeps-1.png" alt="Hyperparameter sweep results" >}}

## Sweeps: 概要

W&B を使ったハイパーパラメーター探索（Sweep）はとても簡単です。以下の3ステップで実施できます。

1. **sweep の定義:** 検索したいパラメータ、探索手法、最適化したいメトリクスなどを指定した辞書や [YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) を作成します。

2. **sweep の初期化:** 1行のコードで sweep を初期化し、作成した sweep 設定の辞書を渡します。
   `sweep_id = wandb.sweep(sweep_config)`

3. **sweep エージェントの実行:** これも1行のコードで `wandb.agent()` を呼び出し、`sweep_id` と、モデルアーキテクチャおよびトレーニングを定義する関数を渡します。
   `wandb.agent(sweep_id, function=train)`


## はじめる前に

W&B をインストールし、Python SDK をノートブックにインポートしましょう。

1. `!pip install` でインストール:

```
!pip install wandb -Uq
```

2. W&B をインポート:

```
import wandb
```

3. W&B にログインし、プロンプトに従って APIキー を入力:

```
wandb.login()
```

## ステップ 1️: sweep を定義する

W&B Sweep は、様々なハイパーパラメーター値を試す戦略と、それらを評価するコードを組み合わせたものです。
sweep を開始する前に、まず _sweep 設定_ で探索の戦略を定義します。

{{% alert %}}
Jupyter Notebook で sweep を開始する場合、作成する sweep 設定はネストした辞書にする必要があります。

コマンドラインで sweep を実行する場合は、[YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) で sweep 設定を指定してください。
{{% /alert %}}

### 検索手法を選択する

まず、設定用辞書内にハイパーパラメーターの検索手法を指定します。[ハイパーパラメーター検索戦略は3種類あります：grid、random、ベイズ探索]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/#method" lang="ja" >}})。

このチュートリアルでは random（ランダム検索）を使用します。ノートブック内で辞書を作成し、`method` キーには `random` を指定します。

```
sweep_config = {
    'method': 'random'
    }
```

最適化したいメトリクスも指定しておきましょう。なお、random 検索手法を使う場合は、必ずしもメトリクスやゴールを明示的に指定する必要はありませんが、後から参照できるように記録しておくのがおすすめです。

```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### 検索対象となるハイパーパラメータを指定する

これで sweep 設定内の検索手法が指定できたので、続いて探索するハイパーパラメーターを指定します。

これには、`parameter` キーに1つ以上のハイパーパラメータ名を指定し、その `value` キーに1つ以上のパラメータ値を指定します。

あるハイパーパラメータで探索する値は、そのパラメータの種類によって変わります。

例えば、機械学習のオプティマイザーを選ぶ場合は、Adam オプティマイザーや SGD など、有限個のオプティマイザー名を指定します。

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

ハイパーパラメーターによっては値を固定にして変化させないまま記録だけしたい場合もあります。その場合は sweep 設定にそのパラメーターを追加し、使いたい値を `value` に指定します。下記の例では `epochs` を1に固定しています。

```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`random` 検索の場合、
パラメータの `values` の中からどれが選ばれるかは run 毎に等確率です。

または、
名前付き `distribution`（分布） を指定し、
そのパラメータ（例えば平均値 `mu` や標準偏差 `sigma` など）を記述することもできます。

```
parameters_dict.update({
    'learning_rate': {
        # 0〜0.1 の一様分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32〜256 の整数値を対数的に均等分布で
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

これで、`sweep_config` は
「どのパラメータを試すのか」と
「何の手法で試すのか」を完全に指定した
ネストされた辞書になります。

sweep 設定の全体像はこのようになります：

```
import pprint
pprint.pprint(sweep_config)
```

設定項目の詳細については [Sweep 設定オプション一覧]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" lang="ja" >}}) をご参照ください。

{{% alert %}}
無限に近い選択肢が考えられるハイパーパラメータの場合は、
候補となる `values` をいくつか絞って指定するのが一般的です。前述の sweep 設定では、`layer_size` や `dropout` パラメータで限られた値のリストを指定しています。
{{% /alert %}}

## ステップ 2️: Sweep の初期化

探索戦略を定義できたら、それを実際に適用する準備を行いましょう。

W&B では、スイープの管理に Sweep Controller を使います。クラウドやローカル複数台で動作可能です。今回は W&B が管理する sweep controller を使います。

なお、sweep controller がスイープ全体を管理し、
実際に sweep を実行するのは _sweep agent_ です。

{{% alert %}}
デフォルトでは、sweep controller は W&B のサーバー上で起動し、sweep agent（実際にスイープを作成する役割）はローカルマシン上で起動します。
{{% /alert %}}

ノートブック内では `wandb.sweep` メソッドで sweep controller を有効化できます。先ほど作成した sweep_config 辞書を渡しましょう。

```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

`wandb.sweep` 関数は `sweep_id` を返します。これは次のステップで sweep を有効化する際に使います。

{{% alert %}}
コマンドラインでは、この関数の代わりに
```python
wandb sweep config.yaml
```
を使用します。
{{% /alert %}}

ターミナル上で W&B Sweeps を使う方法の詳細は [W&B Sweep ウォークスルー]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}}) をご覧ください。

## ステップ 3: 機械学習コードを定義する

sweep を実行する前に、
試したいハイパーパラメータ値でトレーニングを行う処理を定義する必要があります。W&B Sweeps とトレーニングコードを連携させるポイントは、各トレーニング実験ごとに、トレーニング処理が sweep 設定内で定義したハイパーパラメータ値に_確実にアクセスできる_ようにすることです。

下記のコード例では、`build_dataset`, `build_network`, `build_optimizer`, `train_epoch` の各補助関数が、sweep のハイパーパラメータ設定辞書にアクセスして処理します。

以下のトレーニングコードをノートブックで実行してください。関数群は PyTorch で単純な全結合ニューラルネットワークを定義します。

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 新しい wandb run を初期化
    with wandb.init(config=config) as run:
        # 下記のように wandb.agent から呼ばれた場合、
        # この config は Sweep Controller によって指定される
        config = run.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            run.log({"loss": avg_loss, "epoch": epoch})           
```

`train` 関数内では、W&B Python SDK の以下のメソッドを使用しています：
* [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init/" lang="ja" >}}): 新しい W&B Run を初期化します。1回の run が1度のトレーニング関数実行に該当します。
* [`run.config`]({{< relref path="/guides/models/track/config" lang="ja" >}}): sweep で指定したハイパーパラメータ群をトレーニング関数に渡す役割。
* [`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}): 各エポックのトレーニング損失を記録します。

さらに下記の4つの関数
`build_dataset`, `build_network`, `build_optimizer`, `train_epoch` を定義しています。
これらは PyTorch の標準的なパイプラインによくある処理で、
W&B 利用の有無による違いはありません。

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # MNISTトレーニングデータセットのダウンロード
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # 完全結合1層のシンプルなネットワーク
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

    with wandb.init() as run:
        for _, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # ➡ Forward pass
            loss = F.nll_loss(network(data), target)
            cumu_loss += loss.item()

            # ⬅ Backward pass + 重み更新
            loss.backward()
            optimizer.step()

            run.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

PyTorch 向けの W&B 計測の詳細は [こちらの Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb) をご参照ください。

## ステップ 4: sweep agent を有効化する

ここまでで sweep 設定と、それを活用できるトレーニングスクリプトが完成しました。いよいよ sweep agent を有効化しましょう。sweep agent は、sweep 設定内で定義したハイパーパラメーター値で実験（run）を実行してくれます。

`sweep_id` が属する sweep を 1. 指定し、
2. 実行する関数（例では `train`）、
3. （オプションで）コントローラーに何回分の設定を問い合わせるか (`count`)
を引数として、`wandb.agent` メソッドで sweep agent を作成します。

{{% alert %}}
同じ `sweep_id` を指定して、複数の sweep agent を異なる計算資源で同時に起動することも可能です。sweep controller が並列実行も含めて全体を調整してくれます。
{{% /alert %}}

下記のセルは sweep agent を有効化し、トレーニング関数（`train`）を5回実行します。

```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
ここでは sweep 設定で `random` 手法を指定したため、sweep controller によってランダムに生成されたハイパーパラメータ値が使われます。
{{% /alert %}}

ターミナル上で W&B Sweeps を使う方法の詳細は [W&B Sweep ウォークスルー]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}}) をご覧ください。

## Sweep 結果の可視化



### パラレル座標プロット
このプロットではハイパーパラメーター値とモデルメトリクスの関係を可視化します。最良のモデル性能につながったパラメータ組み合わせを見つけるのに役立ちます。

{{< img src="/images/tutorials/sweeps-2.png" alt="Sweep agent execution results" >}}


### ハイパーパラメータインポータンスプロット
hyperparameter importance plot では、どのハイパーパラメータがメトリクスに対して最も予測力を持つかを明らかにします。feature importance（ランダムフォレストモデルによる）や相関関係（暗に線形モデル）も示されます。

{{< img src="/images/tutorials/sweeps-3.png" alt="W&B sweep dashboard" >}}

こうした可視化を活用することで、重要なパラメータや値の範囲に注目し、余分なハイパーパラメータ最適化への計算資源や時間を節約できます。

## W&B Sweeps をもっと知る

[簡単なトレーニングスクリプトや sweep config サンプル](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion) をいくつか用意しましたので、ぜひお試しください。

上記リポジトリには [Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla)、[Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla) など、さらに高度な sweep 機能を試すためのサンプルも含まれています。