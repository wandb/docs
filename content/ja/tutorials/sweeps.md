---
title: スイープでハイパーパラメーターをチューニングする
menu:
  tutorials:
    identifier: sweeps
    parent: null
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" >}}

希望するメトリクス（例: モデルの精度）を満たす機械学習モデルを見つけるには、何度も繰り返し作業が必要になることが多いです。さらに困ったことに、トレーニング run ごとにどのハイパーパラメーターの組み合わせを試すべきか分からないこともあります。

W&B Sweeps を使えば、学習率、バッチサイズ、隠れ層の数、オプティマイザーの種類など、さまざまなハイパーパラメーターの組み合わせを自動探索し、希望するメトリクスにもとづいてモデルを最適化するパラメーターを効率的かつ整理して見つけることができます。

このチュートリアルでは、W&B PyTorch インテグレーションを使ったハイパーパラメーター探索を行います。 [ビデオチュートリアル](https://wandb.me/sweeps-video) もあわせてご覧ください。

{{< img src="/images/tutorials/sweeps-1.png" alt="Hyperparameter sweep results" >}}

## Sweeps: 概要

W&B を使ったハイパーパラメーター探索（sweep）は非常に簡単です。たった 3 ステップで実行できます。

1. **sweep を定義する:** 探索するパラメーター、探索手法、最適化したいメトリクス等を指定した辞書または [YAMLファイル]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}}) を作成します。

2. **sweep を初期化する:** 1行のコードで sweep を初期化し、設定用の辞書を渡します。  
`sweep_id = wandb.sweep(sweep_config)`

3. **sweep agent を起動する:** こちらも 1 行のコードで、`wandb.agent()` に `sweep_id` と、モデルのアーキテクチャ定義・トレーニングを行う関数を渡して実行します。  
`wandb.agent(sweep_id, function=train)`

## 始める前に

W&B をインストールし、Python SDK をノートブックにインポートします。

1. `!pip install` でインストール:

```
!pip install wandb -Uq
```

2. W&B をインポート:

```
import wandb
```

3. W&B にログインし、プロンプトが表示されたら APIキー を入力します。

```
wandb.login()
```

## Step 1️: sweep の定義

W&B Sweep は、さまざまなハイパーパラメータの値を試すための戦略と、それを評価するコードを組み合わせたものです。
sweep を始める前に、_sweep configuration_ で sweep の戦略を定義します。

{{% alert %}}
Jupyter Notebook で sweep を始める場合、sweep configuration はネストされた辞書で記述する必要があります。

コマンドラインで sweep を実行する場合、[YAML ファイル]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}}) で sweep config を指定してください。
{{% /alert %}}

### 検索手法を選ぶ

まずは、設定用の辞書内でハイパーパラメータ探索の手法を指定します。[ハイパーパラメーター探索戦略は grid、random、ベイズ探索 の3つから選択できます]({{< relref "/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/#method" >}})。

このチュートリアルではランダム検索（random search）を使います。ノートブック内で、`method` キーに `random` を指定した辞書を作成してください。

```
sweep_config = {
    'method': 'random'
    }
```

最適化したいメトリクスも指定しておきましょう。ランダム検索の場合は metric や goal は必須ではありませんが、後で参照できるように記録しておくのがおすすめです。

```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### 探索したいハイパーパラメータを指定

sweep configuration に探索手法を設定したら、今度は探索したいハイパーパラメータも指定します。

これには、`parameter` キーにハイパーパラメータの名前を、`value`（または`values`）キーにその候補値を指定します。

どの値を探索するかは、対象となるハイパーパラメータのタイプによって異なります。

例えば、オプティマイザーを選択する場合は Adam、SGD などの有限の名称を指定します。

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

あるハイパーパラメータを追跡のみしたい（値は固定したい）場合は、そのハイパーパラメータを sweep configuration に追加し、特定の値を `value` で指定します。下記例では `epochs` が 1 に固定されています。

```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`random` 検索では、
run ごとに各パラメータの `values` から均等な確率で選ばれます。

また、`distribution`（分布）を名前で指定して、
平均値 `mu` や標準偏差 `sigma` のある `normal`（正規分布）なども使えます。

```
parameters_dict.update({
    'learning_rate': {
        # 0 から 0.1 までの一様分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32 から 256 までの整数値（log が均等に分布）
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

ここまで終わると、`sweep_config` は
どの `parameters` を試すか、どの `method` で試行するか明確に記したネスト構造の辞書になります。

実際にどうなるか出力してみましょう。

```
import pprint
pprint.pprint(sweep_config)
```

設定オプションの全リストは [Sweep configuration options]({{< relref "/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" >}}) をご覧ください。

{{% alert %}}
無数に選択肢があるハイパーパラメータの場合は、
いくつか代表的な `values` を選んで試すのが一般的です。上記 sweep configuration でも `layer_size` や `dropout` には finite な値のリストを設定しています。
{{% /alert %}}

## Step 2️: Sweep を初期化

探索戦略の定義ができたら、それを実行するための準備をします。

W&B では Sweep Controller がクラウドやローカルの複数のマシン上で sweep を管理します。このチュートリアルでは W&B が管理する sweep controller を利用します。

sweep の実行本体は _sweep agent_ と呼ばれるコンポーネントです。

{{% alert %}}
デフォルトでは、Sweep Controller は W&B サーバー上、Sweep Agent（実際のsweepを起動するコンポーネント）はローカルマシン上で動作します。
{{% /alert %}}

ノートブック内で sweep controller を有効化するには、`wandb.sweep` メソッドを利用します。先ほど作成した sweep configuration 辞書を `sweep_config` フィールドに渡します。

```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

`wandb.sweep` 関数は後ほど sweep を有効化する際に使う `sweep_id` を返します。

{{% alert %}}
コマンドラインの場合、下記コマンドに置き換わります。
```python
wandb sweep config.yaml
```
{{% /alert %}}

コマンドラインでの W&B Sweeps 作成方法については [W&B Sweep walkthrough]({{< relref "/guides/models/sweeps/walkthrough" >}}) を参照してください。

## Step 3: 機械学習用のコードを定義する

sweep を実行する前に、
試したいハイパーパラメータ値で使うトレーニング手続きを定義する必要があります。W&B Sweeps をトレーニングコードに組み込むポイントは、各トレーニング実験ごとに sweep configuration で定義したハイパーパラメータ値にアクセスできるようにすることです。

下記のコード例では、`build_dataset`、`build_network`、`build_optimizer`、`train_epoch` の各関数が sweep ハイパーパラメータ設定辞書へアクセスしています。

この後のコードをノートブックで実行してください。各関数は PyTorch でシンプルな全結合ニューラルネットワークを定義しています。

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
        # 下記のように wandb.agent に呼び出された場合は
        # この config は Sweep Controller により自動設定されます
        config = run.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            run.log({"loss": avg_loss, "epoch": epoch})           
```

`train` 関数内で使われている W&B Python SDK の主なメソッド:
* [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init/" >}}): 新しい W&B Run を初期化します。run とはトレーニング関数 1 回の実行です。
* [`run.config`]({{< relref "/guides/models/track/config" >}}): sweep configuration とそれぞれのハイパーパラメータ値を渡します。
* [`run.log()`]({{< relref "/ref/python/sdk/classes/run/#method-runlog" >}}): 各エポックの損失（loss）をログに記録します。

この後には、`build_dataset`、`build_network`、`build_optimizer`、`train_epoch` の4関数が登場します。
これらは基本的な PyTorch パイプラインの標準的な実装例であり、W&B の使用による影響はありません。

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # MNIST 学習データセットをダウンロード
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # 全結合、1 隠れ層
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

            # ➡ フォワードパス
            loss = F.nll_loss(network(data), target)
            cumu_loss += loss.item()

            # ⬅ バックワードパス + 重みの更新
            loss.backward()
            optimizer.step()

            run.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

PyTorch と W&B のインテグレーション詳細は [こちらの Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb) もご覧ください。

## Step 4: sweep agent の起動
ここまでで sweep configuration の定義と、ハイパーパラメータをインタラクティブに使えるトレーニングスクリプトが用意できました。あとは sweep agent を起動するだけです。sweep agent は sweep configuration で定義したハイパーパラメータ値ごとに実験を実行します。

`sweep_id` をもとに `wandb.agent` メソッドで sweep agent を作成しましょう。必要なのは以下 3 点です。

1. sweep agent が紐づく sweep の `sweep_id`
2. sweep が実行する関数（例では `train` を使用）
3. （任意）sweep controller からいくつの設定値を受け取って実行するか（`count`）

{{% alert %}}
同じ `sweep_id` で複数の sweep agent を異なる計算リソース上で起動できます。sweep controller が sweep configuration 通り協調動作するよう管理してくれます。
{{% /alert %}}

下記セルは sweep agent を 5 回 `train` 関数で実行します。

```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
sweep configuration で `random` 検索が指定されているため、sweep controller からはランダム生成されたハイパーパラメータ値が供給されます。
{{% /alert %}}

コマンドラインでの W&B Sweeps 作成方法については [W&B Sweep walkthrough]({{< relref "/guides/models/sweeps/walkthrough" >}}) も参照してください。

## Sweep 結果の可視化

### Parallel Coordinates Plot（並列座標プロット）
ハイパーパラメータの値とモデルメトリクスの関係を可視化します。最もよいモデル性能へ導いたハイパーパラメータの組み合わせを見つけるのに役立ちます。

{{< img src="/images/tutorials/sweeps-2.png" alt="Sweep agent execution results" >}}

### ハイパーパラメータのインポータンスプロット
ハイパーパラメータのインポータンスプロット（parameter importance plot）は、どのハイパーパラメータがメトリクスの予測にどれだけ効いているかを示します。
ここではランダムフォレストモデルによる特徴量重要度と、相関（線形モデル暗黙的なもの）をレポートしています。

{{< img src="/images/tutorials/sweeps-3.png" alt="W&B sweep dashboard" >}}

これらの可視化を活用すれば、重要なパラメータや値の範囲に注目することで、高価なハイパーパラメータ最適化の時間やリソースを節約できます。

## W&B Sweeps をさらに知りたい方へ

シンプルなトレーニングスクリプトと [いくつかの sweep config](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion) を用意していますので、ぜひお試しください。

このリポジトリには [Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla) や [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla) など、より高度な sweep 機能の例も掲載しています。