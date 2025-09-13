---
title: Sweeps で ハイパラメーターをチューニングする
menu:
  tutorials:
    identifier: ja-tutorials-sweeps
    parent: null
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" >}}
望むメトリクス（たとえばモデル精度）を満たす 機械学習 モデルを見つける作業は、通常は何度も繰り返す冗長なタスクになりがちです。さらに悪いことに、特定のトレーニング run に対してどのハイパーパラメーターの組み合わせを使うべきかが不明瞭な場合もあります。

W&B Sweeps を使うと、学習率、バッチサイズ、隠れ層の数、オプティマイザーの種類などのハイパーパラメーターの 値 の組み合わせを自動で探索し、希望するメトリクスに基づいて モデル を最適化する 値 を見つける、整理された効率的な方法を作れます。

このチュートリアルでは、W&B の PyTorch インテグレーションを使ってハイパーパラメーター探索を作成します。あわせて [動画チュートリアル](https://wandb.me/sweeps-video) もご覧ください。

{{< img src="/images/tutorials/sweeps-1.png" alt="ハイパーパラメーター sweep の結果" >}}

## Sweeps: 概要

W&B でハイパーパラメーター sweep を実行するのはとても簡単です。必要なのは 3 ステップだけです。

1. **sweep を定義する:** 検索対象のパラメータ、検索戦略、最適化するメトリクスなどを指定する辞書、または [YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) を作成します。
2. **sweep を初期化する:** 1 行の コード で sweep を初期化し、sweep configuration の辞書を渡します:
`sweep_id = wandb.sweep(sweep_config)`
3. **sweep agent を実行する:** これも 1 行の コード で行えます。`wandb.agent()` を呼び出して `sweep_id` と、モデルのアーキテクチャーを定義してトレーニングする関数を渡します:
`wandb.agent(sweep_id, function=train)`

## 始める前に

W&B をインストールし、W&B Python SDK をノートブックにインポートします。

1. `!pip install` でインストール:

```
!pip install wandb -Uq
```

2. W&B をインポート:

```
import wandb
```

3. W&B にログインし、プロンプトが表示されたら APIキー を入力します:

```
wandb.login()
```

## ステップ 1️: sweep を定義する

W&B Sweep は、多数のハイパーパラメーターの 値 を試す戦略と、それらを評価する コード を組み合わせたものです。
sweep を開始する前に、_sweep configuration_ で sweep の戦略を定義する必要があります。

{{% alert %}}
Jupyter Notebook で sweep を開始する場合、作成する sweep configuration はネストされた辞書である必要があります。

コマンドライン で sweep を実行する場合は、[YAML ファイル]({{< relref path="/guides/models/sweeps/define-sweep-configuration" lang="ja" >}}) で sweep の設定を指定する必要があります。
{{% /alert %}}

### 検索メソッドを選ぶ

まず、設定の辞書内でハイパーパラメーター検索のメソッドを指定します。[選べるハイパーパラメーター検索戦略は 3 つ（グリッド、ランダム、ベイズ探索）です]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/#method" lang="ja" >}})。

このチュートリアルではランダム検索を使用します。ノートブック内で辞書を作成し、`method` キーに `random` を指定してください。

```
sweep_config = {
    'method': 'random'
    }
```

最適化したいメトリクスを指定します。ランダム検索を使う sweep では、メトリクスと目標の指定は必須ではありません。ただし、後から参照できるように sweep の目標を記録しておくのは良いプラクティスです。

```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### 探索するハイパーパラメーターを指定する

検索メソッドを指定したら、次に探索したいハイパーパラメーターを指定します。

そのためには、`parameter` キーに 1 つ以上のハイパーパラメーター名を、`value` キーに 1 つ以上の 値 を指定します。

特定のハイパーパラメーターで探索する 値 は、そのハイパーパラメーターの型に依存します。

たとえば 機械学習 のオプティマイザーを選ぶ場合、Adam オプティマイザーや確率的 勾配 降下法（SGD）のような有限個のオプティマイザー名を 1 つ以上指定します。

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

ハイパーパラメーターを追跡したいが、その 値 を変動させたくない場合もあります。その場合は、sweep configuration にそのハイパーパラメーターを追加し、使用したい固定 値 を指定します。以下のコードセルでは、`epochs` を 1 に設定しています。

```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`random` 検索では、
あるパラメータのすべての `values` は、その run で選ばれる確率が等しくなります。

代わりに、
名前付きの `distribution` と、そのパラメータ（`normal` 分布の平均 `mu` や標準偏差 `sigma` など）を指定することもできます。

```
parameters_dict.update({
    'learning_rate': {
        # 0 から 0.1 の一様分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32 から 256 の整数
        # 対数が一様に分布
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

これで `sweep_config` は、
試したい `parameters` と、それらを試すために使う `method` を正確に指定したネストされた辞書になりました。

sweep configuration がどのようになるか見てみましょう。

```
import pprint
pprint.pprint(sweep_config)
```

設定オプションの一覧は、[Sweep configuration のオプション]({{< relref path="/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" lang="ja" >}}) を参照してください。

{{% alert %}}
無限に 値 があり得るハイパーパラメーターには、
いくつかの `values` に絞って試すのが妥当なことが多いです。たとえば、前述の sweep configuration では、`layer_size` と `dropout` のパラメータキーに有限個の 値 のリストを指定しています。
{{% /alert %}}

## ステップ 2️: Sweep を初期化する

検索戦略を定義したら、それを実行するための準備をします。

W&B は Sweep Controller を使って、クラウドまたはローカルの 1 台以上のマシンにわたって sweep を管理します。このチュートリアルでは、W&B が管理する Sweep Controller を使用します。

Sweep Controller が sweep を管理する一方で、実際に sweep を実行するコンポーネントは _sweep agent_ と呼ばれます。

{{% alert %}}
デフォルトでは、Sweep Controller のコンポーネントは W&B の サーバー 上で起動され、sweep agent（sweep を作成するコンポーネント）はローカルマシンで起動されます。
{{% /alert %}}

ノートブック内では、`wandb.sweep` メソッドで Sweep Controller を起動できます。先ほど定義した sweep configuration の辞書を `sweep_config` フィールドに渡します。

```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

`wandb.sweep` 関数は `sweep_id` を返します。これは後の手順で sweep を起動する際に使います。

{{% alert %}}
コマンドライン では、この関数は次の コマンド に置き換わります
```python
wandb sweep config.yaml
```
{{% /alert %}}

ターミナル での W&B Sweeps の作成方法については、[W&B Sweep のウォークスルー]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}}) を参照してください。

## ステップ 3: 機械学習の コード を定義する

sweep を実行する前に、
試したいハイパーパラメーターの 値 を使うトレーニング手順を定義します。W&B Sweeps をトレーニング コード に組み込むカギは、各トレーニング 実験 において、トレーニングのロジックが sweep configuration で定義したハイパーパラメーターの 値 に アクセス できることを保証することです。

以下のコード例では、`build_dataset`、`build_network`、`build_optimizer`、`train_epoch` の各ヘルパー関数が、sweep のハイパーパラメーター設定の辞書に アクセス します。

以下の 機械学習 のトレーニング コード をノートブックで実行してください。これらの関数は、PyTorch で基本的な全結合 ニューラルネットワーク を定義します。

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 新しい W&B Run を初期化
    with wandb.init(config=config) as run:
        # 下のとおり wandb.agent によって呼ばれた場合、
        # この config は Sweep Controller によって設定される
        config = run.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            run.log({"loss": avg_loss, "epoch": epoch})           
```

`train` 関数の中で、次の W&B Python SDK メソッドが使われていることに気づくはずです。
* [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init/" lang="ja" >}}): 新しい W&B Run を初期化します。各 Run はトレーニング関数 1 回の実行に対応します。
* [`run.config`]({{< relref path="/guides/models/track/config" lang="ja" >}}): 実験したいハイパーパラメーターを含む sweep configuration を受け取ります。
* [`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}): 各 エポック のトレーニング損失を ログ します。

以下のセルでは 4 つの関数
`build_dataset`、`build_network`、`build_optimizer`、`train_epoch`
を定義しています。これらの関数は基本的な PyTorch パイプライン の標準的な構成要素であり、W&B の使用によって実装が変わることはありません。

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # MNIST のトレーニングデータセットをダウンロード
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # 全結合、隠れ層が 1 つ
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

            # ➡ forward pass
            loss = F.nll_loss(network(data), target)
            cumu_loss += loss.item()

            # ⬅ backward pass + 重み更新
            loss.backward()
            optimizer.step()

            run.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

PyTorch と W&B の連携方法の詳細は、[この Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb) を参照してください。

## ステップ 4: sweep agent を起動する
sweep configuration を定義し、ハイパーパラメーターを対話的に扱えるトレーニングスクリプトが用意できたら、sweep agent を起動する準備は完了です。sweep agent は、sweep configuration で定義したハイパーパラメーターのセットを用いて 実験 を実行する役割を担います。

`wandb.agent` メソッドで sweep agent を作成します。次を指定してください。
1. エージェントが属する sweep（`sweep_id`）
2. sweep が実行すべき関数。この例では `train` 関数を使います。
3. （オプション）Sweep Controller に要求する設定の個数（`count`）

{{% alert %}}
同じ `sweep_id` で複数の sweep agent を
異なる計算リソース上で起動できます。Sweep Controller が、あなたが定義した sweep configuration に従って協調して動作するように調整します。
{{% /alert %}}

以下のセルは、トレーニング関数（`train`）を 5 回実行する sweep agent を起動します。

```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
sweep configuration で `random` 検索メソッドを指定したため、Sweep Controller はランダムに生成されたハイパーパラメーターの 値 を提供します。
{{% /alert %}}

ターミナル での W&B Sweeps の作成方法については、[W&B Sweep のウォークスルー]({{< relref path="/guides/models/sweeps/walkthrough" lang="ja" >}}) を参照してください。

## Sweep の 結果を可視化する

### 平行座標プロット
このプロットはハイパーパラメーターの 値 をモデルの メトリクス に対応付けます。最良のモデル性能につながったハイパーパラメーターの組み合わせに絞り込むのに役立ちます。

{{< img src="/images/tutorials/sweeps-2.png" alt="sweep agent の実行結果" >}}

### ハイパーパラメーター インポータンスプロット
ハイパーパラメーターのインポータンスプロットは、メトリクスの良い予測因子となったハイパーパラメーターを可視化します。
ここでは、特徴量重要度（ランダムフォレスト モデル に基づく）と相関（暗に線形モデル）を報告します。

{{< img src="/images/tutorials/sweeps-3.png" alt="W&B sweep ダッシュボード" >}}

これらの 可視化 は、重要なパラメータ（およびその 値 の範囲）に絞り込むことで、コストのかかるハイパーパラメーター最適化の時間とリソースを節約するのに役立ちます。

## W&B Sweeps をもっと学ぶ

遊びながら試せるシンプルなトレーニングスクリプトと [いくつかの種類の sweep config](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion) を用意しました。ぜひ試してみてください。

そのリポジトリには、[Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla) や [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla) など、より高度な sweep 機能を試すための例も含まれています。