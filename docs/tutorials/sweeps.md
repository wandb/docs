
# ハイパーパラメーターのチューニング

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb)

高次元のハイパーパラメータースペースを探索して最もパフォーマンスの高いモデルを見つけることは、非常に難しい作業です。ハイパーパラメーター探索を利用することで、モデルのバトルロイヤルを効率的に実施し、最も正確なモデルを選び出すことができます。これは、ハイパーパラメーターの組み合わせ（例：学習率、バッチサイズ、隠れ層の数、オプティマイザーの種類）を自動的に探索して最適な値を見つけることで実現されます。

このチュートリアルでは、Weights & Biases を使って高度なハイパーパラメーター探索を3つの簡単なステップで実行する方法を見ていきます。

### [ビデオチュートリアル](http://wandb.me/sweeps-video)に沿って進めましょう！

![](https://i.imgur.com/WVKkMWw.png)

# 🚀 セットアップ

まずは、実験管理ライブラリをインストールし、無料の W&B アカウントを設定します:

1. `!pip install` コマンドでインストール
2. ライブラリを Python にインポート
3. `.login()` してプロジェクトにメトリクスをログインできるようにします

Weights & Biases を初めて使用する場合、`login` コマンドはアカウント登録用のリンクを提供します。W&B は個人および学術プロジェクトに無料で使用できます！

```python
!pip install wandb -Uq
```

```python
import wandb

wandb.login()
```

# ステップ1️⃣. Sweep の定義

根本的には、Sweep は一連のハイパーパラメーター値を試すための戦略をコードと組み合わせたものです。あなたはただ、[設定](https://docs.wandb.com/sweeps/configuration)形式で戦略を定義するだけです。

ノートブックで Sweep を設定する際は、その設定オブジェクトはネストされた辞書です。コマンドラインから Sweep を実行する場合、設定オブジェクトは [YAML ファイル](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)です。

一緒に Sweep 設定の定義を見ていきましょう。各コンポーネントについて説明する時間を取るため、ゆっくり進めます。通常の Sweep パイプラインでは、このステップはシングル割り当てで行います。

### 👈 `method` の選択

まず定義する必要があるのは、新しいパラメータ値を選ぶ `method` です。

以下の探索 `methods` を提供しています:
* **`grid`探索** – ハイパーパラメーター値の全ての組み合わせを試行します。非常に効果的ですが、計算コストが高くつくことがあります。
* **`random`探索** – 提供された `distribution` に基づいてランダムに新しい組み合わせを選びます。驚くほど効果的です！
* **`bayes`探索** – ハイパーパラメーターの関数としてメトリクススコアの確率モデルを作成し、メトリクスの改善の可能性が高いパラメーターを選びます。連続パラメーターの小さな数にはうまく機能しますが、大規模にはスケールしません。

この例では `random` を使います。

```python
sweep_config = {
    'method': 'random'
    }
```

`bayes`探索の場合、`metric` についても少し教えてください。モデル出力で見つけるためにその `name` が必要で、`goal` が `minimize`（例えば二乗誤差の場合）か `maximize`（例えば精度の場合）を知る必要があります。

```python
metric = {
    'name': 'loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric
```

`bayes` 探索を実行しない場合でも、後で心が変わる可能性があるため、`sweep_config` に含めるのは悪いアイディアではありません。こういったことをメモしておくのは再現性のための良い方法でもあります。6か月後、もしくは6年後に自分や他の誰かが Sweep を見直すときに、`val_G_batch` が高いべきなのか低いべきなのか分からない場合があります。

### 📃 ハイパー`parameters` の命名

新しいハイパーパラメーター値を試す `method` を選んだら、その `parameters` が何かを定義する必要があります。

ほとんどの場合、このステップは単純です:
`parameter` に名前を付け、そのパラメータの合法的な `values` のリストを指定するだけです。

例えば、ネットワーク用の `optimizer` を選ぶ際には、選択肢は限られています。ここでは、最も人気のある選択肢である `adam` と `sgd` を使用します。潜在的に無限の選択肢があるハイパーパラメーターでも、通常は試すべき選択肢がいくつかしかありません。ここでは隠れ層の `layer_size` と `dropout` を選びます。

```python
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

この Sweep で変動させたくないハイパーパラメーターがある場合でも、それを `sweep_config` に設定したい場合があります。

その場合、直接 `value` を設定します。

```python
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`grid` 検索の場合、これだけで十分です。

`random` 検索の場合、パラメーターの全ての `values` は与えられたランで選ばれる可能性が同じです。

それでは不十分な場合、名前付き`distribution` とそのパラメーター（例えば、正規分布の平均 `mu` と標準偏差 `sigma`）を指定することもできます。

ランダム変数の分布の設定方法については、[こちら](https://docs.wandb.com/sweeps/configuration#distributions)から詳細を確認できます。

```python
parameters_dict.update({
    'learning_rate': {
        # 0から0.1までのフラットな分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32から256までの整数
        # 対数が均等に分布
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

完成したら、`sweep_config` はネストされた辞書になります。この辞書は、試すべき `parameters` と、それを試す `method` を正確に指定します。

```python
import pprint

pprint.pprint(sweep_config)
```

しかし、これだけが設定オプションではありません！

例えば、[HyperBand](https://arxiv.org/pdf/1603.06560.pdf) スケジューリングアルゴリズムを使って run を早期終了させるオプションもあります。[こちら](https://docs.wandb.com/sweeps/configuration#stopping-criteria)で詳細を確認できます。

全ての設定オプションのリストは[こちら](https://docs.wandb.com/library/sweeps/configuration)にありますし、YAML 形式の大規模なサンプルコレクションは[こちら](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)にあります。

# ステップ2️⃣. Sweep の初期化

探索戦略を定義したら、それを実行するための設定を行います。

Sweep の管理を担うのが _Sweep Controller_ です。各 run が完了すると、新しい run を実行するための指示を発行します。この指示を受け取って実際に run を行うのが _エージェント_ です。

通常の Sweep では、Controller は _私たちの_ マシン上に存在し、run を完了するエージェントは _あなたの_ マシン上に存在します。例えば以下の図のように。この労働分担により、エージェントを実行するマシンを追加するだけで Sweep のスケールアップが非常に簡単になります！

<img src="https://i.imgur.com/zlbw3vQ.png" alt="sweeps-diagram" width="500"/>

適切な `sweep_config` と `プロジェクト` 名を指定して `wandb.sweep` を呼び出すことで Sweep Controller を準備できます。

この関数は `sweep_id` を返し、後にエージェントをこの Controller に割り当てるために使用します。

> _補足_: コマンドラインでは、この関数は次のように置き換えられます
```python
wandb sweep config.yaml
```
[コマンドラインでの Sweeps の使用方法についてさらに学ぶ ➡](https://docs.wandb.ai/guides/sweeps/walkthrough)

```python
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

# ステップ3️⃣. Sweep エージェントの実行

### 💻 トレーニング手順の定義

実際に Sweep を実行する前に、これらの値を使用するトレーニング手順を定義する必要があります。

以下の関数では、PyTorch で単純な全結合ニューラルネットワークを定義し、次の `wandb` ツールを追加してモデルのメトリクスをログし、パフォーマンスと出力を可視化し、実験を追跡します:
* [**`wandb.init()`**](https://docs.wandb.com/library/init) – 新しい W&B Run を初期化します。各 Run はトレーニング関数の1回の実行です。
* [**`wandb.config`**](https://docs.wandb.com/library/config) – 全てのハイパーパラメーターを設定オブジェクトに保存し、ログに記録できるようにします。`wandb.config` の使用方法については[こちら](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb)をご覧ください。
* [**`wandb.log()`**](https://docs.wandb.com/library/log) – モデルの振る舞いを W&B にログします。ここではパフォーマンスのみをログします; `wandb.log` でログできる他のリッチメディアについては[この Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb)をご覧ください。

PyTorch との W&B 連携の詳細については[この Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)をご覧ください。

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 新しい wandb run を初期化
    with wandb.init(config=config):
        # wandb.agent が呼び出す場合、この config は Sweep Controller によって設定されます
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

このセルでは、トレーニング手順の4つの部分を定義しています:
`build_dataset`, `build_network`, `build_optimizer`, そして `train_epoch`。

これらは全て基本的な PyTorch パイプラインの標準部分であり、W&B の使用によって影響を受けることはありませんので、詳しくはコメントしません。

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # MNIST トレーニングデータセットをダウンロード
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # 全結合の単一隠れ層
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

        # ➡ forward pass
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ バックプロパゲーション + 重みの更新
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

これで、Sweep を開始する準備が整いました！ 🧹🧹🧹

Sweep Controllers は `wandb.sweep` を実行することで作られ、
誰かが試すべき `config` を求めてくるのを待っています。

その誰かが `エージェント` であり、`wandb.agent` で作成されます。
開始するには、エージェントは次のことを知る必要があります:
1. どの Sweep の一部か (`sweep_id`)
2. 実行する関数 (ここでは `train`)
3. (オプションで) コンフィギュレーションを Controller にリクエストする回数 (`count`)

FYI, 同じ `sweep_id` で複数の `エージェント` を異なるコンピューティングリソースで起動し、Controller が `sweep_config` の戦略に従ってそれらが協力して作業するようにします。これにより、あなたが持っている限りのノードに Sweeps を簡単にスケールアウトすることができます！

> _補足_: コマンドラインでは、この関数は次のように置き換えられます
```bash
wandb agent sweep_id
```
[コマンドラインでの Sweeps の使用方法についてさらに学ぶ ➡](https://docs.wandb.com/sweeps/walkthrough)

以下のセルでは、Sweep Controller が返したランダムに生成されたハイパーパラメーター値を使用して `train` を5回実行する `エージェント` を起動します。実行は5分以内に完了します。

```python
wandb.agent(sweep_id, train, count=5)
```

# 👀 Sweep 結果の可視化

## 🔀 Parallel Coordinates プロット
このプロットはハイパーパラメーター値をモデルのメトリクスにマップします。最も優れたモデルパフォーマンスに繋がったハイパーパラメーターの組み合わせを見つけるのに役立ちます。

![](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5e190366778ad831455f9af2_s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695138341_image.png)

## 📊 ハイパーパラメーターの重要性プロット
ハイパーパラメータ

# 🧤 実際に sweeps を試してみよう

簡単なトレーニングスクリプトと [いくつかの異なる sweep 構成](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion) を用意しましたので、ぜひ試してみてください。

そのリポジトリには、[Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla) や [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla) など、より高度な sweep 機能を試すための例も含まれています。

# 次は？
次のチュートリアルでは、W&B Artifacts を使用してモデルの重みとデータセットのバージョンを管理する方法を学びます:

## 👉 [モデルの重みとデータセットのバージョンを管理する](artifacts)