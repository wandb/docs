# ハイパーパラメーターをチューニングする

[**Colabノートブックで試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb)

高次元のハイパーパラメータースペースを探索して最も性能の良いモデルを見つけるのは非常に手間がかかります。ハイパーパラメーター探索は、モデルをバトルロワイヤル形式で対決させ、最も正確なモデルを選び出すための組織化された効率的な方法を提供します。この方法では、ハイパーパラメーターの値（例えば、学習率、バッチサイズ、隠れ層の数、オプティマイザーの種類）の組み合わせを自動的に探索し、最適な値を見つけることができます。

このチュートリアルでは、Weights & Biasesを使って3つの簡単なステップで高度なハイパーパラメーター探索を実行する方法を紹介します。

### [ビデオチュートリアル](http://wandb.me/sweeps-video)と一緒に学ぼう！

![](https://i.imgur.com/WVKkMWw.png)

# 🚀 セットアップ

まず、実験管理ライブラリをインストールし、無料のW&Bアカウントを設定します。

1. `!pip install` でインストール
2. Pythonにライブラリを `import`
3. `.login()` してメトリクスをプロジェクトにログ

もしWeights & Biasesを初めて使用する場合、 
`login` の呼び出しでアカウント作成のリンクが提供されます。
W&Bは個人および学術プロジェクトに無料で使用できます！

```python
!pip install wandb -Uq
```

```python
import wandb

wandb.login()
```

# Step 1️⃣. Sweepを定義する

本質的に、Sweepは多数のハイパーパラメーターの値を試すための戦略と、それらを評価するコードを組み合わせたものです。
[設定](https://docs.wandb.com/sweeps/configuration) という形で
戦略を定義する必要があります。

ノートブックでSweepを設定する場合、
設定オブジェクトはネストした辞書となります。
コマンドラインでSweepを実行する場合、
設定オブジェクトは
[YAMLファイル](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration) になります。

一緒にSweep設定の定義を進めていきましょう。
各コンポーネントを説明するために、ゆっくり進めます。
一般的なSweepパイプラインでは、
このステップは一度の割り当てで行います。

### 👈 `method` を選択

新しいパラメーター値を選択する `method` を定義する最初のステップです。

以下の検索`methods`を提供しています：
* **`grid` 検索** – ハイパーパラメーター値のすべての組み合わせを繰り返し試行します。
非常に効果的ですが、計算量が多くなることがあります。
* **`random` 検索** – 提供された `distribution` に従って新しい組み合わせをランダムに選択します。驚くほど効果的です！
* **`bayes`ian 検索** – ハイパーパラメーターに対するメトリクススコアの確率モデルを作成し、メトリクスを改善する確率の高いパラメーターを選択します。少数の連続パラメーターに対しては効果的ですが、スケールしにくいです。

ここでは `random` を使用します。

```python
sweep_config = {
    'method': 'random'
    }
```

`bayes`ian Sweepの場合も、 `metric` について少し教えてください。
モデル出力から見つけられるようにその `name` を教えてください
また、 `goal` が `minimize` すること（例えば平方誤差の場合）なのか、
それとも `maximize` すること（例えば精度の場合）なのかも教えてください。

```python
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

`bayes`ian Sweepを実行しない場合でも、この情報を `sweep_config` に含めておくと良いでしょう。後で変更する場合でも役立ちます。また、再現性のためにこのようなことを記録しておくことは良い習慣です。
6か月または6年後に自分や他の人がSweepに戻ってきた際に、 `val_G_batch` が高い方が良いのか低い方が良いのかを確認できるようにしておくためです。

### 📃 ハイパー`parameters` の名前を付ける

新しいハイパーパラメーターの値を試す `method` を選択したら、その `parameters` を定義する必要があります。

ほとんどの場合、このステップは簡単です：
パラメーターに名前を付け、そのパラメーターの合法な `values` のリストを指定するだけです。

例えば、ネットワークの `optimizer` を選択する際には、選択肢は限られています。
ここでは、最も人気のある2つの選択肢、`adam` と `sgd` を使用します。
無限の選択肢がある可能性のあるハイパーパラメーターでも、実際にはいくつかの選択された `values` を試すのが理にかなっています。ここでは、隠れ層の `layer_size` と `dropout` で選択しています。

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

この探索では変動させたくないハイパーパラメーターがあることが多いですが、それでも `sweep_config` に設定しておきたい場合は、その `value` を直接設定します。

```python
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`grid` 検索の場合、それだけで十分です。

`random` 検索の場合、パラメーターのすべての `value` が同じ確率で選択されます。

もしこの動作が望ましくない場合は、名前付きの `distribution` とそのパラメーター、例えば `normal` 分布の平均 `mu` や標準偏差 `sigma` を指定することもできます。

ランダム変数の分布の設定方法については、 [ここを参照](https://docs.wandb.com/sweeps/configuration#distributions)。

```python
parameters_dict.update({
    'learning_rate': {
        # 0から0.1までの一様分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32から256までの整数
        # 対数が均等に分布している
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

設定が完了したとき、 `sweep_config` はネストされた辞書であり、どの `parameters` に興味があるか、どの `method` を使用するかを正確に指定します。

```python
import pprint

pprint.pprint(sweep_config)
```

しかし、設定オプションはこれだけではありません！

例えば、[HyperBand](https://arxiv.org/pdf/1603.06560.pdf) スケジューリングアルゴリズムを使用して `early_terminate` オプションも提供しています。詳細は [こちら](https://docs.wandb.com/sweeps/configuration#stopping-criteria) をご覧ください。

すべての設定オプションのリストは [こちら](https://docs.wandb.com/library/sweeps/configuration)に、YAML形式の例の大集合は [こちら](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)にあります。

# Step 2️⃣. Sweepを初期化する

検索戦略を定義したら、それを実装する何かを設定する時が来ました。

Sweepの中心となるのは _Sweep Controller_ です。
各runが完了するたびに、新しいrunを実行するための指示を発行します。
これらの指示はrunを実際に実行する _agents_ によって受け取られます。

一般的なSweepでは、Controllerは _私たちの_ マシン上に存在し、runを完了するエージェントは _あなたの_ マシン（例えば次の図）上に存在します。
この労働分担により、エージェントを実行するためのマシンを追加するだけでSweepを簡単にスケールアップできます！

<img src="https://i.imgur.com/zlbw3vQ.png" alt="sweeps-diagram" width="500"/>

適切な `sweep_config` と `project` 名を使用して `wandb.sweep` を呼び出すことで、Sweep Controllerを起動できます。

この関数は `sweep_id` を返し、後でこのControllerにエージェントを割り当てる時に使用します。

> _補足_: コマンドラインでは、この関数は次のように置き換えられます
```python
wandb sweep config.yaml
```
[コマンドラインでのSweepの使用についてもっと学ぶ ➡](https://docs.wandb.ai/guides/sweeps/walkthrough)

```python
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

# Step 3️⃣. Sweepエージェントを実行する

### 💻 トレーニング手順を定義する

実際にsweepを実行する前に、それらの値を使用するトレーニング手順を定義する必要があります。

以下の関数では、PyTorchでの単純な完全結合ニューラルネットワークを定義し、次の `wandb` ツールを追加してモデルのメトリクスをログし、パフォーマンスと出力を可視化し、実験を追跡します：
* [**`wandb.init()`**](https://docs.wandb.com/library/init) – 新しいW&B Runを初期化します。各Runはトレーニング関数の単一の実行です。
* [**`wandb.config`**](https://docs.wandb.com/library/config) – 全てのハイパーパラメーターを設定オブジェクトに保存し、ログされるようにします。 `wandb.config` の使用方法については[こちら](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb)をご覧ください。
* [**`wandb.log()`**](https://docs.wandb.com/library/log) – モデルの振る舞いをW&Bにログします。ここではパフォーマンスのみをログしますが、その他のリッチメディアのログについては[このColab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb)をご覧ください。

PyTorchとW&Bの統合に関する詳細は[こちらのColab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)をご覧ください。

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # 新しいwandb runを初期化
    with wandb.init(config=config):
        # wandb.agentによって呼び出された場合、以下のように、
        # このconfigはSweep Controllerによって設定されます
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

このセルはトレーニング手順の4つの部分を定義します：
`build_dataset`, `build_network`, `build_optimizer`, そして `train_epoch`。

これらはすべて基本的なPyTorchパイプラインの一部であり、
W&Bの使用によってその実装が影響を受けることはありません。
したがって、それについてコメントはしません。

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
    network = nn.Sequential(  # 完全結合、単一の隠れ層
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

        # ⬅ backward pass ＋ 重みの更新
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

さあ、sweepを始めましょう！ 🧹🧹🧹

Sweep Controllersは、`wandb.sweep`を実行することで作成され、
誰かが試すための `config`を要求するのを待っています。

その誰かとは `agent` であり、 `wandb.agent` を使用して作成されます。
始めるために、エージェントは次のことを知る必要があります:
1. どのSweepの一部であるか（`sweep_id`）
2. 実行する関数はどれか（ここでは `train`）
3. （オプションで）Controllerにどれだけの`config`を要求するか（`count`）

FYI、異なる計算リソースで同じ `sweep_id` を使用して複数の `agent` を開始することができ、
Controllerはそれらが `sweep_config` に示された戦略に従って協力するようにします。
これにより、利用可能なノードを最大限に活用してSweepを簡単にスケールアップできます！

> _補足_: コマンドラインでは、この関数は次のように置き換えられます
```bash
wandb agent sweep_id
```
[コマンドラインでのSweepの使用についてもっと学ぶ ➡](https://docs.wandb.com/sweeps/walkthrough)

以下のセルは、Sweep Controller から返されたランダムに生成されたハイパーパラメーター値を使用して、`train` を5回実行する `agent` を起動します。実行時間は5分未満です。

```python
wandb.agent(sweep_id, train, count=5)
```

# 👀 Sweep結果を可視化する

## 🔀 パラレルコーディネートプロット
このプロットはハイパーパラメーターの値をモデルメトリクスにマップします。最も良いモデル性能を引き出したハイパーパラメーターの組み合わせを見つけるのに役立ちます。

![](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5e190366778ad831455f9af2_s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695138341_image.png)

## 📊 ハイパーパラメーターインポータンスプロット
ハイパーパラメーターインポータンスプロットは、メトリクスの最も良い予測因子であったハイパーパラメーターを明らかにします。
特徴重要度（ランダムフォレストモデルから）と相関性（暗黙の線形モデル）を報告します。

![](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5e190367778ad820b35f9af5_s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695757573_image.png)

これらの可視化は、最も重要で、さらに探る価値のあるパラメーター（および値の範囲）に絞り込むことで、コストの高いハイパーパラメーター最適化を実行する時間とリソースを節約するのに役立ちます。

# 🧤 sweeps で実践

簡単なトレーニングスクリプトと [いくつかのsweep設定の例](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion) を用意しています。ぜひこれらを試してみてください。

このリポジトリには、[Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla)、 [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla) など、より高度なsweep機能を試すための例も含まれています。

