
# ハイパーパラメータをチューニングする

[**こちらのColabノートブックで試してみてください →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb)

高次元のハイパーパラメータ空間を探索して最もパフォーマンスの高いモデルを見つけるのは、非常に複雑で困難です。ハイパーパラメータスイープは、モデルを効率的かつ組織的にバトルロイヤルする方法を提供し、最も正確なモデルを選び出します。スイープは、自動的にハイパーパラメータの値（例：学習率、バッチサイズ、隠れ層の数、optimizerの種類）の組み合わせを検索して、最適な値を見つけることによりこれを実現します。

このチュートリアルでは、Weights & Biasesを使用して3つの簡単なステップで高度なハイパーパラメータスイープを実行する方法を説明します。

### [ビデオチュートリアル](http://wandb.me/sweeps-video)もあわせてご覧ください！

![](https://i.imgur.com/WVKkMWw.png)

# 🚀 セットアップ

まず、実験追跡ライブラリをインストールし、無料のW&Bアカウントを設定します：

1. `!pip install`でインストール
2. ライブラリをPythonに`インポート`
3. プロジェクトにメトリクスをログできるように`.login()`

もしWeights & Biasesを初めて使用する場合、
`ログイン`の呼び出しはアカウント登録リンクを提供します。
W&Bは個人および学術プロジェクトには無料で使用できます！

```python
!pip install wandb -Uq
```

```python
import wandb

wandb.login()
```

# ステップ 1️⃣. スイープを定義する

基本的に、スイープは一連のハイパーパラメータ値を試すための戦略と、それらを評価するコードを組み合わせたものです。
_戦略を定義_するだけで、[configuration](https://docs.wandb.com/sweeps/configuration)形式で設定できます。

ノートブックでスイープを設定する場合、
configオブジェクトはネストされた辞書です。
コマンドラインからスイープを実行する場合は、
そのconfigオブジェクトは
[YAMLファイル](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)です。

スイープのconfig定義を一緒に見ていきましょう。
各コンポーネントを説明する時間を取るため、
ゆっくり進めます。
通常のスイープ開発フローでは、
このステップは1回の課題として実施されます。

### 👈 `method`を選ぶ

新しいパラメータ値を選ぶために最初に定義する必要があるのは、`method`です。

次の検索`methods`を提供します：
*   **`grid`検索** – ハイパーパラメータの全ての組み合わせを繰り返す。
非常に効果的ですが、計算コストが高い場合があります。
*   **`random`検索** – 提供された`distribution`に従って新しい組み合わせをランダムに選択します。驚くほど効果的です！
*   **`bayes`ian検索** – ハイパーパラメータの関数としてメトリックスコアの確率的モデルを作成し、メトリクスを改善する確率が高いパラメータを選択します。連続パラメータが少ない場合に効果的ですが、スケールには課題があります。

ここでは`random`を使います。

```python
sweep_config = {
    'method': 'random'
    }
```

`bayes`ianスイープの場合は、
`metric`情報を少し提供する必要があります。
その`name`を教えていただければ
モデル出力の中から見つけやすくなり、
`goal`が`minimize`（例：二乗誤差の場合）
または`maximize`（例：正確度）のどちらなのかも知っておく必要があります。

```python
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

`bayes`ianスイープを実行していなくても、
後で考えが変わることもあるので、
この情報を`sweep_config`に含めておくと良いでしょう。
また、こういった情報を保存しておくことは再現性の観点からも良い実務です。
6ヶ月後や6年後に誰かがそのスイープを再度見るときに、
`val_G_batch`が高い方が良いのか低い方が良いのかを知っておく必要があります。

### 📃 ハイパー`パラメータ`に名前をつける

新しいハイパーパラメータ値の`method`を選んだ後、
それらの`parameters`が何であるかを定義する必要があります。

ほとんどの場合、このステップは簡単です：
`parameter`に名前を付けて、
そのパラメータの有効な`values`リストを指定します。

たとえば、ニューラルネットワークの`optimizer`を選ぶ場合、
オプションは限られています。
ここでは、最も人気のある選択肢である`adam`と`sgd`の2つに絞ります。
潜在的な無限の選択肢があるハイパーパラメータについても、
通常は重要な`values`のサブセットしか試す意味がありません。
ここではhidden`layer_size`や`dropout`についていくつかの選択肢を挙げています。

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

このスイープで変更したくないハイパーパラメータが存在する場合もありますが、
それでもその`sweep_config`に設定しておきたい場合があります。

その場合は、`value`を直接設定します：

```python
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`grid`探索の場合、それだけで必要な全てです。

`random`探索の場合、
あるパラメータのすべての`values`が特定のrunで等しく選ばれる可能性があります。

それでは不十分な場合、
名前付きの`distribution`とそのパラメータ（例：正規分布の平均`mu`と標準偏差`sigma`）を指定できます。

ランダム変数の分布を設定する方法についてさらに詳しくは[こちら](https://docs.wandb.com/sweeps/configuration#distributions)を参照してください。

```python
parameters_dict.update({
    'learning_rate': {
        # 0から0.1の範囲の一様分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32から256までの整数
        # 対数が均一に分布している
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

終了すると、`sweep_config`はネストされた辞書となり、
試してみたい正確な`parameters`と
それを試すための`method`が指定されます。

```python
import pprint

pprint.pprint(sweep_config)
```

しかし、これはすべての設定オプションではありません！

例えば、[HyperBand](https://arxiv.org/pdf/1603.06560.pdf)スケジューリングアルゴリズムでrunを早期終了させるオプションも提供しています。詳細は[こちら](https://docs.wandb.com/sweeps/configuration#stopping-criteria)を参照してください。

すべての設定オプションのリストは[こちら](https://docs.wandb.com/library/sweeps/configuration)にあり、
YAML形式の例の大集が[こちら](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)にあります。

# ステップ 2️⃣. スイープを初期化する

検索戦略を定義したら、それを実行するための設定を開始します。

スイープの時計仕掛けの管理者は _Sweep Controller_ として知られています。
各runが完了すると、新しいrunの指示が発行されます。
これらの指示は、実際にrunを実行する _agents_ によって拾われます。

通常のスイープでは、コントローラは _我々の_ マシン上にあり、
runを実行するagentsは _あなたの_ マシン上にあります。
この労働分担により、agentsを実行するマシンを追加するだけでスイープをスケールアップするのが非常に簡単になります！

<img src="https://i.imgur.com/zlbw3vQ.png" alt="sweeps-diagram" width="500"/>

適切な`sweep_config`と`project`名で`wandb.sweep`を呼び出すことで、スイープコントローラを巻き上げることができます。

この関数は、後でこのコントローラにagentsを割り当てるための`sweep_id`を返します。

> _サイドノート_：コマンドラインでは、この関数は
```python
wandb sweep config.yaml
```
に置き換わります。
[コマンドラインでのスイープの使用方法について詳しくはこちら ➡](https://docs.wandb.ai/guides/sweeps/walkthrough)

```python
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

# ステップ 3️⃣. スイープエージェントの実行

### 💻 トレーニング手順を定義する

実際にスイープを実行する前に、
それらの値を使用するトレーニング手順を定義する必要があります。

以下の関数では、PyTorchで単純な全結合ニューラルネットワークを定義し、
次の`wandb`ツールを追加してモデルのメトリクスをログし、パフォーマンスと出力を可視化し、実験を追跡します：
* [**`wandb.init()`**](https://docs.wandb.com/library/init) – 新しいW&B Runを初期化します。それぞれのRunはトレーニング関数の単一実行です。
* [**`wandb.config`**](https://docs.wandb.com/library/config) – すべてのハイパーパラメータを設定オブジェクトに保存し、ログに記録できます。`wandb.config`の使用方法について詳しくは[こちら](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb)を参照してください。
* [**`wandb.log()`**](https://docs.wandb.com/library/log) – モデルの振る舞いをW&Bにログします。ここではパフォーマンスのみをログしていますが、`wandb.log`でログできるその他のリッチメディアについては[こちらのColab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb)を参照してください。

PyTorchとW&Bの連携について詳細は[こちらのColab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)を参照してください。

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
        # 下記のようにwandb.agentによって呼び出された場合、
        # このconfigはスイープコントローラによって設定されます
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

このセルではトレーニング手順の4つのパート
`build_dataset`、`build_network`、`build_optimizer`、`train_epoch`
を定義します。

これらは基本的なPyTorchの開発フローの標準的な一部であり、
W&Bの使用による影響はありませんので、コメントは省略します。

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

        # ⬅ Backward pass + 重みの更新
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

さあ、スイープを始める準備が整いました！ 🧹🧹🧹

スイープコントローラは、`wandb.sweep`を実行することで作成されますが、
実行する`config`を誰かが尋ねるのを待っています。

その誰かが`agent`であり、`wandb.agent`で作成されます。
開始するために`agent`が知っておくべきことは
1. どのスイープに属しているか (`sweep_id`)
2. 実行すべき関数（ここでは`train`）
3. (オプション) コントローラから取得するconfig数 (`count`)

ちなみに、同じ`sweep_id`で複数の`agent`を異なる計算リソース上で開始し、
コントローラはこれらが`sweep_config`で定められた戦略に従って共同で動作するようにします。
これにより、スイープを多くのノードに簡単にスケーリングできます！

> _サイドノート:_ コマンドラインでは、この関数は
```bash
wandb agent sweep_id
```
に置き換わります。
[コマンドラインでのスイープの使用方法について詳しくはこちら ➡](https://docs.wandb.com/sweeps/walkthrough)

以下のセルは、スイープコントローラから返されるランダムに生成されたハイパーパラメータ値を使用して、`train`を5回実行するための`agent`を起動します。実行には5分以内で完了します。

```python
wandb.agent(sweep_id, train, count=5)
```

# 👀 スイープ結果の可視化

## 🔀 パラレル座標プロット
このプロットは、ハイパーパラメータの値をモデルのメトリクスにマッピングします。最良のモデルパフォーマンスをもたらしたハイパーパラメータの組み合わせを特定するのに役立ちます。

![](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5e190366778ad831455f9