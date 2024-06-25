
# ハイパーパラメーターのチューニング

[**ここでColab Notebookを試す →**](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb)

最もパフォーマンスの高いモデルを見つけるために高次元のハイパーパラメータ空間を探索することは、非常に手間がかかります。ハイパーパラメータ探索は、モデルのバトルロワイヤルを整理して効率的に実施し、最も正確なモデルを選ぶ方法を提供します。これにより、学習率、バッチサイズ、隠れ層の数、オプティマイザーの種類などのハイパーパラメータの組み合わせを自動的に検索して最適な値を見つけることができます。

このチュートリアルでは、Weights & Biasesを使用して3つの簡単なステップで高度なハイパーパラメータ探索を実行する方法を見ていきます。

### [ビデオチュートリアル](http://wandb.me/sweeps-video)に従って進めましょう！

![](https://i.imgur.com/WVKkMWw.png)

# 🚀 セットアップ

まず、実験管理ライブラリをインストールし、無料のW&Bアカウントを設定します。

1. `!pip install`でインストール
2. ライブラリをPythonに`import`
3. `.login()`してプロジェクトにメトリクスをログ

Weights & Biasesを初めて使用する場合、`login`の呼び出しでアカウント登録のリンクが表示されます。W&Bは個人および学術プロジェクトで無料で使用できます！


```python
!pip install wandb -Uq
```


```python
import wandb

wandb.login()
```

# ステップ 1️⃣. Sweepの定義

根本的に、Sweepは多数のハイパーパラメーター値を試すための戦略とそれを評価するコードを組み合わせたものです。
あなたはその戦略を[設定](https://docs.wandb.com/sweeps/configuration)の形で定義するだけです。

ノートブックのような環境でSweepを設定する場合、この設定オブジェクトはネストされた辞書です。
コマンドラインでSweepを実行する場合、設定オブジェクトは[YAMLファイル](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)です。

一緒にSweep設定の定義を見ていきましょう。各コンポーネントを説明する機会を得るために、ゆっくり進めます。
典型的なSweepパイプラインでは、このステップは単一の割り当てで行われます。

### 👈 `method`を選択

最初に定義する必要があるのは、新しいパラメータ値を選択する`method`です。

以下の探索`method`を提供しています：
* **`grid` Search** – すべてのハイパーパラメータ値の組み合わせを反復処理します。非常に効果的ですが、計算コストがかかる場合があります。
* **`random` Search** – 提供された`distribution`に従ってランダムに新しい組み合わせを選択します。驚くほど効果的です！
* **`bayes`ian Search** – ハイパーパラメータの関数としてメトリックスコアの確率モデルを作成し、そのメトリックを改善する可能性の高いパラメータを選択します。少数の連続パラメータに対してはよく機能しますが、スケールが難しいです。

今回は`random`を使用します。


```python
sweep_config = {
    'method': 'random'
    }
```

`bayes`ian Sweepの場合には、`metric`についても少し教えてください。それがモデルの出力に見つけられるように名前が必要ですし、そのメトリックを`minimize`（例えば、二乗誤差の場合）するのか、`maximize`（例えば、精度の場合）するのか`goal`も必要です。


```python
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

`bayes`ian Sweepを実行していない場合でも、後で変更することを考えて`sweep_config`にこれを含めることをお勧めします。また、再現性のためにこのようなことをメモしておくことも良い練習です。6か月や6年後に他の誰かがSweepを見て、`val_G_batch`が高い方が良いのか低い方が良いのかわからない時のために。

### 📃 ハイパーパラメータに名前を付ける

新しいハイパーパラメータ値を試す方法を選択したら、次にそのパラメータが何であるかを定義する必要があります。

ほとんどの場合、このステップは簡単です：
パラメータに名前を付け、そのパラメータの合法な値のリストを指定します。

例えば、ネットワークに対する`optimizer`を選ぶときには、選択肢は有限です。ここでは、最も人気のある2つの選択肢、`adam`と`sgd`を使用します。無限の選択肢があるハイパーパラメータでも、実際に試すのはいくつかの選択肢だけです。ここでは難易度`layer_size`と`dropout`で選択肢をいくつか指定しています。


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

しばしば、Sweepでは変更したくないハイパーパラメータがある場合でも、`sweep_config`で設定したい場合があります。

その場合、直接`value`を設定します。


```python
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

`grid`探索の場合、必要なのはそれだけです。

`random`探索の場合、パラメータの全ての`values`は各runで同じ確率で選ばれる可能性があります。

それではうまくいかない場合は、代わりに名前付きの`distribution`とそのパラメータ（`normal`分布の平均`mu`や標準偏差`sigma`など）を指定できます。

ランダム変数の分布の設定方法については[こちら](https://docs.wandb.com/sweeps/configuration#distributions)をご覧ください。


```python
parameters_dict.update({
    'learning_rate': {
        # 0から0.1までの平坦な分布
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # 32から256までの整数
        # 対数が均等に分布するもの
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

終了すると、`sweep_config`はネストされた辞書となり、どの`parameters`を試すか、どの`method`を使用するかを正確に指定します。


```python
import pprint

pprint.pprint(sweep_config)
```

しかし、これは設定オプションの全てではありません！

例えば、[HyperBand](https://arxiv.org/pdf/1603.06560.pdf)スケジューリングアルゴリズムを使用してrunを早期終了するオプションも提供しています。詳細は[こちら](https://docs.wandb.com/sweeps/configuration#stopping-criteria)をご覧ください。

全ての設定オプションのリストは[こちら](https://docs.wandb.com/library/sweeps/configuration)に、YAML形式の例の大集合は[こちら](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)にあります。



# ステップ 2️⃣. Sweepの初期化

検索戦略を定義したら、それを実行するための設定を行う時です。

Sweepの核となるタイムキーパーは _Sweep Controller_ として知られています。
各runが完了するたびに、新しいrunを実行するための新しい指示を提供します。
これらの指示は、実際にrunを実行する _agents_ によって取得されます。

典型的なSweepでは、Controllerは_私たちの_マシン上で動作し、runを完了するagentsは_あなたの_マシン（例えば下図のように）で動作します。この労働の分業により、agentsを実行するためのマシンを追加することでSweepを簡単にスケールアップできます！

<img src="https://i.imgur.com/zlbw3vQ.png" alt="sweeps-diagram" width="500"/>

適切な`sweep_config`と`project`名を使用して`wandb.sweep`を呼び出すことで、Sweep Controllerを立ち上げます。

この関数は、後でagentをこのControllerに割り当てるために使用する`sweep_id`を返します。

> _サイドノート_: コマンドラインでは、この関数は次のように置き換えられます
```python
wandb sweep config.yaml
```
[Sweepsをコマンドラインで使用する方法 ➡](https://docs.wandb.ai/guides/sweeps/walkthrough)を学ぶ


```python
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

# ステップ 3️⃣. Sweepエージェントの実行

### 💻 トレーニング手順の定義

実際にSweepを実行する前に、それらの値を使用するトレーニング手順を定義する必要があります。

以下の関数では、PyTorchでシンプルな全結合ニューラルネットワークを定義し、次の`wandb`ツールを追加してモデルメトリクスのログ、パフォーマンスの可視化、および実験管理を行います：
* [**`wandb.init()`**](https://docs.wandb.com/library/init) – 新しいW&B Runを初期化します。各Runはトレーニング関数の単一の実行です。
* [**`wandb.config`**](https://docs.wandb.com/library/config) – 全てのハイパーパラメータを設定オブジェクトに保存します。`wandb.config`の使用方法については[こちら](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb)で詳細を読みます。
* [**`wandb.log()`**](https://docs.wandb.com/library/log) – モデルの振る舞いをW&Bにログします。ここではパフォーマンスだけをログします。`wandb.log`でログできる他の豊富なメディアについては[このColab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb)をご覧ください。

PyTorchとのW&Bインテグレーションの詳細については[このColab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb)を参照してください。


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
        # wandb.agentが呼び出した場合、
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
`build_dataset`、`build_network`、`build_optimizer`、および`train_epoch`。

これらはすべて基本的なPyTorchパイプラインの標準的な部分であり、W&Bの使用によってその実装が影響されることはないので、コメントは控えます。


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

        # ➡ フォワードパス
        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        # ⬅ バックワードパス + 重みの更新
        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

さあ、Sweepを始める準備が整いました！🧹🧹🧹

Sweep Controllersは、`wandb.sweep`を実行して作成されたものとして、新しい`config`を試すように待っています。

その要求者は`agent`であり、`wandb.agent`で作成されます。
エージェントを起動するには
1. どのSweepの一部であるか（`sweep_id`）
2. 実行する関数（ここでは`train`）
3. （任意で）Controllerに何個のconfigを求めるか（`count`）

情報として、同じ`sweep_id`で複数の`agent`を異なるコンピューティングリソースで開始することができ、Controllerはそれらが`sweep_config`で定められた戦略に従って協力することを保証します。
これにより、多くのノードにSweepを簡単にスケールアップすることができます！

> _サイドノート_: コマンドラインでは、この関数は次のように置き換えられます
```bash
wandb agent sweep_id
```
[Sweepsをコマンドラインで使用する方法 ➡](https://docs.wandb.com/sweeps/walkthrough)を学ぶ

下のセルは、Sweep Controllerによってランダムに生成されたハイパーパラメータ値を使用して、`train`を5回実行する`agent`を起動します。実行には5分未満しかかかりません。


```python
wandb.agent(sweep_id, train, count=5)
```

# 👀 Sweep結果を可視化する



## 🔀 パラレルコーディネートプロット
このプロットはハイパーパラメータ値をモデルメトリクスにマッピングします。最も優れたモデルパフォーマンスに繋がったハイパーパラメータの組み合わせを絞り込むのに役立ちます。

![](https://assets.website-files.com/5ac6b7f2924c652fd013a891/5e190366778ad831455f9af2_s_194708415DEC35F74A7691FF6810D3B14703D1EFE1672ED29000BA98171242A5_1578695138341_image.png)


## 📊 ハイパーパラメータ重要性プロット
ハイパーパラメータ重要性プロットは、メトリクスの最良の予測因子となったハイパーパラメータを浮き彫